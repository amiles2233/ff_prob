library(tidyverse)
library(recipes)
library(slider)
library(rsample)
library(recipes)
library(tensorflow)
library(tfprobability)
library(keras)
library(scales)
library(labeling)

reticulate::use_condaenv('tf2gpu', required = TRUE)


# Lee Sharpe's Game Data
game = read_csv("https://raw.github.com/leesharpe/nfldata/master/data/games.csv")

## Pivot to Long
game_feature <- game %>%
  filter(game_type=="REG", !is.na(home_score)) %>%
  select(game_id, season, week, weekday, spread_line, result, total_line, total, home_team, away_team, home_score, away_score, home_rest, away_rest) %>%
  pivot_longer(cols = home_team:away_rest,
               names_to = c("home_away", ".value"),
               names_pattern = "(.+)_(.+)") %>%
  ## adjust spred and results for away teams
  mutate(spread_line = ifelse(home_away=='away', -spread_line, spread_line),
         result = ifelse(home_away=='away', -result, result),
         allow = total-score,
         ats = as.numeric(result>(-spread_line)),
         over = as.numeric(total>total_line))  %>%
  arrange(season, team, week) %>%
  group_by(season, team) %>%
  ## 4 week sliding windows of % against the spread, over, pts scored and allowed
  mutate(ats_4wk = slide_dbl(ats, mean, .before = 4,  .after = -1, .complete = TRUE),
         over_4wk = slide_dbl(over, mean, .before = 4,  .after = -1, .complete = TRUE),
         pts_scored_4wk_mean = slide_dbl(score, mean, .before = 4, .after = -1, .complete = TRUE),
         pts_allow_4wk_mean = slide_dbl(allow, mean, .before = 4, .after = -1, .complete = TRUE),
         pts_scored_4wk_std_dev = slide_dbl(score, sd, .before = 4, .after = -1, .complete = TRUE),
         pts_allowed_4wk_std_dev = slide_dbl(allow, sd, .before = 4, .after = -1, .complete = TRUE)) %>%
  ungroup() %>%
  arrange(game_id) %>%
  ## after week 5, every team has at least 4 games played
  filter(week>5) %>%
  ## widen by home_away to get single row per game_id
  pivot_wider(id_cols = game_id, names_from = home_away, values_from = ats_4wk:pts_allowed_4wk_std_dev) %>%
  ## join back to original game df
  inner_join(game, by='game_id') %>%
  select(season, result, total, spread_line, total_line, div_game, roof, surface, week, weekday, home_rest, away_rest, ats_4wk_away:pts_allowed_4wk_std_dev_home) %>%
  ## treat week as a categorical var
   mutate(week = as.character(week))

## Pre-2018 for Train/Valid
game_feature_pre_18 <- game_feature[game_feature$season<=2018, -1]

## 2019/20 YTD for Test
game_feature_19_20 <- game_feature[game_feature$season>=2019, -1]


 ## Train/Test Split
split <- initial_split(game_feature_pre_18, prop = 3/5)

labels = c("result", "total")
feature_formula = reformulate(termlabels = names(game_feature)[3:length(game_feature)])

## Create Recipe
recipe_obj <- recipe(feature_formula, data = game_feature) %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) %>%
  step_other(all_nominal(), threshold = .05) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  prep(training = training(split), retain = FALSE)

x_train <- bake(recipe_obj, new_data = training(split), composition = "matrix")
x_valid <- bake(recipe_obj, new_data = testing(split), composition = "matrix")
x_test <- bake(recipe_obj, new_data = game_feature_19_20, composition = "matrix")

y_spread_train <- training(split) %>% .$result
y_points_train <- training(split) %>% .$total

y_spread_valid <- testing(split) %>% .$result
y_points_valid <- testing(split) %>% .$total

y_spread_test <- game_feature_19_20$result
y_points_test <- game_feature_19_20$total

## Specify Model
input_layers <- layer_input(shape = ncol(x_train), name = 'input_layers')

main_layers <- input_layers %>%
  #layer_dense(units = 8, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units = 16, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units = 32, activation='relu', regularizer_l1_l2()) %>%
  layer_dense(units = 64, activation='relu', regularizer_l1_l2()) %>%
  layer_dense(units = 128, activation='relu', regularizer_l1_l2()) %>%
  layer_dense(units = 64, activation='relu', regularizer_l1_l2()) %>%
  layer_dense(units = 32, activation='relu', regularizer_l1_l2()) %>%
  layer_dense(units = 16, activation='relu', regularizer_l1_l2()) 

spread_output <- main_layers %>%
  layer_dense(units = 16, activation='relu') %>%
  layer_dense(units = 4, activation = 'linear') %>%
  layer_distribution_lambda(function(x) {
    tfd_sinh_arcsinh(loc = x[, 1, drop = FALSE],
                     scale = 1e-3 + tf$math$softplus(x[, 2, drop = FALSE]),
                     skewness=x[, 3, drop=FALSE],
                     tailweight= 1e-3 + tf$math$softplus(x[, 4, drop = FALSE]))
  }, 
  name = "spread_output"
  )
  
points_output <- main_layers %>%
  layer_dense(units = 16, activation='relu') %>%
  layer_dense(units = 4, activation = 'linear') %>%
  layer_distribution_lambda(function(x) {
    tfd_sinh_arcsinh(loc = x[, 1, drop = FALSE],
                     scale = 1e-3 + tf$math$softplus(x[, 2, drop = FALSE]),
                     skewness=x[, 3, drop=FALSE],
                     tailweight= 1e-3 + tf$math$softplus(x[, 4, drop = FALSE]))
  }, 
  name = "points_output"
  )

model <- keras_model(
  inputs = input_layers,
  outputs = list(spread_output, points_output)
)

negloglik <- function(y, model) - (model %>% tfd_log_prob(y))

learning_rate <- 0.001

model %>% compile(optimizer = optimizer_nadam(lr = learning_rate), loss = negloglik)

history <- model %>% fit(x=list(x_train), 
                         y=list(spread_output = y_spread_train, points_output = y_points_train),
                         shuffle=TRUE,
                         #validation_split = .4,
                         validation_data = list(x_valid, list(spread_output = y_spread_valid, points_output = y_points_valid)),
                         epochs = 200, 
                         batch_size=32, 
                         callbacks=list(callback_early_stopping(monitor='val_loss', patience = 20))

)


pred_dist <- model(list(tf$constant(x_test)))

pred_dist_spread <- pred_dist[[1]]
pred_dist_points <- pred_dist[[2]]

get_tf_cdf = function(loc, scale, skewness, tailweight, pred){
  1 - as.numeric(tfd_cdf(
    distribution = tfd_sinh_arcsinh(loc=loc, scale=scale, skewness=skewness, tailweight=tailweight),
    value = pred
  ))
}

## make get_cdf more generic
## get_gamma_quantile fun

out_spread <- tibble(
  
  lv_spread = game_feature_19_20$spread_line,
  
  actual_spread = game_feature_19_20$result,
  
  loc = pred_dist_spread$loc %>% as.numeric(),
  scale = pred_dist_spread$scale %>% as.numeric(),
  skewness = pred_dist_spread$skewness %>% as.numeric(),
  tailweight = pred_dist_spread$tailweight %>% as.numeric(),
  
  prob_lv_spread = get_tf_cdf(loc=loc, scale=scale, skewness = skewness, tailweight = tailweight, pred=-lv_spread),
  prob_actual_spread = get_tf_cdf(loc=loc, scale=scale, skewness = skewness, tailweight = tailweight, pred=actual_spread),
  
  quant10 = pred_dist_spread$quantile(.1) %>% as.numeric(),
  quant25 = pred_dist_spread$quantile(.25) %>% as.numeric(),
  quant50 = pred_dist_spread$quantile(.5) %>% as.numeric(),
  quant75 = pred_dist_spread$quantile(.75) %>% as.numeric(),
  quant90 = pred_dist_spread$quantile(.9) %>% as.numeric(),
  
  up10 = actual_spread>quant10,
  up25 = actual_spread>quant25,
  up50 = actual_spread>quant50,
  up75 = actual_spread>quant75,
  up90 = actual_spread>quant90, 
  
  cover = as.numeric(lv_spread+actual_spread>0)

  
) 


#View(out_spread)

out_points <- tibble(
  
  lv_over_under = game_feature_19_20$total_line,
  
  actual_points = game_feature_19_20$total,
  
  loc = pred_dist_points$loc %>% as.numeric(),
  scale = pred_dist_points$scale %>% as.numeric(),
  skewness = pred_dist_points$skewness %>% as.numeric(),
  tailweight = pred_dist_points$tailweight %>% as.numeric(),
  
  prob_lv_over_under = get_tf_cdf(loc=loc, scale=scale, skewness = skewness, tailweight = tailweight, pred=lv_over_under),
  prob_actual_points = get_tf_cdf(loc=loc, scale=scale, skewness = skewness, tailweight = tailweight, pred=actual_points),
  
  quant10 = pred_dist_points$quantile(.1) %>% as.numeric(),
  quant25 = pred_dist_points$quantile(.25) %>% as.numeric(),
  quant50 = pred_dist_points$quantile(.5) %>% as.numeric(),
  quant75 = pred_dist_points$quantile(.75) %>% as.numeric(),
  quant90 = pred_dist_points$quantile(.9) %>% as.numeric(),
  
  up10 = actual_points>quant10,
  up25 = actual_points>quant25,
  up50 = actual_points>quant50,
  up75 = actual_points>quant75,
  up90 = actual_points>quant90,
  
  over = as.numeric(actual_points>lv_over_under)
) 

#View(out_points)

## Spread Accuracy Plots
out_spread %>%
  select(up10:up90) %>% 
  summarize_each(funs = mean) %>%
  pivot_longer(up10:up90, names_to = 'over_quantile', values_to = 'pct') %>%
  ggplot(aes(x=over_quantile, y=pct, label=scales::percent(pct, accuracy = .01))) +
  geom_col() +
  geom_text(vjust=1, color='white') +
  theme_minimal() +
  scale_y_continuous(labels=scales::percent) +
  ggtitle('% Games going above spread quantiles')


## Distribution vs Expected Uniform Distribution
out_spread %>%
  ggplot(aes(x=prob_actual_spread)) +
  geom_density(fill = 'darkgreen') +
  geom_density(data = enframe(as.numeric(tfd_sample(tfd_uniform(), 10000))), 
               mapping = aes(x=value), 
               color='black', linetype='dashed', fill='white', alpha=.3) +
  theme_dark() +
  ggtitle('CDF of Actual Spreads (Home-Away)',
          subtitle = 'White Distribution is Theoretical Uniform Distribution (Expected in Long Run)')

## Variation by Quantile Range
out_spread %>%
  mutate(range_80 = quant90-quant10,
         range_50 = quant75-quant25) %>%
  select(actual_spread, range_80, range_50) %>%
  pivot_longer(range_80:range_50, names_to = 'range_type', values_to = 'range_val') %>%
  ggplot(aes(x=range_val, y=actual_spread, color=range_type, fill=range_type)) +
  geom_point(alpha=.1) +
  geom_smooth() +
  facet_wrap(~range_type, scales = 'free') +
  #scale_y_log10() +
  #coord_cartesian(xlim = c(0, .1)) +
  theme_dark() +
  ylab('Actual Spread (Log Scale') +
  xlab('Quantile Range') +
  ggtitle('Spread Variation by Quantile Range')


# 50% Classification Threshold
out_spread %>%
  ggplot(aes(x=quant50, y=actual_spread)) +
  geom_point(color='green') +
  geom_smooth() +
  theme_dark() +
  ggtitle('50th Percentile vs Actual Spread')

# Confusion Matrix at 50% Cutoff
out_spread %>%
  mutate(predicted = ifelse(prob_lv_spread>.5, "Home Cover", "Away Cover"),
         actual = ifelse(cover==1, "Home Cover", "Away Cover")) %>%
  select(predicted, actual) %>%
  group_by(predicted, actual) %>%
  summarize(n=n(), .groups = 'keep') %>%
  ungroup(actual) %>%
  mutate(n_pred = sum(n),
         pct_pred_correct = n/n_pred) %>%
  ggplot(aes(x=predicted, y=actual, fill=pct_pred_correct, label=n)) +
  geom_tile() +
  geom_text() +
  scale_fill_distiller(palette = "PRGn", direction = 1) +
  theme_dark()

# Cover Probabilities
out_spread %>%
  mutate(prob_lv_spread_cut = cut(prob_lv_spread, seq(from=0, to=1, by=.1))) %>%
  group_by(prob_lv_spread_cut) %>%
  summarize(pct_covered = mean(cover)) %>%
  ggplot(aes(x=prob_lv_spread_cut, y=pct_covered, label = percent(pct_covered, accuracy = .1))) +
  geom_col(fill='green') +
  geom_text(vjust=1) +
  theme_dark() +
  scale_y_continuous(labels = percent) +
  ylab("Pct Home Teams Covering the Spread") +
  xlab("Probability of Home Team Covering the Spread (10 Groups)") +
  ggtitle('% Home Teams Covering by CDF Decile')



## Over/Under Model

## Points Accuracy Plots

# 
out_points %>%
  select(up10:up90) %>% 
  summarize_each(funs = mean) %>%
  pivot_longer(up10:up90, names_to = 'over_quantile', values_to = 'pct') %>%
  ggplot(aes(x=over_quantile, y=pct, label=scales::percent(pct, accuracy = .01))) +
  geom_col(fill='green') +
  geom_text(vjust=1) +
  theme_dark() +
  scale_y_continuous(labels=scales::percent) +
  ggtitle('% Games Going Above O/U Quantiles')


## Distribution vs Expected Uniform Distribution
out_points %>%
  ggplot(aes(x=prob_actual_points)) +
  geom_density(fill = 'green') +
  geom_density(data = enframe(as.numeric(tfd_sample(tfd_uniform(), 10000))), 
               mapping = aes(x=value), 
               color='black', linetype='dashed', fill='white', alpha=.3) +
  theme_dark() +
  ggtitle('CDF of Actual Points (Home-Away)',
          subtitle = 'White Distribution is Theoretical Uniform Distribution (Expected in Long Run)')

## Variation by Quantile Range
out_points %>%
  mutate(range_80 = quant90-quant10,
         range_50 = quant75-quant25) %>%
  select(actual_points, range_80, range_50) %>%
  pivot_longer(range_80:range_50, names_to = 'range_type', values_to = 'range_val') %>%
  ggplot(aes(x=range_val, y=actual_points, color=range_type, fill=range_type)) +
  geom_point(alpha=.1) +
  geom_smooth() +
  facet_wrap(~range_type, scales = 'free') +
  #scale_y_log10() +
  #coord_cartesian(xlim = c(0, .1)) +
  theme_dark() +
  ylab('Actual Points (Log Scale') +
  xlab('Quantile Range') +
  ggtitle('Point Variation by Quantile Range')


# 50% Classification Threshold
out_points %>%
  ggplot(aes(x=quant50, y=actual_points)) +
  geom_point(color='green') +
  geom_smooth() +
  theme_dark() +
  ggtitle('50th Percentile vs Actual Points')

# Cover Probabilities
out_points %>%
  mutate(prob_lv_points_cut = cut(prob_lv_over_under, c(0, .25, .5, .75, 1))) %>%
  group_by(prob_lv_points_cut) %>%
  summarize(pct_over = mean(over),
            group = n()) %>%
  ggplot(aes(x=prob_lv_points_cut, y=pct_over, label = percent(pct_over))) +
  geom_col(fill='green') +
  geom_text(vjust=1) +
  theme_dark() +
  scale_y_continuous(labels = percent) +
  ylab("Pct Home Teams Covering the Spread") +
  xlab("Probability of Home Team Covering the Spread (10 Groups)") +
  ggtitle('% Home Teams Covering by CDF Decile')

# Confusion Matrix at 50% Cutoff
out_points %>%
  mutate(predicted = ifelse(prob_lv_over_under>.5, "Over", "Under"),
         actual = ifelse(over==1, "Over", "Under")) %>%
  select(predicted, actual) %>%
  group_by(predicted, actual) %>%
  summarize(n=n(), .groups = 'keep') %>%
  ungroup(actual) %>%
  mutate(n_pred = sum(n),
         pct_pred_correct = n/n_pred) %>%
  ggplot(aes(x=predicted, y=actual, fill=pct_pred_correct, label=n)) +
  geom_tile() +
  geom_text() +
  scale_fill_distiller(palette = "PRGn", direction = 1) +
  theme_dark()



