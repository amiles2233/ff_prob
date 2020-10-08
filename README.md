# Fantasy Football with Probabilistic Modeling (Using Tensorflow Probability)

## Intro 

Sports, and especially sports with relatively few events, are rife with uncertainty. Most ML models, however, make point predictions (game line of x, spread line of y, etc.). Naturally, this can be confusing to non-technical folks unfamiliar with uncertainty. That is where tensorflow probability comes in. With this tool, rather than modeling single points, we can model distributions that maximize the likelihood of those points. 

The advantages of this are at least twofold:

1. Given that football is a sport with relatively few events, we have less of a risk of overfitting a small n  
2. We can look at the uncertainty and skew of possible outcomes.

The context in which I'll be working is with gambling and daily fantasy football. Specifically, I'm going to be modeling game outcomes that can inform betting on spread and over/unders. 

Listen to any NFL DFS podcast, and they will talk about game scripts, or game context. Essentially, games between two high scoring teams is likely to be beneficial for the offensive players on those teams, etc. Some people use betting lines to get context on which game contexts are likely to produce the highest scoring players. 

The benefit of modeling game outcome *distributions* is that we can now simulate likely game contexts. For example, a sportsbook will set a line at o/u 49, which would be a reasonably high scoring game, but how certain are we that the game will *actually* be high scoring? Our distributions can answer that question. Given a line x, what is the distribution of possible score outcomes.

At the player level, if we use game context as a predictor in our model, we can plug in these game context scenarios to get possible player outcomes. For example, in a game where 70 points is scored, offensive players are going to score big, and vice versa in a game where 7 points is scored. 

This is explicitly a top-down approach, I'm situating players within games, and not building games as the sum of player performance. My reasoning is that predictions are going to be more stable at the game level, and I get to leverage vegas lines (you know, the people who actually put their money where their mouth is). Also, there are a lot of ways to get to a, say, 32-14 game. Combine that projected outcome with player and team features, and you get a scenario to place player predictions within. For example, in a hypothetical game where a team scores 32 and their opponent 14, the ceiling for offensive players is much higher than a 6-14 game. And using the same scenario for a whole team helps reduce inconsistent results (e.g. predicting a reciever to have a big game while predicting the QB to throw a dud)

## Steps (Intuition)

1. Model: Given the betting lines and team history, what is the likely distribution of total points scored and and spread

2. Model: Given the game outcome, team and player history, what is the likely distribution of points scored for a player?

3. Predict: Distrubiton of game scenarios given the betting lines and team history

    a. Use predictions to evaluate spread lines and over/unders
    
    b. Simulate: N pulls from predicting game distribution

4. Predict: Predict player performance given hypothetical game outcome, team and player history.

### Modeling Approach  

![](https://drive.google.com/file/d/1zaqKx_9yzFDL_y1t9Y_1mUr0jF38_5Jn/view?usp=sharing)  

### Prediction Approach  

![](https://drive.google.com/file/d/1OHwmQFOqzAij2IElyLdBv-bVTBlLpDqJ/view?usp=sharing)  

## Steps (Code)

1. Model Game Outcomes: `model_eval_spread_ou_tfp.R`
2. Model Player Outcomes: coming soon...
3. Predict Games Weekly: coming soon...
4. Predict Player Performance Weekly: coming soon...



