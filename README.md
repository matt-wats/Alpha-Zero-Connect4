# Alpha-Zero-Connect4
Using the Alpha Zero algorithm to learn to play Connect 4

# DEMO

Video of me playing, and losing :(, against the bot:

https://github.com/matt-wats/Alpha-Zero-Connect4/assets/112960646/09432674-ddf6-4b01-a014-a551206d69fa



# Motivation

I have always liked board games and strategy games. 
I wanted to create a program that could learn to play games, like Connect 4.

I have already tried to do this (at least) twice before: https://github.com/matt-wats/Learning-Connect-4, https://openprocessing.org/sketch/911295,
but now I know more things and am ready to give it another go!


# Method

I used the Alpha Zero algorithm to train an agent to play connect 4.
How the alrogithm works (loosely):

- An agent plays a series of games against itself
- The agent has two abilites: assigning a value to a given position, and assigning probabilities to what actions will be taken from a position
- The agent's predicted value is trained on what the outcome of the game is
- Each turn, it conducts a Monte Carlo Tree Search using the action probabilites and state values to guide its search
- The action probabilities are trained on the new probabilities defined by the search (how often a node is looked at)
- During training, the moves are chosen according to the search probabilities

In this experiment, the agent is a ResNet with 3 residual blocks.


# Results

Here are the epoch losses during training:
![Epoch Losses](/images/losses.png "Plot of Losses per Epoch")

## Interesting Quirks of the Bot
(yes I am going to personify the bot because I think it's funny)

- Sometimes if it has a forced win, it will delay playing the winning move for a couple of moves (presumably) because the outcome is the same
- Sometimes it will get into a position it isn't familiar with, and just throw the game because it's confused
- It is not symmetric. It favors one side over the other, even if the position itself is symmetric

## Awesome
As I am sure everybody wants to know: YES, this bot does beat my previous attempts. It even beats me sometimes :'(. So I would call
this project an awesome succes!

# Future Improvements

- A larger agent model (I redid the experiment with more games, but with such a small model this causes it to forget basic positions)
- Take advantage of symmetry during training
- Train multiple models at once to create an ensemble method player
