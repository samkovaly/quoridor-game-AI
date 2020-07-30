# Quoridor Game - Reinforcement Learning
This repository achieves three goals:
1. Implements the game of Quoridor ([rules](https://www.ultraboardgames.com/quoridor/game-rules.php)).
1. Uses self-play deep Q-Learning to train an AI agent to play the game.
1. Let's users pause training and play against the AI at any point.

<p align="center">
<img src="/screenshots/learning.jpg" width="200">
</p>

## Key Commands
When the constant `DISPLAY_GAME` is True, then the pygame window will open and the following commands will be available:
* **d** - Toggle redraws of the screen (drawing slows down learning)
* **h** - Jump into the game and start playing!
* **f** - Change the game speed
* **r** - Toggle full inference mode (100% prediction, no random moves for exploration)

## How it works
* `main.py` - Runs a finite number of games. The AI learns by playing itself over and over.
* `game.py` - Has `run()` which is the game loop. Every turn is characterized by an agent evaluating the state, that agent making a move and the state being updated accordingly.
* `display_game.py` Displays the game by mapping the state onto a graphical representation.
* `actions.py` All the actions that an agent can take.
* `state.py` The state of the game. This checks if actions are legal and converts between the global state and the agent's perspective of the state.
* `model.py` This is the Q-learning neural network that makes action predictions and updates depending on the reward feedback.
* `agents.py` This consists of an Agent class and two subclasses, each for the two agents playing. Each agent has a different view of the board so therefore need to convert the state to their perspective.


## Setup
1. Run `python -m venv venv`
1. Run `venv/Scripts/activate` (windows)
1. Run `pip install -r requirements.txt`
1. Run `python main.py`<br>
*Note this project uses an older version of Tensorflow (1.14)*


[MIT License](/license)