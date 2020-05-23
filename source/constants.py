

# SIZE OF THE GAME
BOARD_SIZE = 4                          # board size (less complexity)
NUM_WALLS = 2                           # number of walls each player starts with


# PROGRAM PURPOSE
'''If you want a human to vs a trained agent, then turn these on
    Note: These can be toggles on or off during the games'''
DISPLAY_GAME = True                     # false to avoid pygame altogether
INITIALLY_HUMAN_PLAYING = False         # ultimate test of intelligence
INITIALLY_USING_ONLY_INFERENCE = False
RESTORE = False                         # signifies if agent should be loaded from tensorflow checkpoint (from disc)

INITIAL_GAME_DELAY = 0                  # initial value for game_delay, which simply slows down the game so we can watch the agents plays ;)
GAME_DELAY_SEONDS = 1                   # if game_delay is switched on, this is the delay used


# TRAINING PARAMETERS
NUM_GAMES = 1000

REWARD_WIN = 1.0                        # big bucks
REWARD_BEING_ALIVE = -.04               # yikes

MEMORY_SIZE = 500                       # max number of (s,a,s',r) samples to store for learning at once
BATCH_SIZE = 50                         # how many actions from memory to learn from at a time

MOVE_ACTION_PROBABILITY = .90           # training wheels to encorage the agents to move more often
GAMMA = 0.80                            # future reward discount factor (bellman equation)

# how often to take random actions for the sake of exploration
# decays over games starts from 1 and goes to 0 asymptotically
STARTING_EXPLORATION_PROBABILITY = 1.0
ENDING_EXPLORATION_PROBABILITY = 0.0
EXPLORATION_PROBABILITY_DECAY = 0.00005 # decay of epsilon

PRINT_UPDATE_FREQUENCY = 10


# DISPLAY PARAMETERS
SCREEN_SIZE = 400
SQUARE_TO_WALL_SIZE_RATIO = 5
AGENT_COLOR_TOP = (230, 46, 0) # red
AGENT_COLOR_BOT = (0, 0, 255) # blue
WALL_COLOR = (207, 98, 52) # brown
SQUARE_COLOR = (182, 240, 216) # light green


class BoardElement():
    """ constants that define what the board can hold and what the NN sees as inputs """
    EMPTY = 0           # NN is fed this as input for empty grid spaces
    WALL = -1           # same here for walls
    SELF_AGENT = 5      # nn input
    ENEMY_AGENT = 1     # nn input
    AGENT_TOP = "T"
    AGENT_BOT = "B"
    WALL_HORIZONTAL = "H"
    WALL_VERTICAL = "V"