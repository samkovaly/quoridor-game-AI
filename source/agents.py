import numpy as np
import tensorflow as tf
import random
import os
import math

from point import Point
from actions import StaticActions, MoveAction, WallAction

from memory import Memory, MemoryInstance

#from model import Model

import constants
from constants import BoardElement


class Agent:
    """ Parent class of TopAgent and BottomAgent.
        This hierarchy is needed because these two agents share alot of similar functionality (namely, take_action)
        but differ when it comes to converting the board state into their local state (the stat from their point of view) (get_perspective_state())
        The other key difference is their action_to_global_and_back() function which takes the agent's action and converts it to what the action
        looks like to the global state. For instance, when TopAgent moves up, it's a down move from the board's perspective. but up from the agent's perspective
    """

    def __init__(self, sess, static_actions, model, name):
        self.sess = sess

        # size of the state vector that is fed into the NN
        self.state_size = constants.BOARD_SIZE*2 + 1

        self.memory = Memory(constants.MEMORY_SIZE)
        # model is passed here in order to ensure there is only one model object that trains and performs q-learning
        self.model = model

        # static actions allows us to list out all the actions so that greedy_action and random_action can map
        # action_indexes to actions fast
        self.static_actions = static_actions

        # probability of taking a random aciton, which decays as the training goes on.
        self.exploration_probability = constants.STARTING_EXPLORATION_PROBABILITY
        self.steps = 1

        self.game_loss = 0
        self.recent_loss = 0
        self.recent_loss_counter = 0
        
        self.name = name




    def take_action(self, board_state, only_inference, valid_human_action = None):
        """ takes in the state of the game
            determines an action (random or greedy)
            updates the state and collects the reward
            records (S, A, S', R) as a memory
            trains the NN on a batch of recent memories
        """
        # child method is called here.
        state_vector = self.get_perspective_state(board_state)


        if valid_human_action == None:
            if only_inference or random.random() > self.exploration_probability:
                action_index = self.greedy_action(state_vector, board_state)
            else:
                action_index = self.random_action(board_state)
            # in small grids, agents can become stuck if they are next to a wall and the enemy (can't move)
            # and thus action_index will be None in this case
            if action_index == None:
                return None # return None to signify that this game should be abandoned
            else:
                action = self.static_actions.all_actions[action_index]
        else:
            # human actions are special, they are already verified and legal when they arrive here, 
            # but are board-based actions (not good for topagent), so we need to account for this 
            action = self.action_to_global_and_back(valid_human_action)
            action_index = self.static_actions.get_index_of_action(action)

        

        # child method again - 
        #   state needs to be converted to what it looks like to
        #   the board state, so that we can update the state properly
        state_action = self.action_to_global_and_back(action)
        reward = board_state.apply_action(self.name, state_action)



        next_state_vector = self.get_perspective_state(board_state)

        # memory is our training examples
        self.memory.add_sample(MemoryInstance(state_vector, action_index, reward, next_state_vector))
        # learn off a batch of recent memories
        self.q_learn()

        self.steps += 1
        self.exploration_probability = constants.ENDING_EXPLORATION_PROBABILITY + (constants.STARTING_EXPLORATION_PROBABILITY - constants.ENDING_EXPLORATION_PROBABILITY) \
            * math.exp(-constants.EXPLORATION_PROBABILITY_DECAY * self.steps)

        return reward



    def random_action(self, board_state):
        """ Random action to help with exploration.
            The probability of selecting a random move action over a random wall action is high,
            this is because there are many more wall actions than move actions,
            but in the real game move actions are more frequent, so we want our exploration
            phase of training to reflect this and select move much more often than wall """

        if random.random() < constants.MOVE_ACTION_PROBABILITY:
            actions = self.static_actions.move_actions
        else:
            actions = self.static_actions.all_actions
        
        action_indexes = [i for i in range(len(actions))]
        random.shuffle(action_indexes)

        return self.first_legal_action(action_indexes, board_state)


    def greedy_action(self, state_vector, board_state):
        """ Returns a greedy action taken from the model:
            1. gets the Q values from the model given the state
            2. sorts them
            3. go through them until a valid action is found
            4. return the action or of none are found, return a random valid action
        """

        q_values = self.model.predict_one(state_vector)
        q_values = q_values.flatten()

        _, action_indexes = self.sess.run(tf.nn.top_k(q_values, len(q_values)))
        action_indexes = action_indexes.tolist()

        # return the legal action with the highest q-value
        return self.first_legal_action(action_indexes, board_state)



    def first_legal_action(self, action_indexes, board_state):
        """ takes the first legal action found and returns it 
            This is used to take the highest legal Q valued action
        """
        for action_index in action_indexes:
            if self.is_legal_action(action_index, board_state):
                return action_index


    def is_legal_action(self, action_index, board_state):
        """converts this action to the board action and asks the board state if it's legal"""
        action = self.static_actions.all_actions[action_index]
        state_action = self.action_to_global_and_back(action)
        return board_state.is_legal_action(state_action, self.name)



    
    def get_perspective_state(self, board_state):
        """ Gets the agent's perspective of the state (as a vector)
            TopAgent overrides this since it has a different perspecive than BottomAgent
         """
        full_grid_size = board_state.full_grid_size
        grid = board_state.build_grid(BoardElement.AGENT_BOT, BoardElement.AGENT_TOP)

        vector = []
        for y in range(full_grid_size):
            for x in range(full_grid_size):
                vector.append(grid[x][y])

        # my walls, then enemy walls
        vector.append(board_state.wall_counts[BoardElement.AGENT_BOT])
        vector.append(board_state.wall_counts[BoardElement.AGENT_TOP])

        vector = np.array(vector)
        return vector



    def action_to_global_and_back(self, agent_action):
        return agent_action


    def q_learn(self):
        """ Deep Q learning algorithm with memory. Uses the bellman equation.
            Each training example is (state, action, next state, reward)
            Q is R + gamme * max(s', a')
        """
        batch = self.memory.sample(self.model.get_batch_size())

        states = np.array([val[0] for val in batch])
        # When we first start training, some of the memories of examples could be null (not enough for a full batch yet)
        next_states = np.array([(np.zeros(self.model.get_num_states()) if val[3] is None else val[3]) for val in batch])

        # predict Q(s,a) given the batch of states
        q_s_a = self.model.predict_batch(states)

        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self.model.predict_batch(next_states)

        # setup training arrays
        x = np.zeros((len(batch), self.model.get_num_states()))
        y = np.zeros((len(batch), self.model.get_num_actions()))

        # convert each memory to a trainable example via q-lerning method
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]

            # get the current q values for all actions in state
            current_q = q_s_a[i]

            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a') prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + constants.GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q

        _, l = self.model.train_batch(x, y)

        self.game_loss = l
        self.recent_loss += l
        self.recent_loss_counter += 1



    def get_exploration_probability(self):
        return self.exploration_probability


    def get_game_loss(self):
        """ returns loss from previous game """
        return self.game_loss

    def get_recent_loss(self):
        """ returns the average recent loss since this function was last called """
        recentloss = self.recent_loss / self.recent_loss_counter
        self.recent_loss = 0
        self.recent_loss_counter = 0
        return recentloss









class TopAgent(Agent):
    """ Agent that starts out at the top of the screen and has a perspective that the board is 
        flipped horizontally and vertically
    """
    def __init__(self, sess, static_actions, model):
        Agent.__init__(self, sess, static_actions, model, BoardElement.AGENT_TOP)


    def get_perspective_state(self, board_state):
        """ appends grid squres to the state vector in reversed fashion, effectively flipping the
            horizontal and verical axes. ALso append BoardElement.AGENT_TOP's wall count before
            BoardElement.AGENT_BOT's wll count becaue the current agent must come first to preserve consistency 
        """

        full_grid_size = board_state.full_grid_size
        grid = board_state.build_grid(BoardElement.AGENT_TOP, BoardElement.AGENT_BOT)

        vector = []
        for y in reversed(range(full_grid_size)):
            for x in reversed(range(full_grid_size)):
                vector.append(grid[x][y])

        # my walls, then enemy walls
        vector.append(board_state.wall_counts[BoardElement.AGENT_TOP])
        vector.append(board_state.wall_counts[BoardElement.AGENT_BOT])

        vector = np.array(vector)
        return vector




    def action_to_global_and_back(self, agent_action):
        """ Actions are also flipped on both axes """
        if isinstance(agent_action, MoveAction):
            state_action = MoveAction(Point(-agent_action.direction.X, -agent_action.direction.Y))
            
        else:
            agent_wall_pos = agent_action.position
            wall_pos = Point(constants.BOARD_SIZE - agent_wall_pos.X - 2, constants.BOARD_SIZE - agent_wall_pos.Y - 2)
            # orientation doesn't change
            state_action = WallAction(wall_pos, agent_action.orientation)

        return state_action






class BottomAgent(Agent):
    """ Bottom agent has nothing to override because it's perspecitve is the same as
        the boards and us humans"""
    def __init__(self, sess, static_actions, model):
        Agent.__init__(self, sess, static_actions, model, BoardElement.AGENT_BOT)


    def get_perspective_state(self, board_state):
        return super().get_perspective_state(board_state)


    def action_to_global_and_back(self, agent_action):
        return agent_action