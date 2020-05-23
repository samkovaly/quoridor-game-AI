
import sys
import time
import random
import math


from memory import MemoryInstance
from point import Point

from actions import StaticActions, MoveAction, WallAction
from model import Model
from agents import TopAgent,  BottomAgent

from state import State

from display_game import DisplayGame

import constants
from constants import BoardElement

import pygame




class QuoridorGame:
    """ Quoridor displays the game, runs the game actions, keeps track of the game state,
        and allows humans to play the machine.
    """
    def __init__(self, sess):
        pygame.init()

        # static_actions is used by other objects to ensure consistency with our actions
        static_actions = StaticActions(constants.BOARD_SIZE)
        self.static_actions = static_actions

        # global board state
        self.state = State(static_actions)

        # display_game draws the state to the screen
        if constants.DISPLAY_GAME:
            self.display_game = DisplayGame()
            

        # model is passed to the agents as a reference to ensure both agents update
        # the same model object over the course of training
        print("Setting up agent networks...")
        self.model = Model(self.state.vector_state_size, len(static_actions.all_actions), constants.BATCH_SIZE, constants.RESTORE, sess)
        top_agent = TopAgent(sess, static_actions, self.model)
        bottom_agent = BottomAgent(sess, static_actions, self.model)
        print("completed\n")

        # will iterate through self.agents to create a turn bases system
        self.agents = {BoardElement.AGENT_BOT: bottom_agent, BoardElement.AGENT_TOP: top_agent}
        # if a human is playing, they are assigned the bottom agent, could be top just as easily
        self.human_agent = BoardElement.AGENT_TOP


        # boolen flags to help with development, see the game play,
        # initiate human interaction and have more fun
        self.drawing_screen = constants.DISPLAY_GAME
        self.game_delay = constants.INITIAL_GAME_DELAY
        self.only_inference = constants.INITIALLY_USING_ONLY_INFERENCE
        self.human_playing = constants.INITIALLY_HUMAN_PLAYING

        # statistics
        self.sum_game_lengths = 0
        self.games = 0
        self.victories = {BoardElement.AGENT_TOP: 0, BoardElement.AGENT_BOT: 0}

        self.reward_sum = 0
        self.reset()


    
    def reset(self):
        """ reset state after each game """
        self.actions_taken = 0
        self.state = State(self.static_actions)
        self.human_action = None

        # also reset the visuals
        if self.drawing_screen:
            self.display_game.reset(self.state)



    def run(self):
        """ runs a single Quoridor game until one side wins, performs training, 
            and optionally displays the game and let's people play against the agent
        """

        self.reset()
        game_over = False

        # first player
        if random.random() > 0.5:
            current_agent = BoardElement.AGENT_BOT
        else:
            current_agent = BoardElement.AGENT_TOP

        while not game_over:

            # if a human is playing and they have not initiated a valid action, 
            # then skip this and keep waiting for the action, otherwise run as normal without human
            if not self.human_playing or (self.human_playing and self.human_action != None) or current_agent != self.human_agent:

                agent = self.agents[current_agent]
                reward = agent.take_action(self.state, self.only_inference, self.human_action)

                if reward == None: # 0 moves to make, so abandon this game
                    game_over = True
                    break
                
                self.human_action = None

                self.actions_taken += 1
                self.reward_sum += reward

                # let the opponent have a go
                if current_agent == BoardElement.AGENT_BOT:
                    current_agent = BoardElement.AGENT_TOP
                else:
                    current_agent = BoardElement.AGENT_BOT

                if self.state.winner:
                    # update statistics and exit the loop
                    game_over = True
                    self.victories[agent.name] += 1
                    self.games += 1

                    self.sum_game_lengths += self.actions_taken

                if self.drawing_screen:
                    self.display_game.draw_screen(self.state)


            # slows the training down but interesting to watch
            if(self.game_delay > 0):
                time.sleep(self.game_delay)

            self.check_pygame_events()




    def check_pygame_events(self):
        # check for events, including a windows force close
        for event in pygame.event.get():
            # close window
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # key was pressed
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_d:
                    self.drawing_screen = not self.drawing_screen
                    print('drawing screen:', self.drawing_screen)
                
                elif event.key == pygame.K_f:
                    # This toggles between instant drawing (which is cool to see) and
                    # a more practical slow drawing to inspect the agent's actions
                    if self.game_delay == constants.GAME_DELAY_SEONDS:
                        self.game_delay = 0
                    else:
                        self.game_delay = constants.GAME_DELAY_SEONDS
                    print("game delay:", self.game_delay)
                        
                elif event.key == pygame.K_r:
                    # Turning random off, will cause all actions taken to be a prediction from the model
                    # This effectively turns training off and inference on
                    self.only_inference = not self.only_inference
                    print ("using inference: ", self.only_inference)

                elif event.key == pygame.K_h:
                    # toggle human mode
                    self.human_playing = not self.human_playing
                    print("human playing:", self.human_playing)
                    

            if event.type == pygame.MOUSEBUTTONDOWN and self.drawing_screen:
                if self.human_playing:
                    self.human_action = self.get_human_action_index(pygame.mouse.get_pos())

            



    def get_human_action_index(self, mouse_position):
        """ human actions are special in that they are determined and validated from within Game,
            as opposed to agent actions which are determined and validated from within the agents themselves
            This returned action will directly be applied to the state via the agent.
        """
        mouse_position_point = Point(mouse_position[0], mouse_position[1])
        action = self.action_from_mouse(mouse_position_point)
        if self.state.is_legal_action(action, self.human_agent):
            #action_index = self.static_actions.get_index_of_action(action)
            return action
        else:
            return None


    def action_from_mouse(self, mouse_position):
        """ given the mouse position, this function determines the action that the human is intending to give."""
        square_size = self.display_game.square_size
        wall_size = self.display_game.wall_size

        # distance to the closest left square's left side (same for top)
        distance_to_square_left = mouse_position.X % (square_size + wall_size)
        distance_to_square_top = mouse_position.Y % (square_size + wall_size)

        # if the distance to the left side of the square is less than the square's size,
        # then the mouse must be over a square. Therefore the user is trying to move
        if distance_to_square_left < square_size and distance_to_square_top < square_size:
            return self.move_action_from_mouse(mouse_position)
        else:
            return self.wall_action_from_mouse(mouse_position, square_size)



    def move_action_from_mouse(self, mouse_position):
        """ returns a MoveAction based on the mouse position, this action may or may not be valid"""
        # get this squares grid X and Y
        selected_square_x = int(mouse_position.X / constants.SCREEN_SIZE * constants.BOARD_SIZE)
        selected_square_y=  int(mouse_position.Y / constants.SCREEN_SIZE * constants.BOARD_SIZE)

        # Make a move action whos direction the the delta between the agent and the mouse click square.
        # Will determine if this action is valid later
        position = self.state.agent_positions[self.human_agent]
        new_position = Point(selected_square_x, selected_square_y)
        position_delta = Point(new_position.X - position.X, new_position.Y - position.Y)
        move_action = MoveAction(position_delta)

        return move_action


    def wall_action_from_mouse(self, mouse_position, square_size):
        """ returns a WallAction based on the mouse position, this action may or may not be valid"""
        # must be over a wall
        selected_wall_x = int((mouse_position.X - square_size / 2) * (constants.BOARD_SIZE / constants.SCREEN_SIZE))
        selected_wall_y = int((mouse_position.Y - square_size / 2) * (constants.BOARD_SIZE / constants.SCREEN_SIZE))
        
        # prevent out of bounds
        if selected_wall_x > constants.BOARD_SIZE - 2:
            selected_wall_x = constants.BOARD_SIZE - 2
        if selected_wall_y > constants.BOARD_SIZE - 2:
            selected_wall_y = constants.BOARD_SIZE - 2

        # get the center of this potential wall (same for horizontal as for vertical)
        center_wall_location = Point((selected_wall_x + 1) * constants.SCREEN_SIZE / (constants.BOARD_SIZE), (selected_wall_y + 1) * constants.SCREEN_SIZE / (constants.BOARD_SIZE))
        # if y is closer, then most likely the user wants a horizontal wall
        if (abs(mouse_position.X - center_wall_location.X) > abs(mouse_position.Y - center_wall_location.Y)):
            orientation = BoardElement.WALL_HORIZONTAL
        else:
            orientation = BoardElement.WALL_VERTICAL
        wall_action = WallAction(Point(selected_wall_x, selected_wall_y), orientation)

        return wall_action





    def print_details(self, games_per_epoch):
        """ print details on recent statistics to see how training is coming along"""
        self.model.save()

        local_avg_game_length = self.sum_game_lengths / games_per_epoch
        self.sum_game_lengths = 0

        recent_reward_avg = self.reward_sum / games_per_epoch
        self.reward_sum = 0

        print("Top Victories: ", self.victories[BoardElement.AGENT_TOP])
        print("Bot Victories: ", self.victories[BoardElement.AGENT_BOT])
        print("Local Average Game Length: ", local_avg_game_length)
        print("Local Average Game Reward: ", recent_reward_avg)

        print("Local Average Loss: ", self.agents[BoardElement.AGENT_BOT].get_recent_loss())
        print('exploration_probability', self.agents[BoardElement.AGENT_TOP].get_exploration_probability())