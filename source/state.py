
import math
import copy
import numpy as np

from point import Point
import constants
from actions import StaticActions, MoveAction, WallAction

from astar import a_star
from constants import BoardElement




class State:
    """ state is the state of the game from an independent perspective (each agent has their own perspective).
        This class also holds functions like is_legal_action() which checks whether an action is legal given the current state.
        apply_action() takes a legal action and updates the state accordingly
    """


    def __init__(self, static_actions):
        # there is always (BOARD_SIZE - 1 )*2 possible wall locations because walls are in-between board squares.
        self.walls = [[BoardElement.EMPTY for y in range(constants.BOARD_SIZE-1)] for x in range(constants.BOARD_SIZE-1)]
        self.wall_counts = {BoardElement.AGENT_TOP: constants.NUM_WALLS, BoardElement.AGENT_BOT: constants.NUM_WALLS}

        # wall_counts and agents_positions are indexed by each agent's string identifier since there are only 2 possible agents in the game
        top_agent_pos = Point(math.floor(constants.BOARD_SIZE/2), 0)
        bot_agent_pos = Point(math.floor(constants.BOARD_SIZE/2), constants.BOARD_SIZE-1)
        self.agent_positions = {BoardElement.AGENT_TOP: top_agent_pos, BoardElement.AGENT_BOT: bot_agent_pos}

        # the edge of the board that needs to be reached for an agent to win
        self.agent_goals = {BoardElement.AGENT_TOP: constants.BOARD_SIZE - 1, BoardElement.AGENT_BOT: 0}

        # constant list of all possible actions that the state could see
        self.static_actions = static_actions
        self.winner = None

        self.full_grid_size = constants.BOARD_SIZE*2 -1
        self.vector_state_size = (self.full_grid_size ** 2) + 2






    def is_legal_action(self, action, agent_name):
        """ checks whether this action by this agent is legal or not."""
        if isinstance(action, MoveAction):
            position = self.agent_positions[agent_name]
            return self.legal_move(position, action)
        else:
            return self.legal_wall_placement(agent_name, action)
            


    def legal_move(self, position, move_action):
        """ Determines whether this move_action from this position is legal or not
            Illegal move actions:
                1. agent is moving out of bounds
                2. agent is trying to cross a wall
                3. agent is trying to move into the enemy without using jump
                4. agent is using jump, but it's not over the enemy or it's into a wall
        """
        direction = move_action.direction
        new_position = position + direction

        # out of bounds test
        if new_position.X >= 0 and new_position.X < constants.BOARD_SIZE and new_position.Y >= 0 and new_position.Y < constants.BOARD_SIZE:

            if direction.not_diagonal():
                # normal move (0, 1) or (-1, 0)...
                if direction.abs_sum() == 1:
                    # is the enemy at this new position already??
                    if self.agent_positions[BoardElement.AGENT_TOP] != new_position and self.agent_positions[BoardElement.AGENT_BOT] != new_position:
                        # wall check between the old and new position
                        if not self.wall_between(position, new_position):
                            return True
                # jump (+2, 0)...
                elif direction.abs_sum() == 2:
                    intermediate_position = position + Point(direction.X // 2, direction.Y // 2)
                    # can only jump over the enemy, so they must be there. I check for both agents because
                    # one of them is the enemy, the other agent is the current agent and can't possibly not be at the current position
                    if self.agent_positions[BoardElement.AGENT_TOP] == intermediate_position or \
                        self.agent_positions[BoardElement.AGENT_BOT] == intermediate_position:

                        # check for 2 walls so that the jump will clear
                        if not self.wall_between(position, intermediate_position) and not self.wall_between(intermediate_position, new_position):
                            return True

        return False

    
    def get_valid_neighbors(self, position):
        """Returns a list of all neighbors (Points) of this position."""
        valid_neighbors = []
        move_actions = self.static_actions.move_actions

        # works by calling every possible move action on this position and determining if it's valid
        for action in move_actions:
            if self.legal_move(position, action):
                valid_neighbors.append(self.apply_direction(position, action))
        return valid_neighbors




    def apply_direction(self, position, action):
        """ Applies the direction from a move action to this position to produce a new position"""
        return position + action.direction


    def wall_between(self, position, new_position):
        """ Determines if there is a wall between position and new_position"""
        if position.X == new_position.X:
            # tests the horizontal wall that is centered to the left of position
            if(position.X > 0) and (self.walls[position.X - 1][min(position.Y, new_position.Y)] == BoardElement.WALL_HORIZONTAL):
                return True
            # tests the horizontal wall that is centered to the right of position
            if(position.X < constants.BOARD_SIZE-1) and (self.walls[position.X][min(position.Y, new_position.Y)] == BoardElement.WALL_HORIZONTAL):
                return True
        else:
            # tests the vertical wall that is centered above this position
            if(position.Y > 0) and (self.walls[min(position.X, new_position.X)][position.Y - 1] == BoardElement.WALL_VERTICAL):
                return True
            # tests the vertical wall that is centered below this position
            if(position.Y < constants.BOARD_SIZE-1) and (self.walls[min(position.X, new_position.X)][position.Y] == BoardElement.WALL_VERTICAL):
                return True
        return False



    def legal_wall_placement(self, agent_name, wall_action):
        """Determines whether this wall_action from this agent is legal or not
            Illegal wall placements:
                1. trying to place a wall over another wall location
                2. trying to place a wall taht overlaps another wall nearby
                5. trying to place a wall but the agent is out of walls
                6. trying to place a wall that would eliminate all paths from the enemy position to their goal
                        or from the current agent's position to their goal
        """

        position = wall_action.position
        orientation = wall_action.orientation

        if self.wall_counts[agent_name] == 0:
            return False
        
        if self.get_wall(position) != BoardElement.EMPTY:
            return False

        # need to check if this wall placement will make the game unwinable
        # aka: boxing in the opponent or yourself (no path to goal)
        self.place_wall(position, orientation, agent_name)
        if not self.path_to_goal_exists(BoardElement.AGENT_TOP) or not self.path_to_goal_exists(BoardElement.AGENT_BOT):
            self.remove_wall(position, agent_name)
            return False
        self.remove_wall(position, agent_name)


        # can't partially overlap other placed walls
        if orientation == BoardElement.WALL_VERTICAL:
            # if position +- 1 is out of bounds, then the placemnt is goood
            if (position.Y != constants.BOARD_SIZE-2 and self.walls[position.X][position.Y + 1] == BoardElement.WALL_VERTICAL) \
                or (position.Y != 0 and self.walls[position.X][position.Y - 1] == BoardElement.WALL_VERTICAL):
                return False

        if orientation == BoardElement.WALL_HORIZONTAL:
            if (position.X != constants.BOARD_SIZE-2 and self.walls[position.X + 1][position.Y] == BoardElement.WALL_HORIZONTAL) \
                or (position.X != 0 and self.walls[position.X - 1][position.Y] == BoardElement.WALL_HORIZONTAL):
                return False

        return True


    def get_wall(self, position):
        """ Helper function that returns the wall type at this position"""
        return self.walls[position.X][position.Y]


    def place_wall(self, position, orientation, agent_name):
        """ Places a wall from agent_name of orientation at this position"""
        self.walls[position.X][position.Y] = orientation
        self.wall_counts[agent_name] -= 1


    def remove_wall(self, position, agent_name):
        """ Removes this wall at this position and refunds it to agent_name"""
        self.walls[position.X][position.Y] = BoardElement.EMPTY
        self.wall_counts[agent_name] += 1

    


    def path_to_goal_exists(self, agent_name):
        """ Uses A-Star to check if this agent has a path from it's current position
            to it's goal"""
        start = self.agent_positions[agent_name]

        if agent_name == BoardElement.AGENT_TOP:
            goal_edge = constants.BOARD_SIZE - 1
        else:
            goal_edge = 0

        goal_test = lambda point : point.Y == goal_edge
        heuristic = lambda point : abs(point.Y - goal_edge)

        path_length = a_star(self.get_valid_neighbors, start, goal_test, heuristic)
        return path_length != -1



    def apply_action(self, agent_name, legal_action):
        """ Takes an already tested and tried action (so a legal action) and updates the state with it
            Also returns a reward associated with this action at this state"""
        if isinstance(legal_action, MoveAction):
            return self.apply_move_action(agent_name, legal_action)
        else:
            return self.apply_wall_action(agent_name, legal_action)
            


    def apply_move_action(self, agent_name, move_action):
        """ Takes a valid move action from agent_name and updates the state accordingly"""
        position = self.agent_positions[agent_name]
        new_position = self.apply_direction(position, move_action)
        self.agent_positions[agent_name] = new_position

        # If this position matches the current agent's goal
        # then they win and a different reward is returned
        if new_position.Y == self.agent_goals[agent_name]:
            self.winner = agent_name
            return constants.REWARD_WIN
        
        return constants.REWARD_BEING_ALIVE



    def apply_wall_action(self, agent_name, wall_action):
        """ Takes a valid wall action and updates the state's walls accordingly """
        position = wall_action.position
        orientation = wall_action.orientation

        self.place_wall(position, orientation, agent_name)

        return constants.REWARD_BEING_ALIVE





    def build_grid(self, current_agent, enemy_agent):
        """ transforms the state.walls into a grid where each grid space can be a square, a wall or an agent
            BoardElement.EMPTY for empty squres
            BoardElement.WALL if a wall is present
            This allows a visual picture of what is happening in the game and can easily be vectorized or ML purposes
        """
        grid = [[BoardElement.EMPTY for y in range(self.full_grid_size)] for x in range(self.full_grid_size)]
        
        for y in range(constants.BOARD_SIZE-1):
            for x in range(constants.BOARD_SIZE-1):
                # this grid is almost 2 times as large as the original
                # because for N squres in a row, there are N-1 walls
                grid_x = 2 * x + 1
                grid_y = 2 * y + 1

                if self.walls[x][y] == BoardElement.WALL_HORIZONTAL:
                    grid[grid_x][grid_y] = BoardElement.WALL
                    grid[grid_x - 1][grid_y] = BoardElement.WALL
                    grid[grid_x + 1][grid_y] = BoardElement.WALL

                if self.walls[x][y] == BoardElement.WALL_VERTICAL:
                    grid[grid_x][grid_y] = BoardElement.WALL
                    grid[grid_x][grid_y - 1] = BoardElement.WALL
                    grid[grid_x][grid_y + 1] = BoardElement.WALL


        # Need to also show the agent's location on the grid
        agent_position = self.agent_positions[current_agent]
        grid[agent_position.X * 2][agent_position.Y * 2] = BoardElement.SELF_AGENT

        enemy_position = self.agent_positions[enemy_agent]
        grid[enemy_position.X * 2][enemy_position.Y * 2] = BoardElement.ENEMY_AGENT

        return grid



    def __str__(self):
        grid = self.build_grid(BoardElement.AGENT_BOT, BoardElement.AGENT_TOP)

        s = "grid (AGENT_BOT as current_agent)\n"
        for y in range(self.full_grid_size):
            for x in range(self.full_grid_size):
                s += str(grid[x][y])
            s+= "\n"
        s += "Top Agent's walls left: " + str(self.wall_counts[BoardElement.AGENT_TOP]) + "\n"
        s += "Bot agent's walls left: " + str(self.wall_counts[BoardElement.AGENT_BOT]) + "\n"
        return s
