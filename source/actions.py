

from constants import BoardElement
from point import Point
import constants



class StaticActions:
    """ When agents compute actions, they end up with numbers, so action indexes
        This class is needed to map action indexes to actual action objects that can be passed
        to the board's state
        """
    def __init__(self, board_size):
        move_actions = list()
        move_actions.append(MoveAction(Point(1, 0)))
        move_actions.append(MoveAction(Point(-1, 0)))
        move_actions.append(MoveAction(Point(0, 1)))
        move_actions.append(MoveAction(Point(0, -1)))

        move_actions.append(MoveAction(Point(2, 0)))
        move_actions.append(MoveAction(Point(-2, 0)))
        move_actions.append(MoveAction(Point(0, 2)))
        move_actions.append(MoveAction(Point(0, -2)))

        self.move_actions = move_actions

        # only board_size - 1 walls for each row and column because walls are only between board squares, not outside them
        wall_actions = list()
        for y in range(board_size - 1):
            for x in range(board_size - 1):
                wall_actions.append(WallAction(Point(x, y), BoardElement.WALL_HORIZONTAL))
                wall_actions.append(WallAction(Point(x, y), BoardElement.WALL_VERTICAL))

        self.wall_actions = wall_actions

        self.all_actions = move_actions + wall_actions



    def get_index_of_action(self, action):
        """ gets index of an action, used by human players who get their actions form
            mouse clicks and therfore don't immediately have access to the action's index
            The index is what's fed to the Neural net work so it's necesary for learning """
        if isinstance(action, MoveAction):
            return self.get_index_of_move_action(action)
        if isinstance(action, WallAction):
            return self.get_index_of_wall_action(action)

    def get_index_of_move_action(self, action):
        return self.move_actions.index(action)
    
    def get_index_of_wall_action(self, action):
        return len(self.move_actions) + self.wall_actions.index(action)


class MoveAction:
    def __init__(self, direction):
        self.direction = direction

    def __eq__(self, other):
        if other == None:
            return False
        return self.direction == other.direction


class WallAction:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation
        
    def __eq__(self, other):
        if other == None:
            return False
        return self.position == other.position and self.orientation == other.orientation