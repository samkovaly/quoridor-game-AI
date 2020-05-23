
import pygame

from constants import BoardElement

import constants



class DisplayGame:
    """ class which handles drawing the state of the board to the screen. 
        Also computes the square and wall sizes needed by game.action_from_mouse_position """
    def __init__(self):

        self.square_size = self.compute_square_size()
        self.wall_size = round(self.square_size / constants.SQUARE_TO_WALL_SIZE_RATIO)
        
        self.screen = pygame.display.set_mode((constants.SCREEN_SIZE, constants.SCREEN_SIZE), pygame.SRCALPHA, 32)

    def reset(self, state):
        """ resets the display of the board"""
        self.draw_screen(state)

    def compute_square_size(self):
        """ Too lengthy for one line, much cleaner in this function.
            This equation is non-intuitive, I just simply derived it with pen and paper.
        """
        numerator = constants.SCREEN_SIZE * constants.SQUARE_TO_WALL_SIZE_RATIO
        denominator = (constants.BOARD_SIZE * constants.SQUARE_TO_WALL_SIZE_RATIO) + constants.BOARD_SIZE - 1
        return round(numerator / denominator)



    def draw_screen(self, state):
        """ Draws the state to the pygame window. Should only be called when the state changes"""

        self.screen.fill(0)

        # these were common variables in the calculations below so I extracted them here to same computation
        offset_distance = self.square_size + self.wall_size
        half_square = round(self.square_size / 2)
        agent_radius = round(self.square_size * .40)

        # draw squares
        for y in range(constants.BOARD_SIZE):
            for x in range(constants.BOARD_SIZE):
                pygame.draw.rect(self.screen, constants.SQUARE_COLOR, [x*offset_distance, y*offset_distance, self.square_size, self.square_size])

        # draw agents
        top_agent_pos = state.agent_positions[BoardElement.AGENT_TOP]
        pygame.draw.circle(self.screen, constants.AGENT_COLOR_TOP, (round(top_agent_pos.X * offset_distance + half_square), round(top_agent_pos.Y * offset_distance + half_square)), agent_radius)
        
        bot_agent_pos = state.agent_positions[BoardElement.AGENT_BOT]
        pygame.draw.circle(self.screen, constants.AGENT_COLOR_BOT, (round(bot_agent_pos.X * offset_distance + half_square), round(bot_agent_pos.Y * offset_distance + half_square)), agent_radius)


        # draw walls
        walls = state.walls
        for y in range(len(walls)):
            for x in range(len(walls)):
                if walls[x][y] == BoardElement.WALL_HORIZONTAL:
                    pygame.draw.rect(self.screen, constants.WALL_COLOR, [x * offset_distance, y * offset_distance + self.square_size, self.square_size*2 + self.wall_size, self.wall_size])
                elif walls[x][y] == BoardElement.WALL_VERTICAL:
                    pygame.draw.rect(self.screen, constants.WALL_COLOR, [x * offset_distance + self.square_size, y * offset_distance, self.wall_size, self.square_size*2 + self.wall_size])
        

        pygame.display.flip()
        pygame.display.update()