
import tensorflow as tf

from game import QuoridorGame
from model import Model
from memory import Memory

import constants

import pygame



"""
    Keys: (only when DISPLAY_GAME is true)
        d - toggle display of games
        h - jump into the game and start playing!
        f - sleep up or slow down the games
        r - toggle full inference mode (hardcore) (no randomness)
"""

def main():
    """ An attempt to teach agents to play the game Quoridor.
        This code also allows humans to play the AI.
        AI are trained using self-play deep Q learning, a simple RL technique
        The agents can be saved to file and then loaded back to play against man.
        quoridor rules: https://www.ultraboardgames.com/quoridor/game-rules.php
    """
    
    # tensorflow 1.14-ish session
    with tf.Session() as sess:

        game = QuoridorGame(sess)

        epoch = 0
        print("Learning Initiated...")
        while epoch < constants.NUM_GAMES:
            # print an update or us humans to read
            if epoch % constants.PRINT_UPDATE_FREQUENCY == 0 and epoch != 0:
                print('\nEpoch {} of {}'.format(epoch, constants.NUM_GAMES))
                game.print_details(constants.PRINT_UPDATE_FREQUENCY)
            game.run()
            epoch += 1
    print('Simulation complete')
    pygame.quit()



if __name__ == '__main__':
    main()