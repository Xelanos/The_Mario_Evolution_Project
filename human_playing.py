"""
A method to play nes gym environments using human IO inputs.
The goal of this method is to compare human player to AI agent.
code taken from https://github.com/Kautenja/nes-py/blob/master/nes_py/app/play_human.py
"""

import gym
import nes_py
from pyglet.window import key
import gym_super_mario_bros
import numpy as np
import itertools
import time
from pyglet import clock
from nes_py._image_viewer import ImageViewer
from player import HumanPlayer

# keyboard keys in an array ordered by their byte order in the bitmap
# i.e. right = 7, left = 6, ..., B = 1, A = 0
BUTTONS = np.array([
            key.RIGHT,  # right
            key.LEFT,  # left
            key.DOWN,  # down
            key.UP,  # up
            key.ENTER, # start
            key.SPACE,  # select
            key.Z,  # B
            key.X,  # A
        ])

TIME_PER_GAME = 1500
# the sentinel value for "No Operation"
_NOP = 0


def get_keys_to_action(buttons):
    """
    :param buttons: array of keyboard keys to represent nes controller.
    :return: dictionary of keyboard keys to actions.
    """
    keys_to_action = {}
    # the combination map of values for the controller
    values = 8 * [[0, 1]]
    # iterate over all the combinations
    for combination in itertools.product(*values):
        # unpack the tuple of bits into an integer
        byte = int(''.join(map(str, combination)), 2)
        # unwrap the pressed buttons based on the bitmap
        pressed = buttons[list(map(bool, combination))]
        # assign the pressed buttons to the output byte
        keys_to_action[tuple(sorted(pressed))] = byte

    return keys_to_action


def run(env: nes_py.NESEnv, buttons=BUTTONS):
    # ensure the observation space is a box of pixels
    assert isinstance(env.observation_space, gym.spaces.box.Box)
    # ensure the observation space is either B&W pixels or RGB Pixels
    obs_s = env.observation_space
    is_bw = len(obs_s.shape) == 2
    is_rgb = len(obs_s.shape) == 3 and obs_s.shape[2] in [1, 3]
    assert is_bw or is_rgb
    # get the mapping of keyboard keys to actions in the environment
    keys_to_action = get_keys_to_action(buttons)
    # create the image viewer
    viewer = ImageViewer(
        env.spec.id if env.spec is not None else env.__class__.__name__,
        env.observation_space.shape[0],  # height
        env.observation_space.shape[1],  # width
        monitor_keyboard=True,
        relevant_keys=set(sum(map(list, keys_to_action.keys()), []))
    )
    # create a done flag for the environment
    done = False
    state = env.reset()
    # prepare frame rate limiting
    target_frame_duration = 1 / env.metadata['video.frames_per_second']
    last_frame_time = 0
    # start the main game loop
    try:
        while not done:
            current_frame_time = time.time()
            # limit frame rate
            if last_frame_time + target_frame_duration > current_frame_time:
                continue
            # save frame beginning time for next refresh
            last_frame_time = current_frame_time
            # clock tick
            clock.tick()

            # unwrap the action based on pressed relevant keys
            action = keys_to_action.get(viewer.pressed_keys, _NOP)
            next_state, reward, done, info = env.step(action)
            # reset if the environment is done
            if not done:
                viewer.show(env.unwrapped.screen)
            state = next_state
            # shutdown if the escape key is pressed
            if viewer.is_escape_pressed:
                break
    except KeyboardInterrupt:
        pass

    viewer.close()
    env.close()

if __name__ == "__main__":
    run(gym_super_mario_bros.make('SuperMarioBros-v0'))