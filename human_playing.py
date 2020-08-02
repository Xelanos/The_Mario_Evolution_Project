"""
A method to play nes gym environments using human IO inputs.
The goal of this method is to compare human player to AI agent.
code taken from https://github.com/Kautenja/nes-py/blob/master/nes_py/app/play_human.py
"""

import os
import gym
import gym.wrappers.monitor as monitor
import nes_py
from pyglet.window import key
import gym_super_mario_bros
import numpy as np
import itertools
import time
from pyglet import clock
from nes_py._image_viewer import ImageViewer

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

NO_ADVANCE_STEP_LIMIT = 100
MAX_STEPS_PER_GAME = 2000
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


def run(env: nes_py.NESEnv, max_steps: int = MAX_STEPS_PER_GAME, standing_steps_limit: int = NO_ADVANCE_STEP_LIMIT,
        buttons=BUTTONS, allow_dying=True, record=""):
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
    last_x_pos = 0
    same_x_pos_cunt = 0
    reward_sum = 0
    # record if necessary
    if record:
        assert not os.path.isdir(os.path.dirname(record))
        rec = monitor.video_recorder.VideoRecorder(env, path=record)
    # number of steps for the game
    num_of_steps = 0
    # prepare frame rate limiting
    target_frame_duration = 1 / env.metadata['video.frames_per_second']
    last_frame_time = 0
    # start the main game loop
    try:
        while not done and not viewer.is_escape_pressed:
            if record:
                rec.capture_frame()
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
            num_of_steps += 1
            reward_sum += reward
            if last_x_pos == info['x_pos']:
                same_x_pos_cunt += 1
            else:
                same_x_pos_cunt = 0
                last_x_pos = info['x_pos']
            # end the game if player don't advance:
            if same_x_pos_cunt > standing_steps_limit:
                done = True
                print("player not advancing for {} frames. Ending the game.".format(same_x_pos_cunt))
            # end the game if got to the flag.
            if info['flag_get']:
                done = True
                print("Got the flag!")
            # end the game if times is up
            if num_of_steps > max_steps:
                done = True
                print("Time is up!")
            # show the environment if not done
            if not done:
                viewer.show(env.unwrapped.screen)
            state = next_state
        avg_reward = reward_sum / num_of_steps
        print("Done in {} steps. Average reward {}.".format(num_of_steps, avg_reward))
    except KeyboardInterrupt:
        print("Trail stopped by KeyboardInterrupt.")
        pass
    viewer.close()
    env.close()


if __name__ == "__main__":
    run(gym_super_mario_bros.make('SuperMarioBros-v0'))