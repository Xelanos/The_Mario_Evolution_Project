import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from wrappers import *
import gym.wrappers.monitor as monitor
import time
from pandas import DataFrame
from player import MarioPlayer

INITIAL_LIFE = 2
TIME_SCALE = 2000
NO_ADVANCE_STEP_LIMIT = 100


class RandomMarioPlayer:

    def __init__(self, mario_environment, actions, trials, steps_scale=TIME_SCALE,
                 allow_death=False, standing_steps_limit=NO_ADVANCE_STEP_LIMIT, output_dir=""):
        self.actions = actions
        self.num_of_actions = len(actions)
        self.trials = trials
        self.trial = 0
        self.steps_scale = steps_scale
        self.allow_death = allow_death
        self.standing_steps_limit = standing_steps_limit
        self.record = False
        self.render = False
        self.output_dir = output_dir
        self.record_dir = os.path.join(self.output_dir, "vid")
        self.env = mario_environment

    def run(self, render_every=100, record_every=0):
        outcomes = []
        current_best = float("-inf")
        for trail in range(self.trials):
            self.trial = trail
            print(f'Staring trail {trail + 1}')
            t = time.time()

            self.render = (trail % render_every == 0) if render_every else False
            self.record = (trail % record_every == 0) if record_every else False

            if self.record and not os.path.isdir(self.record_dir):
                os.mkdir(self.record_dir)

            player = MarioPlayer(self.num_of_actions)
            outcome = self.run_player(player)
            if outcome["performance_score"] > current_best:
                player.save_player(self.output_dir, f"trail_{trail}")
                current_best = outcome["performance_score"]
            outcomes.append(outcome)
            print(f"finish {trail + 1} in {time.time() - t}")

        DataFrame(outcomes).to_csv(os.path.join(self.output_dir, "random_output.csv"))
        return outcomes

    def run_player(self, player):
        env = gym_super_mario_bros.make(self.env)
        env = JoypadSpace(env, self.actions)
        env = WarpFrame(env)
        env = FrameStack(env, 4)

        if self.record:
            rec_output_path = os.path.join(self.record_dir, "{name}.mp4".
                                           format(name=self.trial))
            rec = monitor.video_recorder.VideoRecorder(env, path=rec_output_path)

        state = env.reset()
        done = False

        last_x_pos = 0
        same_x_pos_cunt = 0

        for step in range(self.steps_scale):
            if done:
                break
            action = player.act(state)
            state, reward, done, info = env.step(action)

            if self.record:
                rec.capture_frame()
            if self.render:
                env.render()

            player.update_info(info)
            player.update_reward(reward)
            if last_x_pos == info['x_pos']:
                same_x_pos_cunt += 1
            else:
                same_x_pos_cunt = 0
                last_x_pos = info['x_pos']
            if same_x_pos_cunt > self.standing_steps_limit:  # end the run if player don't advance:
                done = True
            if not self.allow_death and info['life'] < INITIAL_LIFE:  # will repeat death, so why try more
                done = True
            if info['flag_get']:  # if got to the flag - run is ended.
                done = True

        if self.record:
            rec.close()
        env.close()

        outcome = player.get_run_info()
        outcome['trial'] = self.trial
        return outcome