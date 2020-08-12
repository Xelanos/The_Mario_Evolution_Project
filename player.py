import os
import json
from wrappers import DEFAULT_WARP_FRAME_HEIGHT,DEFAULT_WARP_FRAME_WIDTH

INITIAL_LIFE = 2
INITIAL_STATUS = 'small'
DEFAULT_ACTION = 0

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
import numpy as np

initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)


class MarioPlayer:

    def __init__(self, number_of_actions, weights=None):
        self.model = self._make_model(number_of_actions, weights)
        self.fitness = 0
        self.farthest_x = 0
        self.farthest_x_time = 400
        self.sum_reward = 0
        self.steps_count = 0
        self.score = 0
        self.lives = INITIAL_LIFE
        self.coins = 0
        self.status = INITIAL_STATUS
        self.did_win = False

    def act(self, state):
        if state is not None:
            state = tf.keras.backend.expand_dims(state, axis=0)
            actions = self.model.predict(state, batch_size=1)
            action = np.argmax(actions)
        else:
            action = DEFAULT_ACTION
        self.steps_count += 1
        return action

    def update_info(self, info):
        if info['x_pos'] > self.farthest_x:
            self.farthest_x = info['x_pos']
            self.farthest_x_time = info['time']
        self.score = info['score']
        self.lives = -1 if info['life'] == 255 else info['life']
        self.coins = info['coins']
        self.status = info['status']
        self.did_win = info['flag_get'] if info['flag_get'] else self.did_win

    def update_reward(self, reward):
        self.sum_reward += reward
        return self.sum_reward

    def get_run_info(self):
        avg_reward = self.sum_reward / self.steps_count if self.steps_count else 0
        return {'avg_reward': avg_reward, 'steps': self.steps_count, 'score': self.score,
                'deaths': INITIAL_LIFE - self.lives, 'coins': self.coins, 'finish_status': self.status,
                'finish_level': self.did_win, 'performance_score': self.calculate_fitness()}

    def calculate_fitness(self, values_weights=np.array([10, 1, 10000, -800])):
        avg_reward = self.sum_reward / self.steps_count if self.steps_count else 0
        died = self.lives < INITIAL_LIFE
        # self.fitness = avg_reward
        # return avg_reward
        values = np.array([avg_reward, self.score, 1 if self.did_win else 0, 1 if died else 0])
        return sum(values*values_weights)

    def _make_model(self, number_of_actions, weights):
        random_uniform_init = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
        model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(DEFAULT_WARP_FRAME_HEIGHT, DEFAULT_WARP_FRAME_WIDTH, 4)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, padding='same', activation=tf.keras.layers.LeakyReLU(), bias_initializer=random_uniform_init, kernel_initializer=random_uniform_init),
                tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(),  bias_initializer=random_uniform_init, kernel_initializer=random_uniform_init),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),  bias_initializer=random_uniform_init, kernel_initializer=random_uniform_init),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu, bias_initializer=random_uniform_init, kernel_initializer=random_uniform_init),
                tf.keras.layers.Dense(number_of_actions, activation=tf.nn.softmax, bias_initializer=random_uniform_init, kernel_initializer=random_uniform_init)
            ])
        if weights is not None:
            model.set_weights(weights)
        return model

    def get_weights(self):
        return self.model.get_weights()

    def save_player(self, output_dir, name):
        weights = self.model.get_weights()
        info = {"fitness": self.calculate_fitness(),
                "farthest_x": int(self.farthest_x),
                "farthest_x_time": int(self.farthest_x_time),
                "sum_reward": int(self.sum_reward),
                "steps_count": self.steps_count,
                "score": self.score,
                "lives": int(self.lives),
                "coins": self.coins,
                "status": self.status,
                "did_win": self.did_win,
                "weights_path": os.path.join(output_dir, name + ".npz"),
                "weights_len": len(weights)}
        np.savez_compressed(os.path.join(output_dir, name), *weights)
        with open(os.path.join(output_dir, name + "_info.json"), 'w') as f:
            json.dump(info, f)

    def load_player(self, input_json_path):
        with open(input_json_path, 'r') as f:
            info = json.load(f)
            self.fitness = info["fitness"]
            self.farthest_x = info["farthest_x"]
            self.farthest_x_time = info["farthest_x_time"]
            self.sum_reward = info["sum_reward"]
            self.steps_count = info["steps_count"]
            self.score = info["score"]
            self.lives = info["lives"]
            self.coins = info["coins"]
            self.status = info["status"]
            self.did_win = info["did_win"]
            loaded_weights = np.load(info["weights_path"])
            weights = []
            for i in range(info["weights_len"]):
                weights.append(loaded_weights["arr_" + str(i)])
            self.model.set_weights(weights)






class BipedalPlayer:

    def __init__(self, weights=None):
        self.model = self._make_model(weights)
        self.fitness = 0
        self.reward = 0
        self.frames_alive = 0
        self.score = 0
        self.did_win = False



    def act(self, state):
        state = tf.keras.backend.expand_dims(state, axis=0)
        actions = self.model.predict(state, batch_size=1).squeeze()
        return actions

    def update_info(self, info):
        pass

    def calculate_fittness(self):
        return self.reward

    def _make_model(self, weights):
        model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(24)),
                tf.keras.layers.Dense(13, bias_initializer=initializer, kernel_initializer=initializer),
                tf.keras.layers.Dense(8, bias_initializer=initializer, kernel_initializer=initializer),
                tf.keras.layers.Dense(13, bias_initializer=initializer, kernel_initializer=initializer),
                tf.keras.layers.Dense(4, activation=tf.nn.tanh, bias_initializer=initializer, kernel_initializer=initializer)
            ])
        if weights is not None:
            model.set_weights(weights)
        return model

    def get_weights(self):
        return self.model.get_weights()

