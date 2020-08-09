import os

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
            grayscale_stat = tf.image.rgb_to_grayscale(state)
            grayscale_stat = tf.keras.backend.expand_dims(grayscale_stat, axis=0)
            actions = self.model.predict(grayscale_stat, batch_size=1)
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

    def calculate_fitness(self, values_weights=np.array([10, 1, 10000])):
        avg_reward = self.sum_reward / self.steps_count if self.steps_count else 0
        # self.fitness = avg_reward
        # return avg_reward
        values = np.array([avg_reward, self.score, 1 if self.did_win else 0])
        return sum(values*values_weights)

    def _make_model(self, number_of_actions, weights):
        model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(240, 256, 1)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation=tf.nn.swish, bias_initializer='random_normal'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.swish,  bias_initializer='random_normal'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation=tf.nn.swish,  bias_initializer='random_normal'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.swish, bias_initializer='random_normal'),
                tf.keras.layers.Dense(number_of_actions, activation=tf.nn.softmax, bias_initializer='random_normal')
            ])
        if weights is not None:
            model.set_weights(weights)
        return model

    def get_weights(self):
        return self.model.get_weights()



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

