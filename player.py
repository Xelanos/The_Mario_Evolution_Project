import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import tensorflow as tf
import numpy as np



class MarioPlayer:

    def __init__(self, number_of_actions, weights=None):
        self.model = self._make_model(number_of_actions, weights)
        self.fitness = 0
        self.farthest_x = 0
        self.farthest_x_time = 400
        self.reward = 0


    def act(self, state):
        grayscale_stat = tf.image.rgb_to_grayscale(state)
        grayscale_stat = tf.keras.backend.expand_dims(grayscale_stat, axis=0)
        actions = self.model.predict(grayscale_stat, batch_size=1)
        action = np.argmax(actions)
        return action

    def update_info(self, info):
        if info['x_pos'] > self.farthest_x:
            self.farthest_x = info['x_pos']
            self.farthest_x_time = info['time']
        self.score = info['score']
        self.did_win = info['flag_get']

    def calculate_fittness(self):
        # self.fitness = self.reward
        # return self.reward
        return 10 * self.farthest_x + self.score + 10000 * self.did_win

    # def _make_model(self, number_of_actions, weights):
    #     model = tf.keras.Sequential([
    #             tf.keras.layers.Flatten(input_shape=(240, 256, 1)),
    #             tf.keras.layers.Dense(80, activation=tf.nn.relu, name='hidden',
    #                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=3)),
    #             tf.keras.layers.Dense(number_of_actions, activation=tf.nn.softmax, use_bias=False)
    #         ])
    #     if weights is not None:
    #         model.get_layer('hidden').set_weights([weights, np.zeros((80,))])
    #     return model

    def _make_model(self, number_of_actions, weights):
        model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(240, 256, 1)),
                tf.keras.layers.Dense(100, activation=tf.nn.swish, name='hidden', use_bias=True),
                tf.keras.layers.Dense(20, activation=tf.nn.swish, name='hidden2', use_bias=True),
                tf.keras.layers.Dense(number_of_actions, activation=tf.nn.softmax, use_bias=True)
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
        self.farthest_x = 0
        self.farthest_x_time = 400
        self.reward = 0
        self.frames_alive = 0
        self.score = 0
        self.did_win = False


    def act(self, state):
        grayscale_stat = tf.image.rgb_to_grayscale(state)
        grayscale_stat = tf.keras.backend.expand_dims(grayscale_stat, axis=0)
        actions = self.model.predict(grayscale_stat, batch_size=1)
        action = np.argmax(actions)
        return action

    def update_info(self, info):
        pass

    def calculate_fittness(self):
        self.fitness = self.reward
        return self.reward
        # if self.farthest_x == 40:
        #     self.fitness = 1
        # self.fitness = self.farthest_x
        # return self.fitness

    def _make_model(self, number_of_actions, weights):
        model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(4)),
                tf.keras.layers.Dense(10, activation=tf.nn.relu, name='hidden', use_bias=False),
                tf.keras.layers.Dense(number_of_actions, activation=tf.nn.softmax, use_bias=False)
            ])
        if weights is not None:
            model.set_weights(weights)
        return model

    def get_weights(self):
        return self.model.get_weights()
