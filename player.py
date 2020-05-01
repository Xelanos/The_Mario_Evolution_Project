import tensorflow as tf
import numpy as np


class Player:

    def __init__(self, model):
        self.model = model
        self.fitness = 0
        self.farthest_x = 0
        self.farthest_x_time = 400
        self.breeding_probability = 0


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

    def calculate_fittness(self):
        if self.farthest_x == 40 and self.farthest_x_time == 400:
            self.fitness = 0
        self.fitness = self.farthest_x + self.farthest_x_time
