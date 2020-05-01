import tensorflow as tf
import numpy as np
from player import Player


class GeneticMario:

    def __init__(self, env, generations, initial_pop):
        self.env = env
        self.num_of_actions = env.action_space.n
        self.generations = generations
        self.inital_pop = initial_pop
        self.poplation = []
        self.__init_pop()

    def run(self, render=False):
        for gen in range(self.generations):
            print(f'Staring generation {gen + 1}')
            for i, player in enumerate(self.poplation):
                # rec = monitor.video_recorder.VideoRecorder(env, path=f"vid/t{i}.mp4")
                self.env.reset()
                done = False
                action = 0
                for step in range(10):
                    if done:
                        break
                    state, reward, done, info = self.env.step(action)
                    # rec.capture_frame()
                    action = player.act(state)
                    player.update_info(info)
                    if info['life'] < 2:
                        done = True

                    # plt.imshow(grayscale_stat, cmap='gray', vmin=0, vmax=255)
                    # plt.show()
                    # hidden_layer = model.get_layer('hidden')
                    if render:
                        self.env.render()
                print(f"Done player number {i}\nBest distance {player.farthest_x} in time {player.farthest_x_time}\n\n")
            self.update_fitness()
            self.update_breeding_probability()
            self.make_new_population()

            # rec.close()

    def update_fitness(self):
        for player in self.poplation:
            player.calculate_fittness()

    def update_breeding_probability(self):
        fitness_sum = 0
        for player in self.poplation:
            fitness_sum += player.fitness
        for player in self.poplation:
            player.breeding_probability = player.fitness / fitness_sum

    def pick_from_population(self):
        i = 0
        r = np.random.uniform()
        while r > 0:
            r = r - self.poplation[i].breeding_probability
            i += 1

        return self.poplation[i - 1]

    def make_new_population(self):
        new_pop = []
        for _ in range(self.inital_pop // 2):
            first_parent = self.pick_from_population()
            second_parent = self.pick_from_population()
            new_pop.extend(self.crossover(first_parent, second_parent))
        self.poplation = new_pop

    def __init_pop(self):
        for _ in range(self.inital_pop):
            model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(240, 256, 1)),
                tf.keras.layers.Dense(9, activation=tf.nn.relu, name='hidden',
                                      kernel_initializer=tf.keras.initializers.RandomNormal,
                                      bias_initializer=tf.keras.initializers.RandomNormal),
                tf.keras.layers.Dense(self.num_of_actions, activation=tf.nn.softmax, use_bias=False)
            ])
            self.poplation.append(Player(model))

    def crossover(self, first_parent, second_parent):
        first_parent_kernel = first_parent.model.get_layer('hidden').get_weights()[0]
        first_parent_bias = first_parent.model.get_layer('hidden').get_weights()[1]

        second_parent_kernel = second_parent.model.get_layer('hidden').get_weights()[0]
        second_parent_bias = second_parent.model.get_layer('hidden').get_weights()[1]

        first_prob = first_parent.breeding_probability
        wieghts_len = len(first_parent_kernel[0])
        crossover_point = np.random.randint(1, wieghts_len - 2)
        reverse_crossover = (wieghts_len - 1) - crossover_point

        first_child_kernel = np.append(first_parent_kernel[:, :crossover_point],
                                       second_parent_kernel[:, crossover_point:], axis=1)
        first_child_bias = np.append(first_parent_bias[:crossover_point], second_parent_bias[crossover_point:])
        first_child_weights = [first_child_kernel, first_child_bias]

        first_model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(240, 256, 1)),
                tf.keras.layers.Dense(9, activation=tf.nn.relu, name='hidden',
                                      kernel_initializer=tf.keras.initializers.RandomNormal,
                                      bias_initializer=tf.keras.initializers.RandomNormal),
                tf.keras.layers.Dense(self.num_of_actions, activation=tf.nn.softmax, use_bias=False)
            ])

        first_model.get_layer('hidden').set_weights(first_child_weights)

        second_child_kernel = np.append(first_parent_kernel[:, :reverse_crossover],
                                        second_parent_kernel[:, reverse_crossover:], axis=1)
        second_child_bias = np.append(first_parent_bias[:reverse_crossover],
                                     second_parent_bias[reverse_crossover:])
        second_child_weights = [second_child_kernel, second_child_bias]

        second_model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(240, 256, 1)),
                tf.keras.layers.Dense(9, activation=tf.nn.relu, name='hidden',
                                      kernel_initializer=tf.keras.initializers.RandomNormal,
                                      bias_initializer=tf.keras.initializers.RandomNormal),
                tf.keras.layers.Dense(self.num_of_actions, activation=tf.nn.softmax, use_bias=False)
            ])

        second_model.get_layer('hidden').set_weights(second_child_weights)

        return [Player(first_model), Player(second_model)]
