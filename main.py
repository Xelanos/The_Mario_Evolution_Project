from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import gym.wrappers.monitor as monitor
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(240, 256, 1)),
    tf.keras.layers.Dense(9, activation=tf.nn.relu,
                          kernel_initializer=tf.keras.initializers.RandomNormal,
                          bias_initializer=tf.keras.initializers.RandomNormal),
    tf.keras.layers.Dense(len(SIMPLE_MOVEMENT), activation=tf.nn.softmax)
])





for i in range(4):
    rec = monitor.video_recorder.VideoRecorder(env, path=f"vid/t{i}.mp4")
    done = True
    action = env.action_space.sample()
    for step in range(500):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(action)
        rec.capture_frame()
        grayscale_stat = tf.image.rgb_to_grayscale(state)
        grayscale_stat = tf.keras.backend.expand_dims(grayscale_stat, axis=0)
        actions = model.predict(grayscale_stat, batch_size=1)
        action = np.argmax(actions)
        # plt.imshow(grayscale_stat, cmap='gray', vmin=0, vmax=255)
        # plt.show()

        env.render()

    rec.close()

    print(f'done {i}')



if __name__ == '__main__':
    pass