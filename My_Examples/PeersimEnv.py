# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gymnasium as gym
import numpy as np
from Agents.Agent import Agent
from My_Examples.Utils import flattenutils as fl

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    env = gym.make("peersim_gym/PeersimEnv-v0")
    env.env.init(configs={"SIZE": "2", "CYCLES": "100"})   # TODO Untested
    # Option 2:
    # env = PeersimEnv(configs=None)  # Note: This is to avoid training stopping at  200 iterations, default of gym.
    obs, done = env.reset()
    flat_obs = fl.flatten_observation(obs)
    shape_obs_flat = np.shape(flat_obs)

    action = env.action_space.sample()
    flat_a = fl.flatten_action(action)
    shape_a_flat = np.shape(flat_a)

    print("Action Space {}".format(shape_a_flat))
    print("State Space {}".format(shape_obs_flat))
    # (taxi row, taxi column, passenger index, destination index)

    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    train = 0
    test = 1
    num_episodes = 1

    # For plotting metrics
    all_epochs = []
    all_penalties = []
    dqn_agent = Agent(input_shape=shape_obs_flat,  # Confirm this is legal!
                      output_shape=shape_a_flat,
                      action_space=env.action_space,
                      batch_size=20,
                      epsilon_start=1.0,
                      epsilon_decay=0.995,
                      epsilon_end=0.01,
                      gamma=0.95,
                      update_interval=150,
                      learning_rate=0.001)
    for i in range(0, num_episodes):
        state, _ = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        dqn_agent.train_model(env, num_episodes)

    print("Training finished.\n")

    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
