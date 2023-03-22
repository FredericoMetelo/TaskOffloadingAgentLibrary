# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym

from Agents.Agent import Agent

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import gym

    env = gym.make("CartPole-v1").env # Note: This is to avoid training stopping at  200 iterations, default of gym.

    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))
    # (taxi row, taxi column, passenger index, destination index)

    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    train = 0
    test = 1
    num_episodes = 100

    # For plotting metrics
    all_epochs = []
    all_penalties = []
    dqn_agent = Agent(input_shape=env.observation_space.shape,  # Confirm this is legal!
                      output_shape=env.action_space.n,
                      action_space=env.action_space,
                      batch_size=20,
                      epsilon_start=1.0,
                      epsilon_decay=0.995,
                      epsilon_end=0.01,
                      gamma=0.95,
                      update_interval=150,
                      learning_rate=0.001)
    for i in range(1, 100001):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False


        dqn_agent.train_model(env, num_episodes)


    print("Training finished.\n")

    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
