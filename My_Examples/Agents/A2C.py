# Inspired by DanielPalaio's Project in:
# https://github.com/DanielPalaio/LunarLander-v2_DeepRL/blob/main/DQN/replay_buffer.py

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import placeholder, mean, square, function, sqrt, exp, epsilon, log
from tensorflow.python.keras.layers import Lambda

from My_Examples.Agents.ReplayMemory import ReplayMemory
from My_Examples.Utils import flattenutils as fl
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

import peersim_gym


# Sauces:
# https://datascience.stackexchange.com/questions/61707/policy-gradient-with-continuous-action-space
#   - Explains how we utilize normal distributions for the actions
# https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
#   - Structure of an actor-critic environment
# https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
#   - Clarify the losses and how the Advantage function is computed.
# https://github.com/nyck33/openai_my_implements/blob/master/continuous/Pendulum-v0/a2cPendulumColabTested.py
#   - Example repository



class A2C:
# Special thanks to nyck33 for the examples he provided on the A2C, https://github.com/nyck33/openai_my_implements/blob/master/continuous/Pendulum-v0/a2cPendulumColabTested.py
    def __init__(self, input_shape, action_space, output_shape, batch_size=500, epsilon_start=0.7, epsilon_decay=0.01,
                 gamma=0.7,
                 epsilon_end=0.01, update_interval=150, learning_rate=0.7):
        # Parameters:
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.input_shape = input_shape
        self.action_shape = output_shape
        self.learning_rate = learning_rate
        self.update_interval = update_interval
        self.action_space = action_space
        self.actions = output_shape  # There are 2 possible outputs.
        self.step = 0

        self.experience = ReplayMemory(self.batch_size, input_shape=input_shape)
        # Note: Make shure same weight vector for both networks
        self.actor, self.critic = self._Network(input_shape=input_shape, output_shape=output_shape, layer_size_vector=[24, 24],
                                   learning_rate=self.learning_rate)

        self.optimizer = [self._critic_optimizer(), self._actor_optimizer()]

    def _Network(self, input_shape, output_shape, learning_rate, layer_size_vector=256):
        # How to build an actor critic for a continuous output function?
        # A policy-gradient method, which the actor is, requires a probability distribution for the actions. To achieve this
        # we outpout the parameters for a Normal distribution over the action space.
        # In this particular case we have two possible outputs and therefore we will be approximating two normal
        # distributions.
        # src: https://github.com/nyck33/openai_my_implements/blob/master/continuous/Pendulum-v0/a2cPendulumColabTested.py

        # Furthermore we utilize a technique called hybrid actor-critic: https://arxiv.org/pdf/1903.01344.pdf
        # TODO: Explain this after reading the paper.

        # Read this paper as well: https://arxiv.org/pdf/1509.02971.pdf

        input = Input(shape=input_shape)
        common = Dense(layer_size_vector, activation="relu")(input)

        actor_mean = Dense(output_shape, activation='softplus', kernel_initializer="he_uniform")(common)
        # Note: Softplus outputs between [0, inf], this part deviates form the guide. May be a breaking point
        actor_std_0 = Dense(output_shape, activation="softlplus", kernel_initializer="he_uniform")(common)
        actor_std = Lambda(lambda x: x + 0.0001)(actor_std_0)  # Ensures std is not 0

        critic_state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(common) \
            # Will approximate the value function A(s) = r + yV(s') - V(s)

        actor = Model(inputs=input, outputs=(actor_std, actor_mean))
        critic = Model(inputs=input, outputs=critic_state_value)

        # See bstriner's comment: https://github.com/keras-team/keras/issues/6124
        # Basically model._make_predict_function() builds and compiles the model for the GPU.
        # Didn't find anywhere why use this over compile.
        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()
        return actor, critic




    def _actor_optimizer(self, state, action, reward, new_state, done):
        action = placeholder(shape=(None, 1))
        advantages = placeholder(shape=(None, 1))
        mean, std = self.actor.output
        var = square(std)
        pdf = 1 / (sqrt(2 * np.pi) * std) * exp(-square(action - mean)/(2 * var)) # Goog old gaussian PDF.
        log_pdf = log(pdf + epsilon())
            # The author of the repo was folloswing a guide that OpenAI took down, the link still exists but is dead.
            # He is also not shure why the original guide did this log(pdf)

        entropy = tf.python.keras.backend.sum(0.5 * (log(2. * np.pi * var) + 1.))

        exp_v = log_pdf * advantages
        # entropy is made small before added to exp_v
        exp_v = tf.python.keras.backend.sum(exp_v + 0.01 * entropy)
        # loss is a negation
        actor_loss = -exp_v

        # use custom loss to perform updates with Adam, ie. get gradients
        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(params=self.actor.trainable_weights, loss=actor_loss)
        # adjust params with custom train function
        train = function([self.actor.input, action, advantages], [], updates=updates)
        # return custom train function
        return train

    def _critic_optimizer(self):
        # Will accomodate the parameters
        discounted_reward = placeholder(shape=(None, 1))
        value = self.critic.output
        # Loss: E_t[(G_t - V(s_t))^2] # Standard MSE loss used to train the value network.
        loss = mean(square(discounted_reward - value))
        optimizer = Adam(lr=self.learning_rate) # Note: Separate into two learning rates. One for Critic another for Actor
        updates = optimizer.get_updates(params=self.critic.trainable_weights, loss=loss)
        train = function([self.critic.input, discounted_reward], [], updates=updates)
        return train
    def get_action(self, observation):
        mean, std = self.actor.predict(observation)
        var = square(std)
        epsilon = np.random.randn(fl.flatten_action(self.action_space).size) # randn stands for randomly sample  a normal(mean = 1, std = 0).
        action = mean + std * epsilon # do those Normal distribution shenanigans to change the distribution
        action_target = np.clip(action[0], 0,  5) # TODO!! Critical!!!! Magic numbers for now
        action_amount = np.clip(action[1], 0, 10)
        return np.concatenate((action_target, action_amount), axis=0)

    def __train(self, s, a, r, s_next, done):
        target = np.zeroes((1, 1))
        advantages = np.zeros((1, self.action_shape))

        value = self.critic.predict(s)[0]
        next_value = self.critic.predict(s_next)[0]

        if done:
            advantages[0] = r - value
            target[0][0] = r
        else:
            advantages[0] = r + self.gamma * s_next - s  # Literally the advantages of the actions taken.
            target[0][0] = r + self.gamma * next_value # Literally the state-value target from bellman equations.

        self.optimizer[0]([s, a, advantages])
        self.optimizer[1]([s, target])

    def train_model(self, env, num_episodes, print_instead=True):
        scores, episodes, avg_scores, obj, avg_episode = [], [], [], [], []
        f = 0
        for i in range(num_episodes):
            done = False
            score = 0.0
            state, _ = env.reset()
            state = fl.flatten_observation(state)
            step = 0
            while not done:
                action, type = self.get_action(state)
                print("\nStep: " + str(step) + " => " + str(action) + ":")
                temp = env.step(fl.deflatten_action(np.floor(action)))
                new_state, reward, done, _, _ = temp
                score += reward
                new_state = fl.flatten_observation(new_state)
                self.__train(state, action, reward, new_state, done)
                state = new_state
                step += 1
            avg_episode.append(score / step)
            scores.append(score)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, num_episodes, score, self.epsilon,
                                                                             avg_score))
            if avg_score >= 200.0 and score >= 250:
                self.q_net.save(("saved_networks/dqn_model{0}".format(f)))
                self.q_net.save_weights(("saved_networks/dqn_model{0}/net_weights{0}.h5".format(f)))
                f += 1
        self.__plot(episodes, scores=scores, avg_scores=avg_scores, per_episode=avg_episode, print_instead=print_instead)
        env.close()

    def __plot(self, x, scores, avg_scores, per_episode, print_instead=False):
        # Setup for print
        fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True)  # Create 1x3 plot

        # Print the metrics:
        ax[0].set_title("Scores")
        ax[0].plot(x, scores)

        ax[1].set_title("Average Scores")
        ax[1].plot(x, avg_scores)

        ax[2].set_title("Average Score in Episode")
        ax[2].plot(x, per_episode)

        if print_instead:
            plt.savefig(f"/Plots/plt_{self.control_type}")
        else:
            plt.show()
        return