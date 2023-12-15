# Inspired by DanielPalaio's Project in:
# https://github.com/DanielPalaio/LunarLander-v2_DeepRL/blob/main/DQN/replay_buffer.py
from tensorflow import divide
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import mean, square, exp, epsilon, log, constant
from tensorflow.python.keras.layers import Lambda

from src.Agents.Agent import Agent
from DeprecatedAgents.ReplayMemory import ReplayMemory
from src.Utils import utils as fl
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense


# Sauces:
# https://datascience.stackexchange.com/questions/61707/policy-gradient-with-continuous-action-space
#   - Explains how we utilize normal distributions for the actions
# https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
#   - Structure of an actor-critic environment
# https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
#   - Clarify the losses and how the Advantage function is computed.
# https://github.com/nyck33/openai_my_implements/blob/master/continuous/Pendulum-v0/a2cPendulumColabTested.py
#   - Example repository

class A2C(Agent):
    # Special thanks to nyck33 for the examples he provided on the A2C, https://github.com/nyck33/openai_my_implements/blob/master/continuous/Pendulum-v0/a2cPendulumColabTested.py
    def __init__(self, input_shape, action_space, output_shape, batch_size=500, epsilon_start=0.7, epsilon_decay=0.01,
                 gamma=0.7,
                 epsilon_end=0.01, update_interval=150, learning_rate=0.7):
        super().__init__(input_shape, action_space, output_shape, batch_size, epsilon_start, epsilon_decay, gamma,
                         epsilon_end, update_interval, learning_rate)

        self.control_type = "A2C"
        self.experience = ReplayMemory(self.batch_size, input_shape=input_shape)
        # Note: Make shure same weight vector for both networks
        self.actor, self.critic = self._Network(input_shape=input_shape, output_shape=output_shape, layer_size=256,
                                                learning_rate=self.learning_rate)

        # self.optimizer = [self._critic_optimizer(), self._actor_optimizer()]
        self.actor_optimizer = Adam(lr=self.learning_rate)
        self.critic_optimizer = Adam(lr=self.learning_rate)

    def _Network(self, input_shape, output_shape, learning_rate, layer_size=256):
        # How to build an actor critic for a continuous output function?
        # A policy-gradient method, which the actor is, requires a probability distribution for the actions. To achieve this
        # we outpout the parameters for a Normal distribution over the action space.
        # In this particular case we have two possible outputs and therefore we will be approximating two normal
        # distributions.
        # src: https://github.com/nyck33/openai_my_implements/blob/master/continuous/Pendulum-v0/a2cPendulumColabTested.py

        # Furthermore we utilize a technique called hybrid actor-critic: https://arxiv.org/pdf/1903.01344.pdf
        # (Actually we don't anymore, but this might be interesting for latter soo I'm leaving this here)

        # Read this paper as well: https://arxiv.org/pdf/1509.02971.pdf

        input = Input(shape=input_shape, name="input_actor")
        common = Dense(layer_size, activation="relu", name='hidden_actor', trainable=True, dtype=tf.float64)(input)
        actor_mean = Dense(output_shape[0], activation='softplus', kernel_initializer="he_uniform", name='actor_mean', trainable=True, dtype=tf.float64)(
            common)
        # Note: Softplus outputs between [0, inf], this part deviates form the guide. May be a breaking point
        actor_std_0 = Dense(output_shape[0], activation="softplus", kernel_initializer="he_uniform", name='std_0', trainable=True, dtype=tf.float64)(
            common)
        actor_std = Lambda(lambda x: x + 0.0001, name='std', dtype=tf.float64)(actor_std_0)
        # Ensures std is not 0, we will be dividing stuff by std.

        input2 = Input(shape=input_shape, name="input_critic")
        common2 = Dense(layer_size, activation="relu", name='hidden_critic', trainable=True)(input2)
        critic_state_value = Dense(1, activation='linear', kernel_initializer='he_uniform', name='state-value', trainable=True, dtype=tf.float64)(common2)
        # Will approximate the value function A(s) = r + yV(s') - V(s)

        actor = Model(inputs=input, outputs=(actor_std, actor_mean))
        critic = Model(inputs=input2, outputs=critic_state_value)

        # See bstriner's comment: https://github.com/keras-team/keras/issues/6124
        # Basically model._make_predict_function() builds and compiles the model for the GPU.
        # Didn't find anywhere why use this over compile.
        actor.make_predict_function()
        critic.make_predict_function()


        actor.summary()
        critic.summary()
        return actor, critic

    def get_action(self, observation):
        mean, std = self.actor.predict(observation)
        var = square(std)
        epsilon = np.random.randn(fl.flatten_action(self.action_space).size)
        # randn stands for randomly sample  a normal(mean = 1, std = 0).
        action = mean + std * epsilon
        # do those Normal distribution shenanigans to change the distribution
        action_target = np.asarray([np.clip(action[0][0], 0, 5)])  # TODO!! Critical!!!! Magic numbers for now

        return np.rint(action_target)

    def learn(self, s, a, r, s_next, k, fin):
        # Preprocessing
        states = np.array(s)
        actions = np.array(a)
        actions = np.reshape(actions, (actions.shape[0], self.action_shape))  # This line might be redundant
        discounted_rewards = self._discount_rewards(rewards=r)
        discounted_rewards = tf.convert_to_tensor(np.reshape(discounted_rewards, (discounted_rewards.shape[0], 1)))
        next_states = np.array(s_next)
        dones = tf.convert_to_tensor(np.reshape(np.logical_not(np.array(fin)).astype(float), (states.shape[0], 1)), dtype=tf.float64)
        pows = np.reshape(np.arange(0, states.shape[0]), (states.shape[0], 1))
        gammas = tf.convert_to_tensor(np.ones(discounted_rewards.shape) * self.gamma)

        pows = tf.pow(gammas, pows)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            values = self.critic(states)
            next_values = self.critic(next_states)
            normal_parameters = self.actor(states, training=True)
            advantages, targets = self._compute_advantages_targets(discounted_rewards, pows=pows, values=values, next_values=next_values, gammas=gammas, dones=dones)
            probs, log_probs, entropies = self._action_probabilities(actions, normal_parameters)
            # Compute the objectives
            # See: https://github.com/RichardMinsooGo-RL-Gym/Bible_4_PI_TF2_A_ActorCritic_Policy_Iterations/blob/main/TF2_A_PI_44_A3C.py
            policy_loss = tf.reduce_mean(log_probs * advantages, 0)
            entropy_loss = tf.reduce_mean(entropies, 0)
            actor_loss = -(policy_loss + entropy_loss * 0.005)
            critic_loss = mean(square(discounted_rewards - values))

        grads1 = tape1.gradient(actor_loss, self.actor.trainable_weights)  # break them apart first??
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_weights))
        self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_weights))
        return

    def _discount_rewards(self, rewards):
        """
        This method will compute an array with the discounted reward for each step.
        If the agent got rewards=[4 4 2] in the last three steps, and gamma=0.5 then the return of this array is
        [2 4 6]
        :param rewards:
        :return:
        """
        discounted_rewards = []
        sum_reward = 0
        rewards.reverse()
        for r in rewards:
            sum_reward = r + self.gamma * sum_reward
            discounted_rewards.append(sum_reward)
        discounted_rewards = np.array(discounted_rewards)
        return discounted_rewards

    def _action_probabilities(self, actions, normal_parameters):


        k_0 = tf.convert_to_tensor(np.ones( actions.shape), dtype=tf.float64)
        k_1 = tf.convert_to_tensor(np.ones(actions.shape) * 2 * np.pi, dtype=tf.float64)

        means, stds = normal_parameters
        vars = tf.square(stds)

        f_0 = (k_1 * stds)
        f_1 = divide(constant(1, dtype=tf.float64), f_0)
        f_2 = exp(-1 * square(actions - means) / (2 * vars))
        probs = f_1 * f_2  # Good old gaussian PDF.

        log_probs = log(probs + epsilon())
        entropies = 0.5 * (log(k_1 * vars) + k_0)


        # for (action, mean, std) in zip(actions, means, stds):  # TODO Finish adding the parameters.
        #     # mean, std = (parameters[0], parameters[1])
        #     var = square(std)
        #
        #
        #     f_0 = (k_1 * std)
        #     f_1 = divide(constant(1), f_0)
        #     f_2 = exp(-1 * square(action - mean) / (2 * var))
        #
        #     pdf = f_1 * f_2  # Good old gaussian PDF.
        #      # The problem is that the network predicts the parameters, and the action is taken from said parameters.
        #      # I only pass the action to this method. I have to pass both the action and the parameters I used for the prediction.
        #     log_pdf = log(pdf + epsilon())
        #
        #
        #     entropy = 0.5 * (log(k_1 * var) + k_0)
        #
        #     probs.append(pdf[0])
        #     log_probs.append(log_pdf[0])
        #     entropies.append(entropy[0])

        return probs, log_probs, entropies

    def _compute_advantages_targets(self, discounted_rewards, pows, values, next_values, gammas, dones):
        advantages = discounted_rewards + dones * (pows * next_values) - values
        targets = discounted_rewards - dones * (self.gamma * next_values)
        # k = 0
        # for (y_reward, value, next_value) in zip(discounted_rewards, values, next_values):
        #     if k == 0:
        #         advantage = y_reward - value
        #         target = y_reward
        #
        #         advantages.append(advantage)
        #         targets.append(target)
        #     else:
        #         advantage = y_reward + tf.math.pow(self.gamma, k) * next_value - value
        #         target = y_reward - self.gamma * next_value
        #
        #         advantages.append(advantage)
        #         targets.append(target)
        return advantages, targets

def is_done(bool_array):
    return all(bool_array)
