# Inspired by DanielPalaio's Project in:
# https://github.com/DanielPalaio/LunarLander-v2_DeepRL/blob/main/DQN/replay_buffer.py

from tensorflow.keras.optimizers import Adam

from DeprecatedAgents.ReplayMemory import ReplayMemory
from src.Utils import utils as fl
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt


# For the actor Critic: https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic

def Network(input_shape, output_shape, learning_rate, no_layers=3, layer_size_vector=[256, 256, 256]):
    # TODO > One possible problem with this is that I can't increase the number of nodes. Because it would change the
    # TODO > shape of the Input layer. What to do?
    # I see two options:
    #   Option 1: Have activation layer with two neurons and activation linear:
    #   This will allow the output to be directly the values of the action.
    #       Pros:
    #       - Can scale indefinitely.
    #       - No problems with the size of the output of truly massive networks
    #       Cons:
    #       - Need to find a way to truncate the output value to be within the indexes of the nodes in the network.
    #       (Negative reward for values outside?). Same for the amount being offloaded. ==> As I suspected solution is
    #       rounding.
    #
    #   Option 2: Have the activation layer have as many neurons as there are nodes in the network. The output value of
    #   the neurons will be the amount to offload.
    #       Pros:
    #       - Could have multiple offloads. (Would need changes in the Reward Function and the way the
    #       environment works)
    #       - No problems with limiting the indexes of the nodes. The reward would eventually truncate the amount of
    #       tasks being offloaded.
    #       Cons:
    #       - No way to have a dynamic network with nodes joining.
    #
    #   Option 3: Scale the node ID's to be between 0 and 1 and the number of tasks to offload to be between 0 and MAX_Q
    #   then use two sigmoid activated output neurons to output the results and scale them back.
    #       Pros:
    #       - Scales the outputs from the beginning.
    #       Cons:
    #       - Truncates the network. Can't be bigger than the number of nodes it started at.

    input = Input(shape=input_shape)
    i = 0
    for layer_size in layer_size_vector:
        if i == 0:
            layer = Dense(4, activation="relu")(input)
        else:
            layer = Dense(layer_size, activation="relu")(layer)
        i += 1
    # I use a linear layer because i consider the actions to be taken as a pair
    # of type (target node, number of actions to take.
    output = Dense(output_shape[0], activation="linear")(layer)  # ReLu instead of Linear.
    # Optimal way to handle this? No. Do I care? yes. Hotel? trivago.
    model = Model(inputs=input, outputs=output, name="Net")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse')
    model.summary()
    return model


class DQN:

    def __init__(self, input_shape, action_space, output_shape, batch_size=500, epsilon_start=0.7, epsilon_decay=0.01,
                 gamma=0.7,
                 epsilon_end=0.01, update_interval=150, learning_rate=0.7):
        # Parameters:
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_interval = update_interval
        self.action_space = action_space
        self.actions = output_shape  # There are 2 possible outputs.
        self.step = 0

        self.experience = ReplayMemory(self.batch_size, input_shape=input_shape)
        # Note: Make shure same weight vector for both networks
        self.policy_qnet = Network(input_shape=input_shape, output_shape=output_shape, layer_size_vector=[24, 24],
                                   learning_rate=self.learning_rate)
        self.target_qnet = Network(input_shape=input_shape, output_shape=output_shape, layer_size_vector=[24, 24],
                                   learning_rate=self.learning_rate)

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
    def epsilon_greedy_policy(self, observation):
        if np.random.random() < self.epsilon:
            action = fl.flatten_action(self.action_space.sample())
            action_type = "Explore"

        else:
            state = np.array([observation])
            actions = self.policy_qnet.predict(
                state)  # I assume that the memory stores the flattened versions of the arrays
            action = actions[0]  # tf.math.argmax(actions, axis=1).numpy()[0]
            action_type = "Exploit"
        return action, action_type  # In my specific case this would not be needed. But I will clean stuff up latter, for now i want to see it running properly

    def store_experience(self, state, action, reward, new_state, done):
        self.experience.store_tuples(state, action, reward, new_state, done)

    def train(self):
        if self.experience.counter < self.experience.size:  # Can't run a batch yet
            return

        # Update Target
        if self.step % self.update_interval == 0:
            self.target_qnet.set_weights(self.policy_qnet.get_weights())

        sarsa_tuples = self.experience.sample_buffer(self.batch_size)
        # Prepare "labels" aka:
        # Q (s, a) <- Q (s, a) + alpha*( R + gamma*max_a( Q(s', a') ) - Q (s, a) )
        #
        state_batch = [sarsa[0] for sarsa in sarsa_tuples]
        q_start = self.policy_qnet.predict_on_batch(np.array(state_batch))  # Q (s, a)

        new_state_batch = [sarsa[3] for sarsa in sarsa_tuples]
        q_next = self.target_qnet.predict_on_batch(np.array(new_state_batch))  # Q (s', a')

        # q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()  # max_a ( Q (s' a') )
        # q_target = np.copy(q_start)  # Q (s, a)
        X = []
        Y = []
        for idx, (s, a, r, s2, d) in enumerate(sarsa_tuples):
            if not d:
                target_q_val = r + self.gamma * np.max(q_next[idx])
            else:
                target_q_val = r
            current_qs = q_next[idx]
            X.append(s)
            Y.append(current_qs)
            # q_target[idx] = target_q_val  # TD target: R + max_a( Q(s', a))
        # I am not entirely shure about this. The basis for doing it like this is that I'am computing a QNet that
        # outputs continuous actions. Therefore, there won't be multiple actions at the end and I only need the one
        # Q-value. Does this make sense mathematically? No Clue.
        self.policy_qnet.fit(np.array(X), np.array(Y), batch_size=self.batch_size, verbose=0, shuffle=True)

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end
        self.step += 1

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
                action, type = self.epsilon_greedy_policy(state)
                print("\nStep: " + str(step) + " => " + type + ":")
                temp = env.step(fl.deflatten_action(np.floor(action)))
                new_state, reward, done, _, _ = temp
                score += reward
                new_state = fl.flatten_observation(new_state)
                self.store_experience(state, action, reward, new_state, done)
                state = new_state
                self.train()
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


