import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from src.Agents.Networks.DQN import DQN
from src.Utils import utils as utils
from src.Agents.Agent import Agent
import peersim_gym.envs.PeersimEnv as pe
from src.Utils.MetricHelper import MetricHelper as mh
from src.Utils.DatasetGen import SarsaDataCollector as dg
import peersim_gym.envs.PeersimEnv as pg


class DDQNAgent(Agent):
    """
    DDQN Agent is a Double Deep Q Network Agent, this agent maintains a replay buffer and a target network.
    We utilize an epsilon-greedy policy to explore the environment.

    There are some notable requirements for this agent:
    1. Because DQN is a Value-based method I need to "Hack" the actions. The output_size must be the total size of the
     Network. This will only really work for smaller Networks... For bigger Networks use A2C or PPO.
    2. actions is the Gymnasium action space, this is used to sample actions for the epsilon-gereeedy policy.

    This Class is based on the implementation by "Machine Learning Phil" in https://www.youtube.com/watch?v=wc-FxNENg9U
    """

    def __init__(self, input_shape, action_space, output_shape, batch_size, memory_max_size=500, epsilon_start=0.7,
                 epsilon_decay=5e-4, gamma=0.7, epsilon_end=0.01, update_interval=150, learning_rate=0.7,
                 control_type="DQN"):
        super().__init__(input_shape, action_space, output_shape, memory_max_size)
        # Parameters:
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma

        self.batch_size = batch_size
        self.update_interval = update_interval

        # Replay Buffer
        self.memory_size = memory_max_size
        self.state_memory = np.zeros((self.memory_size, *self.input_shape), dtype=np.float32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.action_memory = np.zeros((self.memory_size, 1), dtype=np.float32)  # Hard coded, represents the target only
        self.new_state_memory = np.zeros((self.memory_size, *self.input_shape), dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)
        self.memory_counter = 0
        self.control_type = "DDQN"

        # Networks - For some reaon couldn't use the original constructor on the laptop. This has taken too much time
        # so Im hacking it a little. Fix this later.
        self.Q_value = DQN(lr=learning_rate, input_dims=self.input_shape, fc1_dims=256, fc2_dims=256,
                           n_actions=self.action_shape)
        self.target_Q_value = DQN(lr=learning_rate, input_dims=self.input_shape, fc1_dims=256, fc2_dims=256,
                                  n_actions=self.action_shape)
        summary(self.Q_value, input_size=self.input_shape)
        summary(self.target_Q_value, input_size=self.input_shape)

        self.amount_of_metrics = 50
        self.last_losses = np.zeros(self.amount_of_metrics)
        self.last_rewards = np.zeros(self.amount_of_metrics)

    def train_loop(self, env, num_episodes, print_instead=True, controllers=None, warm_up_file=None, load_weights=None):
        # See page 14 from: https://arxiv.org/pdf/1602.01783v2.pdf
        scores, episodes, avg_scores, obj, avg_episode = [], [], [], [], []
        steps_per_return = 5

        last_loss = None
        cumulative_reward = 0
        avg_reward = 0
        self.mh = mh(agents=env.possible_agents, num_nodes=env.number_nodes, num_episodes=num_episodes)
        self.dg = dg(agents=env.possible_agents)
        if warm_up_file is not None:
            self.warm_up(warm_up_file, env.possible_agents)

        if load_weights is not None:
            self.Q_value.load_checkpoint(load_weights)
            self.target_Q_value.load_checkpoint(load_weights)

        for i in range(num_episodes):
            # Prepare variables for the next run
            dones = [False for _ in controllers]
            agent_list = env.agents
            step = 0
            score = 0.0

            # Reset the state
            states, _ = env.reset()
            states = utils.flatten_state_list(states, agent_list)

            while not utils.is_done(dones):
                print(f'Step: {step}\n')
                # Interaction Step:
                targets = {agent: np.floor(self.get_action(np.array([states[idx]]))) for idx, agent in
                           enumerate(agent_list)}
                actions = utils.make_action(targets, agent_list)

                next_states, rewards, dones, _, info = env.step(actions)
                next_states = utils.flatten_state_list(next_states, agent_list)
                for idx, agent in enumerate(agent_list):
                    # Update history
                    self.__store_transition(states[idx], actions[agent]['neighbourIndex'], rewards[agent],
                                            next_states[idx], dones[agent])
                    score += rewards[agent]
                # Advance to next iter
                states = next_states

                # Update metrics
                last_loss = self.learn(s=self.state_memory, a=self.action_memory, r=self.reward_memory,
                                       s_next=self.new_state_memory, k=step, fin=self.terminal_memory)

                print(f'Action(e:{self.epsilon}) {actions}  -   Loss: {last_loss}  -    Rewards: {rewards}')

                if step % steps_per_return == 0 or dones:
                    self.target_Q_value.load_state_dict(self.Q_value.state_dict())

                # TODO This way of computing doesn't make sense for now. But with the distributed agents it will.
                #  +1 point for how ugly this looks
                self.mh.update_metrics_after_step(rewards=rewards,
                                                  losses={agent: last_loss if not last_loss is None else 0 for agent in
                                                          env.agents},
                                                  overloaded_nodes=info[pg.STATE_G_OVERLOADED_NODES],
                                                  average_response_time=info[pg.STATE_G_AVERAGE_COMPLETION_TIMES],
                                                  occupancy=info[pg.STATE_G_OCCUPANCY])
                # self.get_stats(last_loss, score, avg_reward, cumulative_reward, step, step, i, env)
                step += 1
            self.mh.compile_aggregate_metrics(i, step)
            print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, num_episodes, score, self.epsilon,
                                                                             self.mh.episode_average_reward(i)))

        self.mh.plot_agent_metrics(num_episodes=num_episodes, title=self.control_type,
                                   print_instead=print_instead)
        self.mh.plot_simulation_data(num_episodes=num_episodes, title=self.control_type, print_instead=print_instead)
        self.mh.clean_plt_resources()

        env.close()

    def get_action(self, observation, pre_train_policy=False):
        """
        This function returns the action to take given the observation. If pre_train_policy is True, then we are
        training the policy before the agent has any (TODO knowledge of the environment).
         In this case, we just return a random action.
        :param observation:
        :param pre_train_policy:
        :return:
        """
        if pre_train_policy:
            return np.random.choice(self.actions)

        # In this case, we are using a epsilon-greedy policy
        if np.random.random() < self.epsilon:
            print("Exploring")
            action = np.random.choice(self.actions)
        else:
            print("Exploiting")
            # We want to use the target network to get the action, network is in the device. So we send the observation
            # there as well.
            state = T.tensor(np.array([observation]), dtype=T.float32).to(self.Q_value.device)
            actions = self.Q_value.forward(state)
            # We get the index of the highest Q value. This is returned in a tensor, we use item() to convertit to
            # a scaler
            action = T.argmax(actions).item()
        return action

    def __store_transition(self, state, action, reward, n_state, done):
        index = self.memory_counter % self.memory_size  # Allows overwriting old memories
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = n_state
        self.terminal_memory[index] = done
        self.memory_counter += 1

    def learn(self, s, a, r, s_next, k, fin):
        if self.memory_counter < self.memory_size:
            return None
        # We need to zero the gradient optimizer in Pytorch first
        self.Q_value.optimizer.zero_grad()

        # Select a sub-set of the memory by picking batch_size random indexes between 0 and max_mem
        max_mem = min(self.memory_counter, self.memory_size)
        # Turns out we need the batch indexes for proper array slicing...
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(s[batch]).to(self.Q_value.device)
        next_state_batch = T.tensor(s_next[batch]).to(self.Q_value.device)
        reward_batch = T.tensor(r[batch]).to(self.Q_value.device)
        terminal_batch = T.tensor(fin[batch]).to(self.Q_value.device)

        action_batch = a[batch]
        # This does not need to be a tensor, we use this to get the target Q value for the aciton we took.

        q_value = self.Q_value.forward(state_batch)[batch_index, action_batch.squeeze()]  # Q value for the action we took

        q_next_state = self.target_Q_value.forward(next_state_batch)  # This is the Q value for the next state
        q_next_state[terminal_batch] = 0.0  # If we are in a terminal state, the Q value is 0, we only count the rewards

        q_value_target = reward_batch + self.gamma * T.max(q_next_state, dim=1)[0]
        # [0] here is because the torch.max() returns a tuple (values, indexes)

        loss = self.Q_value.loss(q_value, q_value_target).to(self.Q_value.device)  # Calculate the loss
        loss.backward()  # back propagation
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end
        return loss.item()

    def get_stats(self, last_loss, last_reward, avg_reward, cumulative_reward, total_steps, step, episode_number, env):
        index = total_steps % self.amount_of_metrics
        self.last_losses[index] = last_loss
        self.last_rewards[index] = avg_reward

        print(f'Episode: {episode_number}')
        print(f'Current Step: {step}')
        print(f'Last loss: {last_loss}')
        print(f'Last reward: {last_reward}')
        print(f'Last Reward Components: {env.last_reward_components}')
        print(f'Average loss (from last 50 steps): {sum(self.last_losses) / len(self.last_losses)}')
        print(f'Average reward (from last 50 steps): {sum(self.last_rewards) / len(self.last_rewards)}')
        print(f'Cumulative reward: {cumulative_reward}')
        print(f'Current epsilon: {self.epsilon}')
        print(f'Total steps: {total_steps}')

    def warm_up(self, dataset_file, agents):
        """
        This function is used to warm up the agent. This is done by loading a dataset and running the agent on it.
        :param dataset_file: The file containing the dataset.
        :return:
        """
        data = self.dg.load_from_csv_to_arrays(dataset_file)
        states, actions, rewards, next_states, dones = data
        no_batches = len(states) // self.batch_size
        for i in range(no_batches):  # len(states)):
            loss = 0
            print("Beginning warm up batch {}".format(i))
            first_idx = min(i * self.batch_size, len(states) - self.batch_size)
            print(" Covering from: {}".format(first_idx) + "  to {}".format(first_idx + self.batch_size))
            for j in range(self.batch_size):
                self.__store_transition(states[first_idx + j], actions[first_idx + j], rewards[first_idx + j], next_states[first_idx + j], dones[first_idx + j])
                l = self.learn(s=self.state_memory, a=self.action_memory, r=self.reward_memory,
                       s_next=self.new_state_memory, k=i, fin=self.terminal_memory)
                loss += l if not l is None else 0 # for the first time steps where no training is done
            print(f"Loss Total: {loss}  Loss Avg: {loss/self.batch_size}  ") # (This is kinda irrelevant btw)
            self.target_Q_value.load_state_dict(self.Q_value.state_dict())
        self.Q_value.save_checkpoint(filename="warm_up_Q_value.pth.tar", epoch=-1)
        print("Warm up complete")
