import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from src.MARL.Agent import Agent
from src.MARL.Networks.DQN import DQN
from src.Utils import utils as utils
import peersim_gym.envs.PeersimEnv as pe
from src.Utils.MetricHelper import MetricHelper as mh
from src.Utils.DatasetGen import SarsaDataCollector as dg
import peersim_gym.envs.PeersimEnv as pg


class DDQNAgentMARL(Agent):
    """
    DDQN Agent is a Double Deep Q Network Agent, this agent maintains a replay buffer and a target network.
    We utilize an epsilon-greedy policy to explore the environment.

    There are some notable requirements for this agent:
    1. Because DQN is a Value-based method I need to "Hack" the actions. The output_size must be the total size of the
     Network. This will only really work for smaller Networks... For bigger Networks use A2C or PPO.
    2. actions is the Gymnasium action space, this is used to sample actions for the epsilon-gereeedy policy.

    This Class is based on the implementation by "Machine Learning Phil" in https://www.youtube.com/watch?v=wc-FxNENg9U
    and the PyTorch tutorial on DQN https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    """

    def __init__(self, input_shape, action_spaces, output_shape, batch_size, memory_max_size=10000, epsilon_start=0.7,
                 epsilon_decay=5e-4, gamma=0.7, epsilon_end=0.01, update_interval=150, learning_rate=0.7,
                 collect_data=False, save_interval=50, control_type="DQN_MARL", agents=None):
        super().__init__(input_shape, action_spaces, output_shape, memory_max_size, collect_data=collect_data)

        self.possible_agents = agents

        # Parameters:
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma

        self.batch_size = batch_size
        self.update_interval = update_interval

        # Replay Buffer
        self.memory_size = memory_max_size
        self.state_memory = {agent: np.zeros((self.memory_size, *self.input_shape), dtype=np.float32) for agent in
                             self.possible_agents}
        self.reward_memory = {agent: np.zeros(self.memory_size, dtype=np.float32) for agent in
                              self.possible_agents}
        self.action_memory = {agent: np.zeros((self.memory_size, 1), dtype=np.float32) for agent in
                              self.possible_agents}  # Hard coded, represents the target only
        self.new_state_memory = {agent: np.zeros((self.memory_size, *self.input_shape), dtype=np.float32) for agent in
                                 self.possible_agents}
        self.terminal_memory = {agent: np.zeros(self.memory_size, dtype=bool) for agent in
                                self.possible_agents}
        self.memory_counter = 0
        self.control_type = "DDQN_MARL"

        # Networks - For some reaon couldn't use the original constructor on the laptop. This has taken too much time
        # so Im hacking it a little. Fix this later.

        self.Q_values = {}
        self.target_Q_values = {}
        self.action_shape = output_shape
        for agent in self.possible_agents:
            rank = output_shape[agent]
            self.Q_values[agent] = DQN(lr=learning_rate, input_dims=self.input_shape, fc1_dims=512, fc2_dims=256, fc3_dims=128,
                                       n_actions=rank)
            self.target_Q_values[agent] = DQN(lr=learning_rate, input_dims=self.input_shape, fc1_dims=512, fc2_dims=256,
                                              fc3_dims=128, n_actions=rank)
            self.target_Q_values[agent].load_state_dict(self.Q_values[agent].state_dict())

        self.save_interval = save_interval
        for agent in self.possible_agents:
            summary(self.Q_values[agent], input_size=self.input_shape)
            summary(self.target_Q_values[agent], input_size=self.input_shape)

        self.amount_of_metrics = 50
        self.last_losses = np.zeros(self.amount_of_metrics)
        self.last_rewards = np.zeros(self.amount_of_metrics)

    def train_loop(self, env, num_episodes, print_instead=True, controllers=None, warm_up_file=None, load_weights=None,
                   results_file=None):
        # See page 14 from: https://arxiv.org/pdf/1602.01783v2.pdf

        self.mh = mh(agents=env.possible_agents, num_nodes=env.number_nodes, num_episodes=num_episodes,
                     file_name=results_file + "_result")
        self.dg = dg(agents=env.possible_agents)
        if warm_up_file is not None:
            self.warm_up(warm_up_file, env.possible_agents)

        if load_weights is not None: # Not implemented for MARL yet.
            for idx, agent in enumerate(env.possible_agents):
                agent_w = load_weights + f"_{agent}.pth.tar"
                self.Q_values[agent].load_checkpoint(agent_w)
                self.target_Q_values[agent].load_checkpoint(agent_w)

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
                targets = {agent: np.floor(self.get_action(states[idx], agent)) for idx, agent in
                           enumerate(agent_list)}
                actions = utils.make_action(targets, agent_list)

                # self.tally_actions(actions)
                self.mh.register_actions(actions)

                next_states, rewards, dones, _, info = env.step(actions)
                next_states = utils.flatten_state_list(next_states, agent_list)
                for idx, agent in enumerate(agent_list):
                    # Update history
                    self.__store_transition(states[idx], actions[agent]['neighbourIndex'], rewards[agent],
                                            next_states[idx], dones[agent], agent, idx)
                    score += rewards[agent]
                # Advance to next iter
                states = next_states

                # Learn
                last_losses = {agent: 0 for agent in agent_list}
                for agent in agent_list:
                    last_loss = self.learn(s=self.state_memory, a=self.action_memory, r=self.reward_memory,
                                           s_next=self.new_state_memory, k=step, fin=self.terminal_memory, agent=agent)
                    last_losses[agent] = last_loss if not last_loss is None else 0

                print(f'Action(e:{self.epsilon}) {actions}  -   Loss: {last_losses}  -    Rewards: {rewards}')

                if step != 0 and (step % self.update_interval == 0 or dones):
                    for agent in env.agents:
                        self.target_Q_values[agent].load_state_dict(self.Q_values[agent].state_dict())

                step += 1
                self.mh.update_metrics_after_step(rewards=rewards,
                                                  losses=last_losses,
                                                  overloaded_nodes=info[pg.STATE_G_OVERLOADED_NODES],
                                                  average_response_time=info[pg.STATE_G_AVERAGE_COMPLETION_TIMES],
                                                  occupancy=info[pg.STATE_G_OCCUPANCY],
                                                  dropped_tasks=info[pg.STATE_G_DROPPED_TASKS],
                                                  finished_tasks=info[pg.STATE_G_FINISHED_TASKS],
                                                  total_tasks=info[pg.STATE_G_TOTAL_TASKS],
                                                  consumed_energy=info[pg.STATE_G_CONSUMED_ENERGY])

            self.mh.print_action_density_episode()
            self.mh.compile_aggregate_metrics(i, step)
            print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, num_episodes, score, self.epsilon,
                                                                             self.mh.episode_average_reward(i)))
            if i % self.save_interval == 0:
                for agent in env.agents:
                    self.Q_values[agent].save_checkpoint(filename=f"DDQN_Q_value_{i}_{agent}.pth.tar", epoch=i)

        if results_file is not None:
            self.mh.store_as_cvs(results_file)
        self.mh.plot_agent_metrics(num_episodes=num_episodes, title=self.control_type, print_instead=print_instead)
        self.mh.plot_simulation_data(num_episodes=num_episodes, title=self.control_type, print_instead=print_instead)
        self.mh.clean_plt_resources()

        env.close()


    def get_action(self, observation, agent, pre_train_policy=False):
        """
        This function returns the action to take given the observation. If pre_train_policy is True, then we are
        training the policy before the agent has any (TODO knowledge of the environment).
         In this case, we just return a random action.

         Note: Due  to using BatchNorm we must use for prediction
         self.Q_value_forward().eval()
         with torch.no_grad():
            ...

        self.Q_value_forward().train()


        src: https://discuss.pytorch.org/t/how-to-make-predictions-with-a-model-that-uses-batchnorm1d/100187
        :param observation:
        :param pre_train_policy:
        :return:
        """
        if pre_train_policy:
            return np.random.choice(self.actions[agent])

        # In this case, we are using a epsilon-greedy policy
        if np.random.random() < self.epsilon:
            # print(f"(R) Exploring ({self.epsilon})")
            action = np.random.choice(self.actions[agent])
        else:
            # print("(T) Exploiting")
            #  We need to set the network to evaluation mode, because we are only predicting the action for one state,
            #  and batchnorm layers.
            #
            # See
            # https://discuss.pytorch.org/t/how-to-make-predictions-with-a-model-that-uses-batchnorm1d/100187
            # https://stackoverflow.com/questions/58447885/pytorch-going-back-and-forth-between-eval-and-train-modes
            # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

            self.Q_values[agent].eval()
            with T.no_grad():
                state = T.tensor(np.array([observation]), dtype=T.float32).to(self.Q_values[agent].device)
                actions = self.Q_values[agent].forward(state)
            self.Q_values[agent].train()
            # We get the index of the highest Q value. This is returned in a tensor, we use item() to convertit to
            # a scaler
            action = T.argmax(actions).item()
        return action

    def __store_transition(self, state, action, reward, n_state, done, agent, idx_agent):
        index = self.memory_counter % self.memory_size  # Allows overwriting old memories
        self.state_memory[agent][index] = state

        self.action_memory[agent][index] = action
        self.reward_memory[agent][index] = reward
        self.new_state_memory[agent][index] = n_state
        self.terminal_memory[agent][index] = done
        self.memory_counter += 1

    def learn(self, s, a, r, s_next, k, fin, agent):
        if self.memory_counter < self.batch_size:  # self.memory_size:
            return None
        # Select a sub-set of the memory by picking batch_size random indexes between 0 and max_mem
        max_mem = min(self.memory_counter, self.memory_size)
        # Turns out we need the batch indexes for proper array slicing...
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(s[agent][batch]).to(self.Q_values[agent].device)
        next_state_batch = T.tensor(s_next[agent][batch]).to(self.Q_values[agent].device)
        reward_batch = T.tensor(r[agent][batch]).to(self.Q_values[agent].device)
        terminal_batch = T.tensor(fin[agent][batch]).to(self.Q_values[agent].device)

        action_batch = a[agent][batch]
        q_value = self.Q_values[agent].forward(state_batch)[
            batch_index, action_batch.squeeze()]  # Q value for the action we took

        q_next_state = self.target_Q_values[agent].forward(next_state_batch)  # This is the Q value for the next state
        q_next_state[terminal_batch] = 0.0  # If we are in a terminal state, the Q value is 0, we only count the rewards
        aux = T.max(q_next_state, dim=1)
        q_value_target = reward_batch + self.gamma * aux[0]
        loss = self.Q_values[agent].lossFunction(q_value, q_value_target).to(self.Q_values[agent].device)  # Calculate the loss

        self.Q_values[agent].optimizer.zero_grad()
        # https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step
        loss.backward()  # back propagation
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end

        # In-place gradient clipping src:https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        T.nn.utils.clip_grad_value_(self.Q_values[agent].parameters(), 100)
        self.Q_values[agent].optimizer.step()  # Manually confirmed that there is some training going on. The values change at least.
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
            loss = [0 for agent in agents]
            print("Beginning warm up batch {}".format(i))
            first_idx = min(i * self.batch_size, len(states) - self.batch_size)
            print(" Covering from: {}".format(first_idx) + "  to {}".format(first_idx + self.batch_size))
            for j in range(self.batch_size):
                self.__store_transition(states[first_idx + j], actions[first_idx + j], rewards[first_idx + j],
                                        next_states[first_idx + j], dones[first_idx + j])
                idx = 0
                for agent in agents:
                    l = self.learn(s=self.state_memory, a=self.action_memory, r=self.reward_memory,
                               s_next=self.new_state_memory, k=i, fin=self.terminal_memory, agent=agent)
                    loss[idx] += l if not l is None else 0  # for the first time steps where no training is done
                    idx += 1
            print(f"Loss Total: {loss}  Loss Avg: {loss / self.batch_size}  ")  # (This is kinda irrelevant btw)
            for agent in agents:
                self.target_Q_values[agent].load_state_dict(self.Q_values[agent].state_dict())
        for agent in agents:
            self.Q_values[agent].save_checkpoint(filename="warm_up_Q_value.pth.tar", epoch=-1)
        print("Warm up complete")

    def tally_actions(self, actions):
        for worker, action in actions.items():
            self.mh.register_action(action[pe.ACTION_NEIGHBOUR_IDX_FIELD], worker)




    # def inference_loop(self, env, num_episodes, print_instead=True, controllers=None, load_weights=None, results_file=None):
    #     self.result_file = self.file_name + '_result'
    #
    #     self.mh = mh(agents=env.possible_agents, num_nodes=env.number_nodes, num_episodes=num_episodes,
    #                  file_name=results_file)
    #     self.dg = dg(agents=env.possible_agents)
    #
    #     if load_weights is not None:  # Not implemented for MARL yet.
    #         for idx, agent in enumerate(env.possible_agents):
    #             agent_w = load_weights + f"_{agent}.pth.tar"
    #             self.Q_values[agent].load_checkpoint(agent_w)
    #             self.target_Q_values[agent].load_checkpoint(agent_w)
    #     else:
    #         print("No weights to load, exiting")
    #         return
    #
    #     for i in range(num_episodes):
    #         # Prepare variables for the next run
    #         dones = [False for _ in controllers]
    #         agent_list = env.agents
    #         step = 0
    #         score = 0.0
    #
    #         # Reset the state
    #         states, _ = env.reset()
    #         states = utils.flatten_state_list(states, agent_list)
    #
    #         while not utils.is_done(dones):
    #             print(f'Step: {step}\n')
    #             # Interaction Step:
    #             targets = {agent: np.floor(self.get_action(states[idx], agent)) for idx, agent in
    #                        enumerate(agent_list)}
    #             actions = utils.make_action(targets, agent_list)
    #
    #             self.tally_actions(actions)
    #
    #             next_states, rewards, dones, _, info = env.step(actions)
    #             next_states = utils.flatten_state_list(next_states, agent_list)
    #             for idx, agent in enumerate(agent_list):
    #                 # Update history
    #                 # self.__store_transition(states[idx], actions[agent]['neighbourIndex'], rewards[agent],
    #                 #                         next_states[idx], dones[agent], agent, idx)
    #                 score += rewards[agent]
    #             # Advance to next iter
    #             states = next_states
    #
    #             # Learn
    #             last_losses = []
    #             for agent in agent_list:
    #                 # last_loss = self.learn(s=self.state_memory, a=self.action_memory, r=self.reward_memory,
    #                 #                        s_next=self.new_state_memory, k=step, fin=self.terminal_memory, agent=agent)
    #                 last_losses.append(0)
    #
    #             print(f'Action(e:{self.epsilon}) {actions}  -   Loss: {last_losses}  -    Rewards: {rewards}')
    #
    #             # if step != 0 and (step % self.update_interval == 0 or dones):
    #             #     for agent in env.agents:
    #             #         self.target_Q_values[agent].load_state_dict(self.Q_values[agent].state_dict())
    #
    #             step += 1
    #             self.mh.update_metrics_after_step(rewards=rewards,
    #                                               losses={agent: 0 for agent in
    #                                                       env.agents},
    #                                               overloaded_nodes=info[pg.STATE_G_OVERLOADED_NODES],
    #                                               average_response_time=info[pg.STATE_G_AVERAGE_COMPLETION_TIMES],
    #                                               occupancy=info[pg.STATE_G_OCCUPANCY])
    #         self.mh.compile_aggregate_metrics(i, step)
    #         print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, num_episodes, score, self.epsilon,
    #                                                                          self.mh.episode_average_reward(i)))
    #         # if i % self.save_interval == 0:
    #         #     for agent in env.agents:
    #         #         self.Q_values[agent].save_checkpoint(filename=f"DDQN_Q_value_{i}_{agent}.pth.tar", epoch=i)
    #
    #     if results_file is not None:
    #         self.mh.store_as_cvs(results_file)
    #     self.mh.plot_agent_metrics(num_episodes=num_episodes, title=self.control_type, print_instead=print_instead)
    #     self.mh.plot_simulation_data(num_episodes=num_episodes, title=self.control_type, print_instead=print_instead)
    #     self.mh.clean_plt_resources()
    #
    #     env.close()