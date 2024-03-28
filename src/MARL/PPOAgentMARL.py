import numpy as np
import peersim_gym.envs.PeersimEnv as pe
import peersim_gym.envs.PeersimEnv as pg
from peersim_gym.envs.PeersimEnv import PeersimEnv

from src.MARL.Agent import Agent
from src.MARL.Networks.PPO import PPO
from src.Utils import utils
from src.Utils.MetricHelper import MetricHelper as mh


class PPOAgentMARL(Agent):
    """
    PPO Agent
    Sources:
    - https://www.youtube.com/watch?v=hlv79rcHws0
    - https://spinningup.openai.com/en/latest/algorithms/ppo.html
    - https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, input_shape, action_space, output_shape, agents, learning_rate=0.7, gamma=0.4,
                 steps_for_return=150, policy_clip=0.1, batch_size=64, N=2048, gae_lambda=0.95,
                 collect_data=False, save_interval=50, control_type="PPO"):
        super().__init__(input_shape, action_space, output_shape, learning_rate, collect_data=collect_data)
        self.last_val = {agent: 0 for agent in agents}
        self.last_prob = {agent: 0 for agent in agents}
        self.steps_for_return = steps_for_return
        self.gamma = gamma
        self.control_type = control_type
        self.possible_agents = agents
        self.save_interval = save_interval

        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.N = N

        self.PPOs = {}
        self.action_shape = output_shape
        if self.steps_for_return < self.batch_size:
            print("Steps for return is smaller than the batch size, setting the steps for return to the batch size")
            self.steps_for_return = self.batch_size
        for agent in self.possible_agents:
            rank = output_shape[agent]
            self.PPOs[agent] = PPO(lr=learning_rate, input_dims=self.input_shape, fc1_dims=512, fc2_dims=256, fc3_dims=128,
                                    policy_clip=0.1, batch_size=64, N=2048, gae_lambda=0.95, n_actions=rank, agents=agents)

        self.amount_of_metrics = 50
        self.last_losses = np.zeros(self.amount_of_metrics)
        self.last_rewards = np.zeros(self.amount_of_metrics)



    def train_loop(self, env: PeersimEnv, num_episodes, print_instead=True, controllers=None, warm_up_file=None,
                   load_weights=None, results_file=None):
        super().train_loop(env, num_episodes, print_instead, controllers)
        # See page 14 from: https://arxiv.org/pdf/1602.01783v2.pdf

        scores, episodes, avg_scores, obj, avg_episode = [], [], [], [], []
        steps_per_return = self.steps_for_return
        self.mh = mh(agents=env.possible_agents, num_nodes=env.number_nodes, num_episodes=num_episodes,
                     file_name=results_file + "_result")

        if load_weights is not None:
            for idx, agent in enumerate(env.possible_agents):
                agent_w = load_weights + f"_{agent}.pth.tar"
                self.PPOs[agent].load_checkpoint(agent_w, "")  # need to convert this to what I've been using.

        for i in range(num_episodes):
            # Prepare variables for the next run
            dones = [False for _ in controllers]
            agent_list = env.agents
            step = 0
            # Episode metrics
            score = 0.0

            # Reset the state
            states, _ = env.reset()
            states = utils.flatten_state_list(states, agent_list)

            while not utils.is_done(dones):
                print(f'Step: {step}\n')
                # Interaction Step:
                targets = {agent: np.floor(self.get_action(np.array([states[idx]])[0], agent)) for idx, agent in
                           enumerate(agent_list)}
                actions = utils.make_action(targets, agent_list)

                self.mh.register_actions(actions)

                next_states, rewards, dones, _, info = env.step(actions)
                next_states = utils.flatten_state_list(states=next_states, agents=agent_list)
                for idx, agent in enumerate(agent_list):
                    total_reward_in_step = self.remember(states[idx], actions[agent]['neighbourIndex'], self.last_val[agent], self.last_prob[agent], rewards[agent], next_states[idx], dones[agent], agent)
                    score += total_reward_in_step

                # Advance to next iter
                states = next_states
                last_losses = {agent: 0 for agent in agent_list}

                step += 1
                if step % steps_per_return == 0 or self.check_all_done(dones):
                    # Here we will learn the paths from all the agents
                    print("Training...")
                    for agent in agent_list:
                        last_loss = self.learn(s=None, a=None, r=None, s_next=None, k=step, fin=None, agent=agent) # Not used inside of PPO
                        last_losses[agent] = last_loss if not last_loss is None else 0

                print(f'Action{actions}  -   Loss: {last_losses}  -    Rewards: {rewards}')
                self.mh.update_metrics_after_step(rewards=rewards,
                                                  losses=last_losses,
                                                  overloaded_nodes=info[pg.STATE_G_OVERLOADED_NODES],
                                                  average_response_time=info[pg.STATE_G_AVERAGE_COMPLETION_TIMES],
                                                  occupancy=info[pg.STATE_G_OCCUPANCY],
                                                  dropped_tasks=info[pg.STATE_G_DROPPED_TASKS],
                                                  finished_tasks=info[pg.STATE_G_FINISHED_TASKS],
                                                  total_tasks=info[pg.STATE_G_TOTAL_TASKS])

            # Update final metrics
            self.mh.print_action_density_episode()
            self.mh.compile_aggregate_metrics(i, step)
            if i % self.save_interval == 0:
                for agent in env.agents:
                    self.PPOs[agent].save_checkpoint(filename=f"{self.control_type}_value_{i}_{agent}.pth.tar")

            print("Episode {0}/{1}, Score: {2}, AVG Score: {3}".format(i, num_episodes, score,
                                                                       self.mh.episode_average_reward(i)))

        if results_file is not None:
            self.mh.store_as_cvs(results_file)
        self.mh.plot_agent_metrics(num_episodes=num_episodes, title=self.control_type, print_instead=print_instead)
        self.mh.plot_simulation_data(num_episodes=num_episodes, title=self.control_type, print_instead=print_instead)
        self.mh.clean_plt_resources()

    def learn(self, s, a, r, s_next, k, fin, agent):
        loss = self.PPOs[agent].learn()
        return loss

    def get_action(self, observation, agent):
        action = self.PPOs[agent].choose_action(observation)
        self.last_val[agent] = action[1]
        self.last_prob[agent] = action[2]
        return action[0]



    def tally_actions(self, actions):
        for worker, action in actions.items():
            self.mh.register_action(action[pe.ACTION_NEIGHBOUR_IDX_FIELD], worker)

    def check_all_done(self, dones):
        """
        This is possible because all the agents are done at the same time, and only when the episode ends
        :param dones:
        :return:
        """
        return any([d for d in dones.values()])

    def remember(self, state, action, vals, probs, reward, n_state,  done, agent):
        return self.PPOs[agent].remember(state, action, vals, probs, reward, n_state, done, agent)

