import numpy as np
from peersim_gym.envs.PeersimEnv import PeersimEnv
from torchsummary import summary

from src.Agents.Agent import Agent
from src.Agents.Networks.A2C import ActorCritic
from src.Utils import utils


class A2CAgent(Agent):
    """
    Actor-Critic Agent
    Sources:
    - https://www.youtube.com/watch?v=OcIx_TBu90Q&t=1050s  | Video with tips of how to implement A3C
    """

    def __init__(self, input_shape, action_space, output_shape, agents, learning_rate=0.7, gamma=0.4, steps_for_return=150,
                 control_type="A2C"):
        super().__init__(input_shape, action_space, output_shape, learning_rate)
        self.gamma = gamma
        self.control_type = control_type

        self.A2C = ActorCritic(lr=learning_rate, input_dims=self.input_shape, fc1_dims=256, fc2_dims=256,
                               n_actions=self.actions)
        summary(self.A2C, input_size=self.input_shape)
        self.agent_states = {
            agent: {
                'state': [],
                'action': [],
                'reward': [],
                'next_state': [],
                'done': []
            } for agent in agents
        }

    def train_loop(self, env: PeersimEnv, num_episodes, print_instead=False, controllers=None):
        super().train_loop(env, num_episodes, print_instead, controllers)
        # See page 14 from: https://arxiv.org/pdf/1602.01783v2.pdf
        scores, episodes, avg_scores, obj, avg_episode = [], [], [], [], []
        steps_per_return = 5
        for i in range(num_episodes):
            # Prepare variables for the next run
            dones = [False for _ in controllers]

            total_reward = 0
            step = 0
            total_steps = 0
            # Episode metrics
            score = 0.0

            # Reset the state
            states, _ = env.reset()
            agent_list = env.agents
            states = utils.flatten_state_list(states, agent_list)

            while not utils.is_done(dones):
                print(f'Step: {step}\n')
                # Interaction Step:
                targets = {agent: np.floor(self.get_action(np.array([states[idx]]))) for idx, agent in
                           enumerate(agent_list)}
                actions = utils.make_action(targets, agent_list)

                next_states, rewards, dones, _, _ = env.step(actions)
                next_states = utils.flatten_state_list(states=next_states, agents=agent_list)
                total_reward_in_step = self.__store_agent_step_data(states, actions, rewards, next_states, dones,
                                                                    agent_list)
                score += total_reward_in_step
                # Advance to next iter
                states = next_states
                step += 1
                total_steps += 1
                if step % steps_per_return == 0 or dones:
                    # Here we will learn the paths from all the agents
                    for agent in agent_list:
                        s, a, r, s_next, fin = self.__get_agent_step_data(agent)
                        if s and a and r and s_next and fin: # Check if fin is always not empty as well
                            self.learn(s=s, a=a, r=r, s_next=s_next, k=step, fin=fin)
                    step = 0
            # Update final metrics
            avg_episode.append(score / total_steps)
            scores.append(score)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)

        self.plot(episodes, scores=scores, avg_scores=avg_scores, per_episode=avg_episode,
                    print_instead=print_instead)
        self.plot2(episodes, title=self.control_type, per_episode=avg_episode, print_instead=print_instead)

    def learn(self, s, a, r, s_next, k, fin):
        self.A2C.remember_batch(states=s, actions=a, rewards=r, dones=fin)  # States should be ordered.
        self.A2C.optimizer.zero_grad()
        loss = self.A2C.calculate_loss(fin)
        loss.backward()
        self.A2C.clear_memory()

    def get_action(self, observation):
        return self.A2C.choose_action(observation)

    def __store_agent_step_data(self, states, actions, rewards, next_states, dones, agent_list):
        total_rwrd = 0
        for idx, agent in enumerate(agent_list):
            # Update history
            agent_data = self.agent_states[agent]
            agent_data['state'].add(states[idx])
            agent_data['action'].add(actions[agent])
            agent_data['reward'].add(rewards[agent])
            agent_data['next_state'].add(next_states[idx])
            agent_data['done'].add(dones[idx])
            total_rwrd += rewards[agent]
        return total_rwrd

    def __get_agent_step_data(self, agent):
        agent_data = self.agent_states[agent]
        return agent_data['state'], agent_data['action'], agent_data['reward'], agent_data['next_state'], agent_data[
            'done']

    def __clean_agent_step_data(self, agents):
        for agent in agents:
            self.agent_states[agent] = {
                'state': [],
                'action': [],
                'reward': [],
                'next_state': [],
                'done': []
            }
