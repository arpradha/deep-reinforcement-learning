from unityagents import UnityEnvironment
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent

# select this option to load version 1(with a single agent) of the environment
fname = r"C:\Users\anu\ProjectData\Reacher_Windows_x86_64\Reacher.exe"
env = UnityEnvironment(file_name=fname)

if __name__ == '__main__':
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)

    agent = Agent(state_size=3, action_size=1, random_seed=2)
    n_episodes = 1000
    max_t = 300
    scores_deque = deque(maxlen=100)
    print_every = 100
    scores = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        print(state)
        agent.reset()
        score = 0
        for t in range(max_t):
            action = np.array([agent.act(state)])
            # print(action)
            # next_state, reward, done, _ = env.step(action)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            # print(next_state, reward, done)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    env.close()
