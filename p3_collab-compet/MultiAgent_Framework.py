import numpy as np
import config

from prioritized_memory import Memory
from d4pg_agent import Agent
from collections import deque
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ExperienceQueue:
    def __init__(self, queue_length=100):
        self.states = deque(maxlen=queue_length)
        self.actions = deque(maxlen=queue_length)
        self.rewards = deque(maxlen=queue_length)

class MultiAgent:

    def __init__(self, state_size, action_size, seed, num_agents=2):
        self.BATCH_SIZE = config.BATCH_SIZE
        self.GAMMA = config.GAMMA
        self.TAU = config.TAU

        self.UPDATE_EVERY = config.UPDATE_EVERY
        self.num_mc_steps = config.N_STEPS
        self.experiences = [ExperienceQueue(config.N_STEPS) for _ in range(config.N_AGENTS)]
        self.memory = Memory(config.BUFFER_SIZE)
        self.Agents = [Agent(state_size, action_size, seed) for _ in range(num_agents)]
        self.batch_size = config.BATCH_SIZE
        self.t_step = 0
        self.train_start = config.BATCH_SIZE

        self.rewards_queue=[deque(maxlen=config.N_STEPS),deque(maxlen=config.N_STEPS)]
        self.states_queue=[deque(maxlen=config.N_STEPS),deque(maxlen=config.N_STEPS)]

    def acts(self, states, add_noise=0.0):
        acts = []
        for s, a in zip(states, self.Agents):
            acts.append(a.act(np.expand_dims(s, 0), add_noise))
        return np.vstack(acts)


    # borrow from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter14
    def distr_projection(self, next_distr_v, rewards_v, dones_mask_t, gamma):
        next_distr = next_distr_v.data.cpu().numpy()
        rewards = rewards_v.data.cpu().numpy()
        dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, config.NUM_ATOMS), dtype=np.float32)
        dones_mask = np.squeeze(dones_mask)
        rewards = rewards.reshape(-1)

        for atom in range(config.NUM_ATOMS):
            tz_j = np.minimum(config.Vmax, np.maximum(config.Vmin, rewards + (config.Vmin + atom * config.DELTA_Z) * gamma))
            b_j = (tz_j - config.Vmin) / config.DELTA_Z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l

            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l

            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

        if dones_mask.any():
            proj_distr[dones_mask] = 0.0
            tz_j = np.minimum(config.Vmax, np.maximum(config.Vmin, rewards[dones_mask]))
            b_j = (tz_j - config.Vmin) / config.DELTA_Z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            if dones_mask.shape == ():
                if dones_mask:
                    proj_distr[0, l] = 1.0
                else:
                    ne_mask = u != l
                    proj_distr[0, l] = (u - b_j)[ne_mask]
                    proj_distr[0, u] = (b_j - l)[ne_mask]
            else:
                eq_dones = dones_mask.copy()

                eq_dones[dones_mask] = eq_mask
                if eq_dones.any():
                    proj_distr[eq_dones, l[eq_mask]] = 1.0
                ne_mask = u != l
                ne_dones = dones_mask.copy()
                ne_dones[dones_mask] = ne_mask
                if ne_dones.any():
                    proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                    proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

        return torch.FloatTensor(proj_distr).to(device)

    def step(self, states, actions, rewards, next_states, dones):

        for agent_index in range(len(self.Agents)):
            self.states_queue[agent_index].appendleft([states[agent_index],actions[agent_index]])
            self.rewards_queue[agent_index].appendleft(rewards[agent_index]*self.GAMMA**config.N_STEPS)
            for i in range(len(self.rewards_queue[agent_index])):
                self.rewards_queue[agent_index][i] = self.rewards_queue[agent_index][i]/self.GAMMA
            if len(self.rewards_queue[agent_index])>=config.N_STEPS:# N-steps return: r= r1+gamma*r2+..+gamma^(t-1)*rt
                temps=self.states_queue[agent_index].pop()
                state = torch.tensor(temps[0]).float().unsqueeze(0).to(device)
                next_state = torch.tensor(next_states[agent_index]).float().unsqueeze(0).to(device)
                action = torch.tensor(temps[1]).float().unsqueeze(0).to(device)
                self.Agents[agent_index].critic_local.eval()
                with torch.no_grad():
                    Q_expected = self.Agents[agent_index].critic_local(state, action)
                self.Agents[agent_index].critic_local.train()
                self.Agents[agent_index].actor_target.eval()
                with torch.no_grad():
                    action_next = self.Agents[agent_index].actor_target(next_state)
                self.Agents[agent_index].actor_target.train()
                self.Agents[agent_index].critic_target.eval()
                with torch.no_grad():
                    Q_target_next = self.Agents[agent_index].critic_target(next_state, action_next)
                    Q_target_next =F.softmax(Q_target_next, dim=1)
                self.Agents[agent_index].critic_target.train()
                sum_reward=torch.tensor(sum(self.rewards_queue[agent_index])).float().unsqueeze(0).to(device)
                done_temp=torch.tensor(dones[agent_index]).float().to(device)
                Q_target_next=self.distr_projection(Q_target_next,sum_reward,done_temp,self.GAMMA**config.N_STEPS)
                Q_target_next = -F.log_softmax(Q_expected, dim=1) * Q_target_next
                error  = Q_target_next.sum(dim=1).mean().cpu().data
                self.memory.add(error, (states[agent_index], actions[agent_index], sum(self.rewards_queue[agent_index]), next_states[agent_index], dones[agent_index]))
                self.rewards_queue[agent_index].pop()
                if dones[agent_index]:
                    self.states_queue[agent_index].clear()
                    self.rewards_queue[agent_index].clear()

        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            #print(self.memory.tree.n_entries)
            if self.memory.tree.n_entries > self.train_start:
                for agent_index in range(len(self.Agents)):
                    # prioritized experienc replay
                    batch_not_ok=True
                    while batch_not_ok:
                        mini_batch, idxs, is_weights = self.memory.sample(self.BATCH_SIZE)
                        mini_batch = np.array(mini_batch).transpose()
                        if mini_batch.shape==(5,self.BATCH_SIZE):
                            batch_not_ok=False
                        else:
                            print(mini_batch.shape)
                    try:
                        statess = np.vstack([m for m in mini_batch[0] if m is not None])
                    except:
                        print('states not same dim')
                        pass
                    try:
                        actionss = np.vstack([m for m in mini_batch[1] if m is not None])
                    except:
                        print('actions not same dim')
                        pass
                    try:
                        rewardss = np.vstack([m for m in mini_batch[2] if m is not None])
                    except:
                        print('rewars not same dim')
                        pass
                    try:
                        next_statess = np.vstack([m for m in mini_batch[3] if m is not None])
                    except:
                        print('next states not same dim')
                        pass
                    try:
                        doness = np.vstack([m for m in mini_batch[4] if m is not None])
                    except:
                        print('dones not same dim')
                        pass
                    # bool to binary
                    doness = doness.astype(int)
                    statess = torch.from_numpy(statess).float().to(device)
                    actionss = torch.from_numpy(actionss).float().to(device)
                    rewardss = torch.from_numpy(rewardss).float().to(device)
                    next_statess = torch.from_numpy(next_statess).float().to(device)
                    doness = torch.from_numpy(doness).float().to(device)
                    experiences=(statess, actionss, rewardss, next_statess, doness)
                    self.learn(self.Agents[agent_index],experiences,idxs)


    def learn(self, agent, experiences, idxs):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # Compute critic loss
        q_logits_expected = agent.critic_local(states, actions)
        actions_next = agent.actor_target(next_states)
        q_targets_logits_next = agent.critic_target(next_states, actions_next)
        q_targets_distr_next = F.softmax(q_targets_logits_next, dim=1)
        q_targets_distr_projected_next = self.distr_projection(q_targets_distr_next, rewards, dones,
                                                               self.GAMMA ** self.num_mc_steps)
        cross_entropy = -F.log_softmax(q_logits_expected, dim=1) * q_targets_distr_projected_next
        critic_loss = cross_entropy.sum(dim=1).mean()
        with torch.no_grad():
            errors = cross_entropy.sum(dim=1).cpu().data.numpy()
        # update priority
        for i in range(self.BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # Compute actor loss
        actions_pred = agent.actor_local(states)
        crt_distr_v = agent.critic_local(states, actions_pred)
        actor_loss = -agent.critic_local.distr_to_q(crt_distr_v)
        actor_loss = actor_loss.mean()
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic_local, agent.critic_target, self.TAU)
        agent.soft_update(agent.actor_local, agent.actor_target, self.TAU)

    def sample(self):
        # prioritized experience replay
        mini_batch, idxs, is_weights = self.memory.sample(self.BATCH_SIZE)
        mini_batch = np.array(mini_batch).transpose()
        statess = np.vstack([m for m in mini_batch[0] if m is not None])
        actionss = np.vstack([m for m in mini_batch[1] if m is not None])
        rewardss = np.vstack([m for m in mini_batch[2] if m is not None])
        next_statess = np.vstack([m for m in mini_batch[3] if m is not None])
        doness = np.vstack([m for m in mini_batch[4] if m is not None])
        # bool to binary
        doness = doness.astype(int)
        statess = torch.from_numpy(statess).float().to(device)
        actionss = torch.from_numpy(actionss).float().to(device)
        rewardss = torch.from_numpy(rewardss).float().to(device)
        next_statess = torch.from_numpy(next_statess).float().to(device)
        doness = torch.from_numpy(doness).float().to(device)
        return (statess, actionss, rewardss, next_statess, doness), idxs