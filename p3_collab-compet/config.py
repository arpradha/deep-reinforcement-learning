BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-2 # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay

NUM_ATOMS = 51
Vmin = -0.7
Vmax = 0.7
DELTA_Z = (Vmax - Vmin)/(NUM_ATOMS-1)

UPDATE_EVERY = 6
N_STEPS = 7
N_AGENTS = 2