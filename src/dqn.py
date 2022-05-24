'''
Original can be found here:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Most other tutorials tend to use tensorflow or come in the form of terrible
medium articles that would've been better as blog posts. I am not very familiar
with tensorflow so I've opted to use Torch.

This is a script that straightforwardly trains a network on the fake "game".
'''

import torch as t
import torch.nn as nn
from collections import namedtuple, deque
import random

import guessing_game as game

# Set device
device = 'cuda' if t.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Game parameters
GAME_SIZE: int = 5
GAME_LENGTH: int = 1
GAME_MIN: int = -10
GAME_MAX: int = 10
MEMORY: int = 5


def win_condition(x: int):
    return x


Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Dead simple DQN


class DQN(nn.Module):
    def __init__(self, game_size, memory):
        # Episodes in memory
        self.memory = memory
        self.game_size = game_size
        super(DQN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            # game_size*self.memory: Episodes currently in memory, plus current episode
            # +self.memory: Previous guesses at each episode
            nn.Linear(self.game_size*(self.memory+1), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.game_size),
        )

    def forward(self, x):
        x = t.flatten(x)
        padding = t.zeros(self.game_size*(self.memory+1) - len(x)).to(device)
        x = t.cat((x, padding))
        # print(x.shape)
        logits = self.linear_relu_stack(x)
        return logits


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


env = game.Game(GAME_SIZE, GAME_LENGTH, GAME_MIN, GAME_MAX, win_condition)
# Create the policy under evaluation and the target policy
policy_net = DQN(GAME_SIZE, MEMORY)
target_net = DQN(GAME_SIZE, MEMORY)
# Make them equal to start
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = t.optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    # This is a legitimately bad part of Python but I don't know how to fix it. This is a
    # script and not a library so I'm okay doing this I think.
    global steps_done
    sample = random.random()
    epsilon_threshold = EPS_END + \
        (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > epsilon_threshold:
        # Don't need to compute gradients here, so turn them off to save memory
        with torch.no_grad():
            # state = t.FloatTensor()
            # Original often forgets this part, please do not forget this part if you choose to
            # evaluate this.
            state = state.to(device)
            # Select what the policy net thinks is a good idea
            # Original code does this in a much more confusing way, so I cleaned it up by using
            # argmax. Has exactly equivalent output. Original for comparison:
            # policy_net(state).max(1)[1].view(1, 1)
            # tensor.max is a fairly screwy interface and I don't like its argument format.
            return t.argmax(policy_net(state)).view(1, 1)
    else:
        # Random action in shape t.Size([1,1])
        # Helps fit in with other adapted code
        return t.tensor([[random.randrange(GAME_SIZE)]], device=device, dtype=t.long)


episode_durations = []


def optimize_model():
    # If memory can't hold a batch, quit
    if len(memory) < BATCH_SIZE:
        return

    # sample a batch
    transitions = memory.sample(BATCH_SIZE)
    # * operator: "all arguments?"
    # Example gives:
    # https://stackoverflow.com/a/19343/3343043
    batch = Transition(*zip(*transitions))
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                       batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state
    #                                             if s is not None])
