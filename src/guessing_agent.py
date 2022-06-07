import torch as t
import torch.nn as nn
import guessing_game as game

# Swap device
device = 'cuda' if t.cuda.is_available() else 'cpu'
print(f'Using {device} device')

GAME_SIZE = 5
GAME_LENGTH = 1
GAME_MIN = -10
GAME_MAX = 10
MEMORY = 5

gamma = 0.99

def win_condition(x):
    return x

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


# Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer decision
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model = DQN(GAME_SIZE, MEMORY).to(device)

def do_episode(state_history):
    # Occasionally do random action
    # Do normal action
    if len(state_history) > MEMORY:
        state_history = state_history[1:] + [env.current_state()]
    else:
       state_history.append(env.current_state())
    state = t.FloatTensor(state_history)
    state = state.to(device)
    predicted_answer = int(t.argmax(model(state)))
    status, next_state = env.guess(predicted_answer)
    if status:
        return 1
    else:
        return 0
    
env = game.Game(GAME_SIZE, GAME_LENGTH, GAME_MIN, GAME_MAX, win_condition)
reward = 0
steps_per_episode = 1000
episodes = 100000
history = []
for episode in range(0, episodes):
    # New instance
    env = game.Game(GAME_SIZE, GAME_LENGTH, GAME_MIN, GAME_MAX, win_condition)
    print(env.current_state())
    episodic_reward = 0
    for timestep in range(0,steps_per_episode):
        episodic_reward = (gamma**timestep)*do_episode(history)
        if episodic_reward > 0:
            reward += episodic_reward
            print(f'Episode #{episode}: {timestep}')
            break
print(reward/episodes)
