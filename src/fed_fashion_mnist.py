'''
Usage: python fed_fashion_mnist [CLIENTS] [C] [H]
CLIENTS: The number of simulated client devices (i.e. mobile devices) used in the training environment
    Default: 3
C: Number of clients selected at each training round
    Default: 2
H: Number of minibatches to run at each client
    Default: 1

Major thanks to Ece Işık Polat whose code on Towards Data Science was 
massively helpful to this implementation. Her code can be found here:
https://towardsdatascience.com/federated-learning-a-simple-implementation-of-fedavg-federated-averaging-with-pytorch-90187c9c9577

However, the code was provided in a series of gists. Together, these did not 
provide a working implementation, only inspiration for one. This is a working 
implementation using minimal dependencies (only torch, and optionally requests
to download the dataset). 

This example trains FashionMnist using a trivial fully connected
network. With a single node, this converges in ~10 epochs to ~70% accuracy. With
more nodes, the accuracy should degrade slightly, since they will have both fewer
samples and potentially much more non-iid data. Try it on other datasets!

Because there is no guarantee that the data will be remotely IID, you might try
running the program multiple times with the same hyperparameters to get better
results if you remain unconvinced this technique works.

TODO:
    * Add multiprocessing for simulated devices. Epochs > 1 are currently too 
    slow to train. This does however increase code complexity AND resource 
    requirements.
    * Make this work for more kinds of models.
    * Overlapping datasets to accomodate small datasets.
    * Better comments.
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

import random
from sys import argv


# Swap device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def get_params_data(m):
    '''
    Gets all data values from a model
    '''
    return [p.data.clone() for p in m.parameters()]

def avg_weights(ms):
    '''
    Average the weights of a list of models.

    All models must have exactly the same parameter shapes.
    '''
    # sum_weights([]) == []
    if len(ms) == 0:
        return []
    zeros = []
    # Create zero tensors of the required shapes
    for p in get_params_data(ms[0]):
        zeros.append(torch.zeros(size=p.shape).to(device))
    params_by_model = [get_params_data(m) for m in ms]
    for model in params_by_model:
        for t,zero in zip(model, zeros):
            zero += t
    for t in zeros:
        t /= len(ms)
    return zeros

def apply_weights(w, m):
    '''
    Apply the weight data stored in `w` to `m`.
    '''
    for p,w_p in zip(m.parameters(),w):
        p.data = w_p.clone()

'''
This part of the code shows a simulated test case.
'''

# Load training data
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

# Load test data
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

test_dataloader = DataLoader(test_data, batch_size=64)

# Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

def models_list(num_models, ds):
    '''
    Create models, as well as their associated sections of the dataset, their optimizers (if you
    forget this part, none of them will train), and their dataloaders.
    '''
    models = []
    datasets = torch.utils.data.random_split(ds, [int(len(ds)/num_models) for i in range(num_models)])
    for i in range(num_models):
        model = NeuralNetwork().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        dataset = datasets[i]
        loader = DataLoader(dataset, batch_size=64)
        models += [(model, optimizer, dataset, loader)]
    return models

def train_loop(dataloader, model, loss_fn, optimizer):
    '''
    This is a basic training loop, nothing special here unless you care about how we're shipping
    the data to the GPU. You might care about that if you are e.g. trying to simulate 800 devices
    over 4 GPUs or something.
    '''
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)

def test_loop(dataloader, model, loss_fn):
    '''
    Basic testing loop. Again, nothing special.
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return correct

if len(argv) > 1:
    # Number of simulated client devices
    num_clients = int(argv[1])
else: 
    num_clients = 3

if len(argv) > 2:
    # Number of clients selected at each training round
    C = int(argv[2])
else:
    C = 2

if len(argv) > 3:
    # Number of minibatches to run on each client
    H = int(argv[3])
else:
    H = 1

models = models_list(num_clients, training_data)
test_model = NeuralNetwork()
for model,_,_,_ in models:
    model.to(device)
for round in range(100):
    round_models = random.sample(models, k=C)
    print(f'Round {round+1}', end='...', flush=True)
    accuracies = []
    # Train the models for an epoch
    for model, opt, dataset, loader in round_models:
        epochs = H
        for t in range(epochs):
            train_loop(loader, model, loss_fn, opt)
            accuracies.append(test_loop(test_dataloader, model, loss_fn))
    print(accuracies)

    ws = avg_weights([m for m,_,_,_ in round_models])

    with torch.no_grad():
        for model, _,_,_ in models:
            apply_weights(ws, model)
        apply_weights(ws, test_model)

    acc = test_loop(test_dataloader, test_model, loss_fn)
    print(f'Global model accuracy: {acc*100:.2f}%')
