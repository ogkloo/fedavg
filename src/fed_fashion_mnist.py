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

def avg_weights(models):
    ''' 
    Averages the weights of the model described above. All models in the list must
    be exactly the same.
    '''
    model = models[0]

    layer1_mean_weight = torch.zeros(size=model.linear_relu_stack[0].weight.shape).to(device)
    layer1_mean_bias = torch.zeros(size=model.linear_relu_stack[0].bias.shape).to(device)

    # model.linear_relu_stack[1] is a relu layer, so we skip to 2
    layer2_mean_weight = torch.zeros(size=model.linear_relu_stack[2].weight.shape).to(device)
    layer2_mean_bias = torch.zeros(size=model.linear_relu_stack[2].bias.shape).to(device)

    # Same as above
    layer3_mean_weight = torch.zeros(size=model.linear_relu_stack[4].weight.shape).to(device)
    layer3_mean_bias = torch.zeros(size=model.linear_relu_stack[4].bias.shape).to(device)

    # This is an optimization, with no real logical meaning.
    # It turns off the ability to do backward steps, reducing memory usage.
    with torch.no_grad():
        for model in models:
            layer1_mean_weight += model.linear_relu_stack[0].weight.data.clone()
            layer1_mean_bias += model.linear_relu_stack[0].bias.data.clone()
            layer2_mean_weight += model.linear_relu_stack[2].weight.data.clone()
            layer1_mean_bias += model.linear_relu_stack[2].bias.data.clone()
            layer3_mean_weight += model.linear_relu_stack[4].weight.data.clone()
            layer3_mean_bias += model.linear_relu_stack[4].bias.data.clone()
        layer1_mean_weight = layer1_mean_weight/len(models)
        layer1_mean_bias = layer1_mean_bias/len(models)
        layer2_mean_weight = layer2_mean_weight/len(models)
        layer2_mean_bias = layer2_mean_bias/len(models)
        layer3_mean_weight = layer3_mean_weight/len(models)
        layer3_mean_bias = layer3_mean_bias/len(models)
    return layer1_mean_weight, layer1_mean_bias, layer2_mean_weight, layer2_mean_bias, layer3_mean_weight, layer3_mean_bias

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

def apply_avg_weights(weight1, bias1, weight2, bias2, weigh3, bias3, model):
    '''
    Applies the weight averages.
    '''
    model.linear_relu_stack[0].weight.data = weight1.clone()
    model.linear_relu_stack[0].bias.data = torch.nn.Parameter(bias1.clone())
    model.linear_relu_stack[2].weight.data = weight2.clone()
    model.linear_relu_stack[2].bias.data = torch.nn.Parameter(bias2.clone())
    model.linear_relu_stack[4].weight.data = weight3.clone()
    model.linear_relu_stack[4].bias.data = torch.nn.Parameter(bias3.clone())

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

    weight1, bias1, weight2, bias2, weight3, bias3 = avg_weights([m for m,_,_,_ in round_models])

    with torch.no_grad():
        for model, _,_,_ in models:
            apply_avg_weights(weight1, bias1, weight2, bias2, weight3, bias3, model)
        apply_avg_weights(weight1, bias1, weight2, bias2, weight3, bias3, test_model)

    acc = test_loop(test_dataloader, test_model, loss_fn)
    print(f'Global model accuracy: {acc*100:.2f}%')
