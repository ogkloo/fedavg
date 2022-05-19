'''
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
'''

import torch
from torch import nn
import random
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


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

# Averages the weights of some models as described above
def avg_weights(models):
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

# build a bunch of models
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

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

# Create models, their optimizers, and their datasets
def models_list(num_models, ds):
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
        model.linear_relu_stack[0].weight.data = weight1.clone()
        model.linear_relu_stack[0].bias.data = torch.nn.Parameter(bias1.clone())
        model.linear_relu_stack[2].weight.data = weight2.clone()
        model.linear_relu_stack[2].bias.data = torch.nn.Parameter(bias2.clone())
        model.linear_relu_stack[4].weight.data = weight3.clone()
        model.linear_relu_stack[4].bias.data = torch.nn.Parameter(bias3.clone())

models = models_list(10, training_data)
test_model = NeuralNetwork()
for round in range(100):
    for model,_,_,_ in models:
        model.to(device)
    round_models = random.sample(models, k=5)
    print(f'Round {round+1}', end='...', flush=True)
    accuracies = []
    # Train the models for an epoch
    for model, opt, dataset, loader in round_models:
        epochs = 1
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
    print(f'Global model accuracy: {acc*100}')
