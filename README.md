# fedavg
Pytorch fedavg implementation.

## FedAvg

FederatedAveraging, or **FedAvg**, is a distributed machine learning algorithm
built for 

## Bandit-solving for participant selection

We can characterize client selection for latency given what the server knows as
a multi-armed bandit problem over highly non-stationary distributions.

There is some information contained in the latency of any given device

### Previous work

### Clients, participants, and selection

### Bandit solvers

### Why should this work?

### Challenges and future work

# Nix
This project uses nix to manage system dependencies. After installing nix, 
you can activate a shell by using:

```
$ nix-shell --pure
[nix-shell] $ virtualenv torch-env  
[nix-shell] $ source torch-env/bin/activate
[nix-shell] $ pip install requirements.txt
```

After that, you can use the project as intended.
