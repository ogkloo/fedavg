import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


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
    end_weights = []
    # Create zero tensors of the required shapes
    for p in get_params_data(ms[0]):
        end_weights.append(torch.zeros(size=p.shape).to(device))
    params_by_model = [get_params_data(m) for m in ms]
    for model in params_by_model:
        for t, end_weight in zip(model, end_weights):
            end_weight += t
    for t in end_weights:
        t /= len(ms)
    return end_weights


def apply_weights(w, m):
    '''
    Apply the weight data stored in `w` to `m`.
    '''
    for p, w_p in zip(m.parameters(), w):
        p.data = w_p.clone()

# IID == Well mixed sample of episodes
# non-IID == Correlated in time
