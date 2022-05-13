'''
Server for federated averaging
'''

from random import sample


def average(tensors):
    '''
    Returns the average of `tensors`.
    params:
        tensors: List of tensors to average. Fails if the dimensions are different.
    '''
    return sum(tensors)/len(tensors)


class Server():
    def __init__(self, clients, participant_portion, model):
        '''
        params:
            clients: List of clients
            participant_portion (float): Portion of the clients _C_ selected at each timestep.
            model: Torch specification of a model
        '''
        self.clients = clients
        self.participant_portion = participant_portion
        self.global_model = model

    def new_participants(self):
        '''
        Gets the next round of participants.

        Warning: Non-deterministic.
        '''
        return sample(self.clients, k=(len(self.clients) * self.participant_portion))

    def measure_latency(client):
        '''
        Measures the latency between a client with a very small packet.
        '''
        print("Unimplemented!")
        assert(False)

    def run(self, bind_address, port):
        '''
        Starts the server, binding to `bind_address` and `port`.
        '''
        print("Unimplemented!")
        assert(False)
