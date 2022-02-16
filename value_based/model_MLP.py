import torch
import torch.nn as nn
import torch.nn.functional as F

## Reading for Tf primed coder:
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html  // pytorch module expl. 
# https://discuss.pytorch.org/t/input-size-of-fc-layer-in-tutorial/14644 // Input size of fc layer in tutorial
# https://discuss.pytorch.org/t/what-does-the-fc-in-feature-mean/4889  // What does the .fc.in_feature mean?
# https://www.askpython.com/python-modules/initialize-model-weights-pytorch  // initialize weights

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()  # inherits from the nn.Module class 
                                            # https://realpython.com/python-super/#an-overview-of-pythons-super-function
        self.seed = torch.manual_seed(seed)        
        
        "*** my CODE HERE ***"
        #self.conv1 = nn.Conv2d(1, 20, 5)
        #self.conv2 = nn.Conv2d(20, 20, 5)
        
        # let's start with a shallow and small MLP...
        self.fc1   = nn.Linear(state_size, 64)  # (in/out) features
        ###  nn.init.uniform_(self.fc1.weights, -1/state_size, 1/state_size) # init weights 
        
        self.fc2   = nn.Linear(64, 32)  # (in/out) features   
        
        self.fc2b   = nn.Linear(32, 10)  # (in/out) features   
        
        self.fc3   = nn.Linear(10, action_size) 
        
        
    def forward(self, state):   # Defines the computation performed at every call.
        """Build a network that maps state -> action values."""
        #x = F.relu(self.conv1(state))
        x = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x))
        x2b = F.relu(self.fc2b(x2))  ## added this 
        x3 = self.fc3(x2b)
        return x3
        
