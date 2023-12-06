import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ActorNetwork(nn.Module):
    '''
        Policy network of agents, this does not have access to global state
        Agents are limited to their local observations when deciding an action
        Maps observations to an action, this is a deterministic policy but we will probably
        do a epsilon greedy action selection to have some exploration, we have softmax 
        to have a proper prob distrbution over actions, we will choose the highest probability
        action and act discretely
    '''
    def __init__(self, alpha: float, input_dims: int, fc1_dims: int, fc2_dims: int, 
                 n_actions: int, name: str, chkpt_dir: str):
        super(ActorNetwork, self).__init__()

        self.chkpt_file: str = os.path.join(chkpt_dir, name)  # Path for the checkpoint file

        # Define the fully connected layers
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)  # Output layer for actions

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # Device configuration
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)  # Move the network to the configured device

    def forward(self, state: T.Tensor) -> T.Tensor:
        # Forward pass through the network
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)  # Action probabilities

        return pi

    def save_checkpoint(self) -> None:
        # Save the network state
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self) -> None:
        # Load the network state
        self.load_state_dict(T.load(self.chkpt_file))
