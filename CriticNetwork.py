import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class CriticNetwork(nn.Module):
    '''
        Critic network is a Q function learned *per* agent, if this was a solely cooperative
        enviroment we probably would have been fine with a single critic network for all agents but per 
        agent learning enables mixed cooperative and competitive behaviour since adverserial 
        agent will have a different objective(oppsite of the predetors), 
        we expect the cooperative agents to have a similar critic network since they have a mutual goal.
        This network has access to centralized information which enables  multi agent learning
        Maps states and a set of agent actions to a reward value Q(s,a1,a2,a3...)= value
    '''
    def __init__(self, beta: float, input_dims: int, fc1_dims: int, fc2_dims: int, 
                 n_agents: int, n_actions: int, name: str, chkpt_dir: str):
        super(CriticNetwork, self).__init__()
        T.autograd.set_detect_anomaly(True)
        self.chkpt_file: str = os.path.join(chkpt_dir, name)  # Path for the checkpoint file

        # Define the fully connected layers
        # Q(s,a1,a2,a3...an), s= input dims
        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)  # Output layer for Q-value

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # Device configuration
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)  # Move the network to the configured device

    def forward(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
        # Forward pass through the network
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self) -> None:
        # Save the network state
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self) -> None:
        # Load the network state
        self.load_state_dict(T.load(self.chkpt_file))
