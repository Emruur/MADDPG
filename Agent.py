import torch as T
from CriticNetwork import CriticNetwork
from ActorNetwork import ActorNetwork
import numpy as np

class Agent:
    def __init__(self, actor_dims: int, critic_dims: int, n_actions: int, 
                 n_agents: int, agent_idx: int, chkpt_dir: str,
                 alpha: float = 0.01, beta: float = 0.01, fc1: int = 64, 
                 fc2: int = 64, gamma: float = 0.95, tau: float = 0.01):
        self.gamma: float = gamma  # Discount factor for future rewards
        self.tau: float = tau  # Parameter for soft update of target networks
        self.n_actions: int = n_actions
        self.agent_name: str = 'agent_%s' % agent_idx
        # Actor and critic networks
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir, name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        # Target networks
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation: np.ndarray) -> np.ndarray:

        observation_np = np.array(observation)

        # Add an extra dimension to 'observation_np'
        observation = np.expand_dims(observation_np, axis=0)

        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions + noise

        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau: float = None) -> None:
        '''
            Updates the target networks, we dont just copy the parameters 
            from the network to target for learning stability, taus is the modification ampunt
        '''
        if tau is None:
            tau = self.tau

        # Soft update of the target networks
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self) -> None:
        # Save checkpoints for all networks
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self) -> None:
        # Load checkpoints for all networks
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
