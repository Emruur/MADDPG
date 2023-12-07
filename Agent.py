import os
from CriticNetwork import CriticNetwork
from ActorNetwork import ActorNetwork
import numpy as np
import tensorflow as tf

class Agent:
    def __init__(self, actor_dims: int, critic_dims: int, n_actions: int, 
                 n_agents: int, agent_idx: int, chkpt_dir: str,
                 alpha: float = 0.01, beta: float = 0.01, fc1: int = 64, 
                 fc2: int = 64, gamma: float = 0.95, tau: float = 0.01):
        self.gamma: float = gamma  # Discount factor for future rewards
        self.tau: float = tau  # Parameter for soft update of target networks
        self.n_actions: int = n_actions
        self.agent_name: str = 'agent_%s' % agent_idx

        self.ad= actor_dims
        self.cd= critic_dims
        self.na= n_actions
        self.nag= n_agents

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

        # Convert to TensorFlow tensor and process through the actor network
        state = tf.convert_to_tensor(observation, dtype=tf.float32)
        actions = self.actor(state)

        # Generate noise and add it to the actions
        noise = tf.random.uniform(shape=actions.shape)
        action = actions + noise

        # Convert the action tensor to a numpy array
        return action.numpy()[0]


    def update_network_parameters(self, tau: float = None) -> None:
        if tau is None:
            tau = self.tau

        # Update actor network parameters
        for target_param, param in zip(self.target_actor.variables, self.actor.variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

        # Update critic network parameters
        for target_param, param in zip(self.target_critic.variables, self.critic.variables):
            target_param.assign(tau * param + (1 - tau) * target_param)


    def build_models(self):
        dummy_state_actor = tf.random.normal(shape=(1, self.ad))
        dummy_action = tf.random.normal(shape=(1, self.na))
        dummy_state_critic = tf.random.normal(shape=(1, self.cd))
        dummy_combined_input_critic = tf.concat([dummy_state_critic, tf.tile(dummy_action, [1, self.nag])], axis=1)

        # Make dummy calls to build the models
        self.actor(dummy_state_actor)
        self.target_actor(dummy_state_actor)
        self.critic(dummy_combined_input_critic, dummy_action)
        self.target_critic(dummy_combined_input_critic, dummy_action)
    def save_models(self) -> None:
        return
        # Save checkpoints for all networks
        
        self.build_models()
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self) -> None:
        return
        # Load checkpoints for all networks
        # Create dummy inputs for actor and critic networks
        self.build_models()

        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()