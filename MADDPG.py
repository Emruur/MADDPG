from Agent import Agent
from typing import List, Tuple
import numpy as np
import tensorflow as tf

class MADDPG:
    def __init__(self, actor_dims: List[int], critic_dims: int, n_agents: int, n_actions: int, 
                 scenario: str = 'simple', alpha: float = 0.01, beta: float = 0.01, 
                 fc1: int = 64, fc2: int = 64, gamma: float = 0.99, tau: float = 0.01, 
                 chkpt_dir: str = 'tmp/maddpg/'):
        self.agents:List[Agent] = []  # List to store agents
        self.n_agents:int = n_agents  # Number of agents
        self.n_actions:int = n_actions  # Number of actions
        chkpt_dir += scenario  # Directory for saving checkpoints

        # Initialize each agent
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                                     n_actions, n_agents, agent_idx, alpha=alpha, 
                                     beta=beta, chkpt_dir=chkpt_dir))

    def save_checkpoint(self) -> None:
        # Save model checkpoints for all agents
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self) -> None:
        # Load model checkpoints for all agents
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs: List[np.ndarray]) -> List[np.ndarray]:
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions_perm, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(np.stack(actions_perm), dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.bool)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            # Convert states to TensorFlow tensors
            new_states = tf.convert_to_tensor(actor_new_states[agent_idx], dtype=tf.float32)
            mu_states = tf.convert_to_tensor(actor_states[agent_idx], dtype=tf.float32)

            # Forward pass through the target and regular actor networks
            new_pi = agent.target_actor(new_states)
            pi = agent.actor(mu_states)

            # Append actions for all agents
            all_agents_new_actions.append(new_pi)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        # Concatenate actions for all agents
        new_actions = tf.concat(all_agents_new_actions, axis=1)
        mu = tf.concat(all_agents_new_mu_actions, axis=1)
        old_actions = tf.concat(old_agents_actions, axis=1)

        for agent_idx, agent in enumerate(self.agents):
            with tf.GradientTape(persistent=True) as critic_tape:
                # Forward pass through the target critic and critic networks
                critic_value_ = agent.target_critic(states_, new_actions)
                critic_value_ = tf.reshape(critic_value_, [-1])
                critic_value_ = tf.where(dones[:, 0], tf.zeros_like(critic_value_), critic_value_)
                
                critic_value = agent.critic(states, old_actions)
                critic_value = tf.reshape(critic_value, [-1])

                # Calculate the target and critic loss
                target = rewards[:, agent_idx] + agent.gamma * critic_value_
                critic_loss = tf.keras.losses.MSE(target, critic_value)


            # Compute gradients and update critic weights
            critic_grad = critic_tape.gradient(critic_loss, agent.critic.trainable_variables)
            agent.critic.optimizer.apply_gradients(zip(critic_grad, agent.critic.trainable_variables))

            with tf.GradientTape(persistent=True) as actor_tape:
                # Forward pass through the critic using mu for actor loss
                mu_states= tf.convert_to_tensor(actor_states[agent_idx], dtype=tf.float32)
                all_agents_new_mu_actions[agent_idx]=  agent.actor(mu_states)
                mu = tf.concat(all_agents_new_mu_actions, axis=1)
                actor_loss = agent.critic(states, mu)
                actor_loss = tf.reshape(actor_loss, [-1])
                actor_loss = -tf.reduce_mean(actor_loss)

            # Compute gradients and update actor weights
            #FIXME actor_grad is none
            actor_grad = actor_tape.gradient(actor_loss, agent.actor.trainable_variables)
            agent.actor.optimizer.apply_gradients(zip(actor_grad, agent.actor.trainable_variables))

            # Update target networks
            agent.update_network_parameters()