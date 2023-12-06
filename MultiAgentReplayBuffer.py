import numpy as np
from typing import List, Tuple

class MultiAgentReplayBuffer:
    def __init__(self, max_size: int, critic_dims: int, actor_dims: List[int], 
                 n_actions: int, n_agents: int, batch_size: int):
        self.mem_size: int = max_size  # Maximum size of the memory buffer
        self.mem_cntr: int = 0  # Memory counter to keep track of the number of stored transitions
        self.n_agents: int = n_agents  # Number of agents in the environment
        self.actor_dims: List[int] = actor_dims  # Dimensions of the actor's observation space for each agent
        self.batch_size: int = batch_size  # Batch size for sampling from the buffer
        self.n_actions: int = n_actions  # Number of possible actions

        # Memory buffers for storing states, next states, rewards, and terminal flags
        self.state_memory: np.ndarray = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory: np.ndarray = np.zeros((self.mem_size, critic_dims))
        self.reward_memory: np.ndarray = np.zeros((self.mem_size, n_agents))
        self.terminal_memory: np.ndarray = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self) -> None:
        # Initialize memory buffers for each actor (agent)
        self.actor_state_memory: List[np.ndarray] = []
        self.actor_new_state_memory: List[np.ndarray] = []
        self.actor_action_memory: List[np.ndarray] = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs: List[np.ndarray], state: np.ndarray, action: List[np.ndarray], 
                         reward: np.ndarray, raw_obs_: List[np.ndarray], 
                         state_: np.ndarray, done: np.ndarray) -> None:
        # Store a new transition in the buffer
        index: int = self.mem_cntr % self.mem_size  # Circular buffer index

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
        # Sample a batch of transitions
        max_mem: int = min(self.mem_cntr, self.mem_size)
        batch: np.ndarray = np.random.choice(max_mem, self.batch_size, replace=False)

        states: np.ndarray = self.state_memory[batch]
        rewards: np.ndarray = self.reward_memory[batch]
        states_: np.ndarray = self.new_state_memory[batch]
        terminal: np.ndarray = self.terminal_memory[batch]

        actor_states: List[np.ndarray] = []
        actor_new_states: List[np.ndarray] = []
        actions: List[np.ndarray] = []

        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self) -> bool:
        # Check if the buffer is ready for sampling
        return self.mem_cntr >= self.batch_size
