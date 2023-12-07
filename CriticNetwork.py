import os
import tensorflow as tf

class CriticNetwork(tf.keras.Model):
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
        self.chkpt_file = os.path.join(chkpt_dir, name + '.h5')
        self.in_dims= input_dims
        self.num_actions= n_actions
        self.num_agents= n_agents
        # Define the fully connected layers
        print(f"Critic network for {name} is initialized with input dimension {input_dims + n_agents * n_actions} | InDims={input_dims} | num_acts={n_actions} | num_agents= {n_agents}| fc1,fc2={fc1_dims,fc2_dims}")
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu', 
                                         input_shape=(input_dims + n_agents * n_actions,))
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.q = tf.keras.layers.Dense(1, activation=None)

        # Optimizer
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=beta)

    def call(self, state: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        x = tf.concat([state, action], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.q(x)
        return q

    def save_checkpoint(self):
        # Save the network state
        self.save_weights(self.chkpt_file)

    def load_checkpoint(self):
        # Load the network state
        self.load_weights(self.chkpt_file)



