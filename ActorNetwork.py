import tensorflow as tf
import os

class ActorNetwork(tf.keras.Model):
    '''
        Policy network of agents, this does not have access to global state
        Agents are limited to their local observations when deciding an action
        Maps observations to an action, this is a deterministic policy but we will probably
        do a epsilon greedy action selection to have some exploration, we have softmax 
        to have a proper prob distrbution over actions, we will choose the highest probability
        action and act discretely
    '''
    def __init__(self, alpha: float, input_dims: int, fc1_dims: int, fc2_dims: int, 
                 n_actions: int, name: str, chkpt_dir: str) -> None:
        super(ActorNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name + '.h5')
        self.in_dims= input_dims
        # Define the fully connected layers
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu', 
                                         input_shape=(input_dims,))
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.pi = tf.keras.layers.Dense(n_actions, activation=None)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

    def call(self, state: tf.Tensor) -> tf.Tensor:
        x = self.fc1(state)
        x = self.fc2(x)
        pi = tf.nn.softmax(self.pi(x), axis=1)
        return pi

    def save_checkpoint(self) -> None:
        # Save the network state
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        self.save_weights(self.chkpt_file)

    def load_checkpoint(self) -> None:
        # Load the network state
        self.load_weights(self.chkpt_file)


