from collections import deque
import numpy as np
import random
import tensorflow as tf


class Model():
    '''
    Deep Q-learning model (DQN) that uses a neural network to approximate the Q-function.
    This is the same as the single_feature_model.py, but this model takes in multiple features.
    The number of features is defined by the num_features parameter.
    '''
    def __init__(self, state_size, num_features, action_space=3, model_name="model"):
        self.num_features = num_features
        self.state_size = state_size
        self.action_space = action_space # Action space of 3 - Buy, Sell and Hold
        self.model_name = model_name
        self.model = None

        # Replay memory is a queue of the last 2000 transactions
        # This is used to train the model
        self.memory = deque(maxlen=2000) 
        self.inventory = []

        self.gamma = 0.95 # Maximize reward over long-term
        self.epsilon = 1.0 # 1.0 means model will explore the environment randomly during training
        self.epsilon_final = 0.01 # Decrease epsilon over time to take less random actions
        self.epsilon_decay = 0.995 # Decay rate of epsilon

    def model_builder(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation="relu", input_shape=(None, self.state_size, self.num_features)))
        model.add(tf.keras.layers.Dense(units=64, activation="relu"))
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_space, activation="linear")) # We use linear activation cuz it's a regression task w/ no set range
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        self.model = model
        return model
    
    def trade(self, state):
        '''
        The model generates a random floating point number between 0 and 1. 
        If the number is less than epsilon, the model will explore the environment randomly.
        If the number is greater than epsilon, the model will exploit the environment by choosing the best action.
        '''
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        state = state.reshape(-1, 1, self.state_size, self.num_features) # Reshape the state to have shape (batch_size, state_size, num_features)
        # state = np.expand_dims(state, axis=0) # Add a dimension to the state to have shape (batch_size, 1, state_size, num_features)
        actions = self.model.predict(state, verbose=0)
        return np.argmax(actions, axis=-1)[0][0][0] # For some reason, the model outputs a 3D array, so we need to squeeze it to get the action
    
    def batch_train(self, batch_size):
        batch = random.sample(self.memory, batch_size) # (state, action, reward, next_state, done)
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        targets = self.model.predict(states, verbose=0)
        targets_next = self.model.predict(next_states, verbose=0)

        # print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape, targets.shape, targets_next.shape)
        # (32, 1, 10, 6) (32,) (32,) (32, 1, 10, 6) (32,) (32, 1, 10, 3) (32, 1, 10, 3) -> Shapes of the arrays for multi_feature model
        # (32, 1, 10)    (32,) (32,) (32, 1, 10)    (32,) (32, 3)        (32, 1, 3) -> Shapes of the arrays for single_feature model

        for i in range(batch_size):
            if dones[i]:
                # Add reward to the last dimension
                targets[i, :, :, actions[i]] = rewards[i]
            else:
                targets[i, :, :, actions[i]] = rewards[i] + self.gamma * np.amax(targets_next[i], axis=-1)
        
        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay