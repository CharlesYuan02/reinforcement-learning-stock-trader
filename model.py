from collections import deque
import numpy as np
import random
import tensorflow as tf


class Model():
    def __init__(self, state_size, action_space=3, model_name="model"):
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
        model.add(tf.keras.layers.Dense(units=32, activation="relu", input_shape=(None, self.state_size)))
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
        state = state.reshape(1, 1, -1) # Reshape the state to have shape (1, 1, state_size)
        actions = self.model.predict(state, verbose=0)
        return np.argmax(actions[0])
    
    def batch_train(self, batch_size):
        batch = random.sample(self.memory, batch_size) # (state, action, reward, next_state, done)
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        targets = self.model.predict(states, verbose=0)
        targets_next = self.model.predict(next_states, verbose=0)

        # Flatten the targets arrays to have shape (batch_size, action_space)
        targets = targets.reshape(batch_size, self.action_space)

        # print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape, targets.shape, targets_next.shape)

        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                # print(targets.shape, targets)
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(targets_next[i])

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay