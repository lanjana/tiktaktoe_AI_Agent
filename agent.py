import tensorflow as tf
import numpy as np
from collections import deque


class TTT_Agent:
    def __init__(self, symbol):
        self.symbol = symbol
        self.learning_rate = 0.001
        self.gamma = 0.95

        self.input_size = 9
        self.output_size = 9

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()

        self.batch_size = 1000

        self.memory = deque(maxlen=10_000)

        self.epsilon = 1
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.99

        # self.model = tf.keras.models.load_model('./agent2.keras')
        self.build_model()

    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=(self.input_size,)))
        self.model.add(tf.keras.layers.Dense(256, activation="relu"))
        self.model.add(tf.keras.layers.Dense(self.output_size, activation='softmax'))

        self.model.compile(
            optimizer="adam", loss="categorial_crossentropy", metrics=["accuracy"])

    def get_state(self, board):
        state = []
        for i in range(len(board)):
            if board[i] == self.symbol:
                state.append(1)
            elif board[i] == " ":
                state.append(0)
            else:
                state.append(-1)

        return np.array(state)

    def act(self, state):
        self.state = state

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.output_size)
        else:
            state = state.reshape(-1,9)
            action = self.model.predict(state, verbose=0)[0]
            action = np.argmax(action)

        model_out = [0] * self.output_size
        model_out[action] = 1
        self.action = model_out

        return action

    def remember(self, reward, next_state, done):
        self.memory.append((self.state, self.action, reward, next_state, done))
        if done:
            self.train(short_memory=False)
        else:
            self.train(short_memory=True)

    def train(self, short_memory):
        if short_memory:
            sample = [self.memory[-1]]
        # elif len(self.memory) > self.batch_size:
        #     sample = np.random.cho(self.memory, size=self.batch_size)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)

        states = np.array(states).reshape(-1, self.input_size)
        actions = np.array(actions).reshape(-1, self.output_size)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states).reshape(-1, self.input_size)
        dones = np.array(dones).reshape(-1, 1)

        # next_actions = self.model.predict(next_states, verbose=0)
        Q_new = np.copy(rewards)
        Q_new = (1-self.gamma) * Q_new + (self.gamma) * Q_new

        targets = np.copy(actions)
        for ind in range(actions.shape[0]):
            action_ind = np.argmax(targets[ind])
            for i in range(3):
                targets[ind, i] += -Q_new[ind]
            targets[ind, action_ind] += 2*Q_new[ind]

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        with tf.GradientTape() as tape:
            pred = self.model(states, training=True)
            loss = self.loss(targets, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
