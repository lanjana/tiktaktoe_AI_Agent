import pickle as pk
import tensorflow as tf
import numpy as np
from collections import deque


class TTT_Agent:
    def __init__(self, symbol):
        self.symbol = symbol
        self.learning_rate = 0.01
        self.gamma = 0.95

        self.input_size = 9
        self.output_size = 9

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()

        self.batch_size = 1000

        self.memory = deque(maxlen=2_000)

        self.epsilon = 1
        self.min_epsilon = 0.001
        self.epsilon_decay = 0.999

        # self.model = tf.keras.models.load_model('./Models/agent1.keras')
        self.build_model()

    def build_model(self):
        self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Flatten(input_shape=(self.input_size,)))
        self.model.add(tf.keras.layers.Dense(
            350, activation="relu", input_shape=(self.input_size,)))
        self.model.add(tf.keras.layers.Dense(250, activation="relu"))
        self.model.add(tf.keras.layers.Dense(100, activation="relu"))
        self.model.add(tf.keras.layers.Dense(50, activation="relu"))

        self.model.add(tf.keras.layers.Dense(
            self.output_size, activation='linear'))

        self.model.compile(optimizer="adam", loss="mse")

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
        self.state = np.array(state).reshape(-1, self.input_size)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.output_size)
            model_out = [0] * self.output_size
            model_out[action] = 1
        else:
            model_out = self.model.predict(self.state, verbose=0)[0]
            action = np.argmax(model_out)

        self.action = model_out

        return action

    def remember(self, reward, next_state, done):
        self.memory.append((
            self.state, self.action, reward, next_state, done
        ))

        # if done:
        #     self.train(short_memory=False)
        # else:
        #     self.train(short_memory=True)

    def train2(self, short_memory):
        if short_memory:
            sample = [self.memory[-1]]
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)

        states = np.array(states).reshape(-1, self.input_size)
        next_states = np.array(next_states).reshape(-1, self.input_size)
        rewards = np.array(rewards).reshape(-1, 1)
        actions = np.array(actions).reshape(-1, self.output_size)

        next_rewards = self.model.predict(next_states, verbose=0)
        targets = self.model.predict(states, verbose=0)

        for ind in range(len(states)):
            Q_new = rewards[ind]
            if not dones[ind]:
                Q_new = rewards[ind] + self.gamma * np.max(next_rewards[ind])

            action_ind = np.argmax(targets[ind])

            targets[ind, action_ind] = Q_new

        unoccupied = np.where(states != 0)
        rows, cols = unoccupied
        targets[rows, cols] = -10

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True)

        self.model.fit(states, targets, epochs=100,
                       batch_size=5, callbacks=[early_stopping])

    def train(self, short_memory):
        if short_memory:
            sample = [self.memory[-1]]
        # elif len(self.memory) > self.batch_size:
        #     sample = np.random.choice(self.memory, size=self.batch_size)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)

        states = np.array(states).reshape(-1, self.input_size)
        next_states = np.array(next_states).reshape(-1, self.input_size)
        rewards = np.array(rewards).reshape(-1, 1)
        actions = np.array(actions).reshape(-1, self.output_size)
        next_rewards = self.model.predict(next_states, verbose=0)

        states = tf.convert_to_tensor(states, dtype=tf.float32)

        with tf.GradientTape() as tape:
            pred = self.model(states, training=True)
            targets = np.copy(pred)
            for ind in range(len(states)):
                Q_new = rewards[ind]
                if not dones[ind]:
                    Q_new = rewards[ind] + self.gamma * \
                        np.max(next_rewards[ind])

                action_ind = np.argmax(targets[ind])
                targets[ind, action_ind] += Q_new

            unoccupied = np.where(states != 0)
            rows, cols = unoccupied
            targets[rows, cols] = -10

            targets = tf.convert_to_tensor(targets, dtype=tf.float32)

            loss = self.loss(targets, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
