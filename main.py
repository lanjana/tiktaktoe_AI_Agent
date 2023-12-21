from agent import TTT_Agent
from game import TicTacToe
import numpy as np
import pickle as pk
import matplotlib as plt


def save_func():
    global agent1, agent2, count, game, done, score_X, score_Y
    agent1.model.save("./agent1.keras")

    with open("./data_sheet.pkl", "wb") as file:
        pk.dump(agent1.memory, file)

    with open("./score_X.pkl", "wb") as file:
        pk.dump(score_X, file)

    with open("./score_Y.pkl", "wb") as file:
        pk.dump(score_Y, file)


def done_func():
    global agent1, agent2, count, game, done, score_X, score_Y
    if done:
        count += 1
        print(game.no_y_wins, game.no_x_wins,
              count, round(agent1.epsilon, 3))

        score_X.append(game.no_x_wins)
        score_Y.append(game.no_y_wins)

        if count % 100 == 0:
            agent1.train2(short_memory=False)
            # agent2.train2(short_memory=False)

        if agent1.epsilon > agent1.min_epsilon:
            agent1.epsilon *= agent1.epsilon_decay
            agent2.epsilon *= agent2.epsilon_decay
        game.restart()

        # agent1.model.save("agent1.keras")
        # agent2.model.save("agent2.keras")
        return True
    else:
        return False


agent1 = TTT_Agent("X")
agent2 = TTT_Agent("O")
game = TicTacToe()

score_X, score_Y = [], []


count = 0
while True:
    move, done = False, False
    while not (move or done):
        # move = game.make_move(np.random.choice(
        #     [ind for ind, value in enumerate(game.board) if value == " "]))
        # done = game.check_winner()

        state1 = agent1.get_state(game.board)
        action1 = agent1.act(state1)
        move = game.make_move(action1)
        reward = game.x_player_reward
        next_state = agent1.get_state(game.board)
        done = game.check_winner()
        agent1.remember(reward, next_state, done)
        # agent1.train(short_memory=True)

    if done_func():
        continue

    game.display_board()
    move, done = False, False
    while not (move or done):
        move = game.make_move(int(input("insert number")))
        # move = game.make_move(np.random.choice(
        #     [ind for ind, value in enumerate(game.board) if value == " "]))
        done = game.check_winner()

        # state2 = agent2.get_state(game.board)
        # action2 = agent2.act(state2)
        # move = game.make_move(action2)
        # reward = game.y_player_reward
        # next_state = agent2.get_state(game.board)
        # done = game.check_winner()
        # agent2.remember(reward, next_state, done)
        # agent2.train(short_memory=True)

    if done_func():
        continue
