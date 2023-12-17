from agent import TTT_Agent
from game import TicTacToe


if __name__ == '__main__':
    agent1 = TTT_Agent("X")
    agent2 = TTT_Agent("O")

    game = TicTacToe()

    while True:
        game.display_board()

        move = False
        while not move:
            state1 = agent1.get_state(game.board)
            action1 = agent1.act(state1)
            move = game.make_move(action1)
            reward = game.x_player_reward
            next_state = agent1.get_state(game.board)

            agent1.remember(reward, next_state, False)

        action2 = agent2.act(state)
