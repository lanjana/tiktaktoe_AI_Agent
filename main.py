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
            done = game.check_winner()
            agent1.remember(reward, next_state, done)

        if done:
            game.restart()
            agent1.epsilon *= agent1.epsilon_decay
            agent2.epsilon *= agent2.epsilon_decay
            agent1.model.save("agent1.keras")
            agent2.model.save("agent2.keras")
            continue
        
        while not move:
            # position = int(input("insert number"))
            # move = True
            # game.make_move(position)

            state2 = agent2.get_state(game.board)
            action2 = agent2.act(state2)
            move = game.make_move(action2)
            reward = game.y_player_reward
            next_state = agent2.get_state(game.board)
            done = game.check_winner()
            agent2.remember(reward, next_state, done)

        # if done:
        #     agent1.model.save("agent1.keras")
        #     agent2.model.save("agent2.keras")
        #     game.restart()


