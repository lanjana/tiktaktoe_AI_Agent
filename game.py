

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.x_player_reward = 0
        self.y_player_reward = 0

    def display_board(self):
        # pass
        for i in range(0, 9, 3):
            print(
                f"{self.board[i]} | {self.board[i + 1]} | {self.board[i + 2]}")
            if i < 6:
                print("-" * 9)

    def make_move(self, position):
        if self.board[position] == ' ':
            self.board[position] = self.current_player
            self.switch_player()

            return True
        else:
            # print("Invalid move. The position is already occupied.")
            if self.current_player == "X":
                self.x_player_reward = -1
            else:
                self.y_player_reward = 0

            return False

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        win = False
        for i in range(0, 3):
            if (self.board[i] == self.board[i + 3] == self.board[i + 6] != ' ') or (self.board[i * 3] == self.board[i * 3 + 1] == self.board[i * 3 + 2] != ' '):
                win = True

        if self.board[0] == self.board[4] == self.board[8] != ' ' or \
           self.board[2] == self.board[4] == self.board[6] != ' ':
            win = True

        if win:
            if self.current_player == "X":
                self.x_player_reward = 10
            else:
                self.y_player_reward = 10
            return True

        if ' ' not in self.board:
            self.x_player_reward = 2
            self.y_player_reward = 2
            return True

        return False

    def restart(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.x_player_reward = 0
        self.y_player_reward = 0


if __name__ == "__main__":
    game = TicTacToe()

    while not game.check_winner():
        game.display_board()
        try:
            position = int(
                input(f"{game.current_player}'s turn. Enter position (1-9): ")) - 1
            if 0 <= position <= 8:
                if game.make_move(position):
                    continue
            else:
                pass
                # print("Invalid input. Please enter a number between 1 and 9.")
        except ValueError:
            pass
            # print("Invalid input. Please enter a number.")

    game.display_board()
