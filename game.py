

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.x_player_reward = 0
        self.y_player_reward = 0
        self.no_x_wins = 0
        self.no_y_wins = 0

    def display_board(self):
        for i in range(0, 9, 3):
            print(
                f"{self.board[i]} | {self.board[i + 1]} | {self.board[i + 2]}")
            if i < 6:
                print("-" * 9)

    def make_move(self, position):
        self.x_player_reward = 0
        self.y_player_reward = 0
        if self.board[position] == ' ':
            self.board[position] = self.current_player
            self.switch_player()
            return True

        else:
            print("Invalid move. The position is already occupied.")
            if self.current_player == "X":
                self.x_player_reward = -10
            else:
                self.y_player_reward = -10

            return False

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        self.x_player_reward = 0
        self.y_player_reward = 0
        win = False
        for i in range(0, 3):
            if (self.board[i] == self.board[i + 3] == self.board[i + 6] != ' ') or (self.board[i * 3] == self.board[i * 3 + 1] == self.board[i * 3 + 2] != ' '):
                win = True

        if self.board[0] == self.board[4] == self.board[8] != ' ' or \
           self.board[2] == self.board[4] == self.board[6] != ' ':
            win = True

        if win:
            if self.current_player == "X":
                self.y_player_reward = 5
                self.x_player_reward = 0
                self.no_y_wins += 1
            else:
                self.x_player_reward = 5
                self.y_player_reward = 0
                self.no_x_wins += 1
            return True

        if ' ' not in self.board:
            self.x_player_reward = 0
            self.y_player_reward = 0
            return True

        return False

    def restart(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.x_player_reward = -2
        self.y_player_reward = -2
