from enum import Enum

class Player(Enum):
    X = 'X'
    O = 'O'

class AtaxxGame:
    BOARD_SIZE = 7
    DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))

    def __init__(self):
        # Initialize the game board, current player, and place initial pieces
        self.board = [[' ' for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.current_player = Player.X
        self.place_initial_pieces()

    def place_initial_pieces(self):
        # Place initial pieces at specified positions on the board
        positions = [(0, 0), (self.BOARD_SIZE - 1, self.BOARD_SIZE - 1),
                     (0, self.BOARD_SIZE - 1), (self.BOARD_SIZE - 1, 0)]
        for i, j in positions:
            self.board[i][j] = Player.X.value if (i, j) in positions[:2] else Player.O.value

    def display_board(self):
        # Display the current state of the board
        for row in self.board:
            print(' '.join(row))

    def is_valid_move(self, x, y):
        # Check if a move is valid based on the board boundaries and empty cell
        return 0 <= x < self.BOARD_SIZE and 0 <= y < self.BOARD_SIZE and self.board[x][y] == ' '

    def get_opponent_piece(self):
        # Get the opponent's game piece
        return Player.O.value if self.current_player == Player.X else Player.X.value

    def make_move(self, x, y):
        # Make a move on the board if it's valid
        if not self.is_valid_move(x, y):
            return False

        player_piece = self.current_player.value
        self.board[x][y] = player_piece
        self.expand_pieces(x, y)
        self.current_player = Player.O if self.current_player == Player.X else Player.X
        return True

    def expand_pieces(self, x, y):
        # Expand pieces in all directions after a move
        player_piece = self.current_player.value
        opponent_piece = self.get_opponent_piece()

        for dx, dy in self.DIRECTIONS:
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.BOARD_SIZE and 0 <= ny < self.BOARD_SIZE and self.board[nx][ny] == opponent_piece:
                self.board[nx][ny] = player_piece
                nx += dx
                ny += dy

    def evaluate_board(self):
        # Evaluate the current board's state and calculate the score
        player_piece = self.current_player.value
        opponent_piece = self.get_opponent_piece()

        player_score = sum(row.count(player_piece) for row in self.board)
        opponent_score = sum(row.count(opponent_piece) for row in self.board)

        return player_score - opponent_score

    def minimax(self, depth, is_maximizing):
        # Implement the minimax algorithm for finding the best move
        if depth == 0:
            return self.evaluate_board()

        best_score = float('-inf') if is_maximizing else float('inf')

        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if self.is_valid_move(i, j):
                    self.board[i][j] = self.current_player.value
                    self.expand_pieces(i, j)
                    score = self.minimax(depth - 1, not is_maximizing)
                    self.board[i][j] = ' '
                    self.current_player = Player.O if self.current_player == Player.X else Player.X

                    if is_maximizing:
                        best_score = max(best_score, score)
                    else:
                        best_score = min(best_score, score)

        return best_score

    def get_best_move(self, depth):
        # Get the best move using the minimax algorithm
        best_move = None
        best_score = float('-inf')

        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if self.is_valid_move(i, j):
                    self.board[i][j] = self.current_player.value
                    self.expand_pieces(i, j)
                    score = self.minimax(depth - 1, False)
                    self.board[i][j] = ' '
                    self.current_player = Player.O if self.current_player == Player.X else Player.X

                    if score > best_score:
                        best_score = score
                        best_move = (i, j)

        return best_move

    def play_game(self):
        # Main game loop to play the Ataxx game
        while True:
            self.display_board()
            print(f"Player {self.current_player.value}'s turn.")

            if self.current_player == Player.O:
                # AI's turn, get the best move using minimax
                depth = 3
                ai_move = self.get_best_move(depth)
                print(f"AI plays at ({ai_move[0]}, {ai_move[1]})")
                self.make_move(*ai_move)
            else:
                # Human player's turn, get input for the move
                x, y = map(int, input('Enter the coordinates (row and column) for your move, separated by a space: ').split())
                if self.make_move(x, y):
                    if sum(row.count(Player.X.value) for row in self.board) == 0:
                        self.display_board()
                        print(f'Player {Player.O.value} wins!')
                        break
                    elif not any(' ' in row for row in self.board):
                        self.display_board()
                        print('It\'s a tie!')
                        break

if __name__ == "__main__":
    # Start the game when the script is run
    game = AtaxxGame()
    game.play_game()
