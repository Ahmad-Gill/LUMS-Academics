import math
import time

class MathDuel:
    def __init__(self, start_number=16, allowed_moves=None):
        if allowed_moves is None:
            allowed_moves = [1, 2, 3]
        self.start_number = start_number
        self.allowed_moves = allowed_moves
        self.current_number = start_number
        self.player_turn = 1  # Player 1 starts (MAX)
        self.game_over = False
        self.winner = None
        self.move_history = []

    def reset(self):
        """Reset the game state."""
        self.current_number = self.start_number
        self.player_turn = 1
        self.game_over = False
        self.winner = None
        self.move_history = []

    def make_move(self, move):
        """Make a move: subtract move, update winner and switch turns."""
        if move > self.current_number or move not in self.allowed_moves:
            print("Choose a correct move!")
            return False
        self.current_number -= move
        self.move_history.append((self.player_turn, move))
        if self.current_number == 0:
            self.winner = self.player_turn
            self.game_over = True
        else:
            self.player_turn = 1 - self.player_turn
        return True
    def evaluate(self):  
        """TODO: Implement evaluation function. Return: +1 if MAX wins -1 if MIN wins 0 if game not over """
        if self.winner==1:
            return 1 
        elif self.winner==0:
            return -1
        return 0

    def minimax(self, state, depth, is_maximizing, alpha=-math.inf, beta=math.inf):
        if state == 0:
            return self.evaluate()
        if depth == 0:
            return 0

        if is_maximizing:
            best_val = -math.inf
            for move in self.allowed_moves:
                if move <= state:
                    current_number_backup = self.current_number
                    player_turn_backup = self.player_turn
                    winner_backup = self.winner
                    game_over_backup = self.game_over
                    self.current_number = state
                    self.player_turn = 1 
                    self.make_move(move)
                    val = self.minimax(self.current_number, depth - 1, False, alpha, beta)
                    self.current_number = current_number_backup
                    self.player_turn = player_turn_backup
                    self.winner = winner_backup
                    self.game_over = game_over_backup
                    best_val = max(best_val, val)
                    alpha = max(alpha, best_val)
                    if beta <= alpha:
                        break
            return best_val
        else:
            best_val = math.inf
            for move in self.allowed_moves:
                if move <= state:
                    current_number_backup = self.current_number
                    player_turn_backup = self.player_turn
                    winner_backup = self.winner
                    game_over_backup = self.game_over
                    self.current_number = state
                    self.player_turn = 0 
                    self.make_move(move)
                    val = self.minimax(self.current_number, depth - 1, True, alpha, beta)
                    self.current_number = current_number_backup
                    self.player_turn = player_turn_backup
                    self.winner = winner_backup
                    self.game_over = game_over_backup
                    best_val = min(best_val, val)
                    beta = min(beta, best_val)
                    if beta <= alpha:
                        break
            return best_val



    def get_best_move(self, depth=10):
        """Find the optimal move for the current player using minimax."""
        is_maximizing = (self.player_turn == 1)
        best_move = None
        best_val = -math.inf if is_maximizing else math.inf

        for move in self.allowed_moves:
            if move <= self.current_number:
                val = self.minimax(self.current_number - move, depth - 1, not is_maximizing)
                if (is_maximizing and val > best_val) or (not is_maximizing and val < best_val):
                    best_val = val
                    best_move = move

        return best_move

    def print_game_state(self):
        """Print the current state of the game."""
        player_name = "Player 1" if self.player_turn == 1 else "AI"
        role = "MAX" if self.player_turn == 1 else "MIN"
        print(f"\nCurrent number: {self.current_number}")
        print(f"{player_name}'s turn ({role})")
        print(f"Allowed moves: {[m for m in self.allowed_moves if m <= self.current_number]}")
        if self.game_over:
            print(f"Game over! {player_name} wins!")
