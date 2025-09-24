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
        """TODO: Implement making a move.
        - Subtract `move` from current_number
        - Check if the game is over
        - Switch turns
        """
        if move > self.current_number or move not in self.allowed_moves:
             print("chuse the corect movenumber ")
             return 
        else:
                self.current_number -=move
                if self.current_number == 0 :
                    self.winner=self.player_turn
                else:
                    self.player_turn =1-self.player_turn
                
        return

    def evaluate(self):
        """TODO: Implement evaluation function.
        Return:
          +1 if MAX wins
          -1 if MIN wins
           0 if game not over
        """
        if self.winner==1:
            return 1
        elif self.winner==0:
            return -1
        return 0

    def minimax(self, state, depth, is_maximizing, alpha=-math.inf, beta=math.inf):
        """TODO: Implement minimax with alpha-beta pruning.
        Arguments:
          - state: current number
          - depth: search depth
          - is_maximizing: True if MAX's turn, False if MIN's
        Return:
          Evaluation value for the state
        """
        pass

    def get_best_move(self):
        """TODO: Use minimax to find the best move for the current player."""
        pass

    def print_game_state(self):
        """Print the current state of the game."""
        player_name = "Player 1" if self.player_turn == 1 else "AI"
        role = "MAX" if self.player_turn == 1 else "MIN"
        print(f"\nCurrent number: {self.current_number}")
        print(f"{player_name}'s turn ({role})")
        print(f"Allowed moves: {[m for m in self.allowed_moves if m <= self.current_number]}")
        if self.game_over:
            print(f"Game over! {player_name} wins!")

