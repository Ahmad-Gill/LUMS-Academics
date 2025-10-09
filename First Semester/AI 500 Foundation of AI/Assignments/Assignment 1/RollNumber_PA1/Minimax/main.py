from M252800665_MiniMax import MathDuel

def main():
    game = MathDuel(start_number=17, allowed_moves=[1, 2, 3])

    print("Welcome to Math Duel!")
    print(f"Starting number: {game.start_number}")
    print(f"Allowed moves: {game.allowed_moves}")
    print("Player 1 (You) vs AI")
    print("The player who makes the number reach 0 wins!\n")

    while not game.game_over:
        game.print_game_state()

        if game.player_turn == 1:  # Human player
            while True:
                try:
                    move = int(input("Your move: "))
                    if move in game.allowed_moves and move <= game.current_number:
                        break
                    else:
                        print(f"Invalid move. Choose from {[m for m in game.allowed_moves if m <= game.current_number]}")
                except ValueError:
                    print("Please enter a valid number.")
        else:  # AI player
            print("AI is thinking...")
            move = game.get_best_move()
            print(f"AI plays: {move}")

        game.make_move(move)

    # Show final state
    game.print_game_state()

if __name__ == "__main__":
    main()
