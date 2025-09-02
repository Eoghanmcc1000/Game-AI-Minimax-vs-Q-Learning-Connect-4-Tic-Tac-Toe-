# Tic Tac Toe
import random
import time  # Added for timing metrics
#import numpy as np

class TicTacToe:
    def __init__(self):
        # Initialize empty board as list of 9 spaces
        self.board = [" " for _ in range(9)]
        # Set player symbols - O for human player, X for AI
        self.player1 = "O"
        self.ai_player = "X"
        self.ai_player2 = "O"
        self.random = "O"
        # Add metrics for minimax
        self.minimax_stats = {
            'states_visited': 0,
            'total_games': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'unique_states': set(),  # Track unique board states
            'decision_times': [],    # Track time per decision
            'moves_per_game': []     # Track moves per game
        }

    def print_board(self, board):
        """Print the current state of the board"""
        # Print board row by row with separators
        for i in range(0, 9, 3):
            # Print each row with | separators between cells
            print(f"{board[i]} | {board[i+1]} | {board[i+2]}")
            # Print horizontal line between rows, except after last row
            if i < 6:
                print("---------")

    def available_moves(self, board):
        """Return a list of available moves"""
        # Return indices of empty spots on the board using list comprehension
        return [i for i, spot in enumerate(board) if spot == " "]
    
    def make_move(self, move, player, board):
        """Make a move on the board"""
        # Check if selected spot is empty
        if board[move] == " ":
            # Place player's symbol in the spot
            board[move] = player
            return True
        # Return False if spot is already taken
        return False
    
    def is_board_full(self, board):
        """Check if the board is full"""
        # Return True if all spots on the board are taken
        return " " not in board
    
    def check_winner(self, board):
        """ Check if there is a winner. Returns the winner if there is one, otherwise returns None """
        # check rows
        for i in range(0, 9, 3):
            if board[i] == board[i+1] == board[i+2] != " ":
                return board[i]
        # check columns
        for i in range(3):
            if board[i] == board[i+3] == board[i+6] != " ":
                return board[i]
        # check diagonals
        if board[0] == board[4] == board[8] != " ":
            return board[0]
        if board[2] == board[4] == board[6] != " ":
            return board[2]
        # if no winner, return None
        return None
    
    def game_over(self, board):
        """ Check if the game is over """
        return self.check_winner(board) is not None or self.is_board_full(board)
    
    def find_winning_move(self, board, player):
        """Find a winning move for the given player"""
        for move in self.available_moves(board):
            # Try the move
            board_copy = board.copy()
            board_copy[move] = player
            # Check if this move results in a win
            if self.check_winner(board_copy) == player:
                return move
        # No winning move found
        return None
    
    def get_strategic_move(self, board, player):
        """Strategic player that only wins or blocks, otherwise random"""
        # First, check if we can win
        winning_move = self.find_winning_move(board, player)
        if winning_move is not None:
            return winning_move
            
        # If no winning move, check if we need to block opponent
        opponent = self.ai_player if player == self.random else self.random
        blocking_move = self.find_winning_move(board, opponent)
        if blocking_move is not None:
            return blocking_move
            
        # Otherwise, make a random move
        return random.choice(self.available_moves(board))
    
    def minimax(self, board, depth, is_maximizing):
        # Count unique states visited
        board_tuple = tuple(board)
        if board_tuple not in self.minimax_stats['unique_states']:
            self.minimax_stats['unique_states'].add(board_tuple)
            self.minimax_stats['states_visited'] += 1
        
        # base case - check if the previous move is a winner
        if self.check_winner(board) == self.ai_player:
            return 1
        if self.check_winner(board) == self.random:  #self.player1
            return -1
        if self.is_board_full(board):
            return 0
        
        # Count branches explored only for non-terminal states
        available_moves = self.available_moves(board)
        
        # depth = levels in recursion 
        # is_maximizing = True if the current player is the AI, False if the current player is the human

        if is_maximizing:
            best_score = float('-inf')
            for move in available_moves:
                # make the move on a copy of the board
                new_board = board.copy()
                new_board[move] = self.ai_player
                # recursive call to minimax with the new board
                score = self.minimax(new_board, depth + 1, False)
                # update the best score
                best_score = max(score, best_score)
            return best_score    

        else:
            best_score = float('inf')
            for move in available_moves:
                # make the move on a copy of the board
                new_board = board.copy()
                new_board[move] = self.random  #self.player1
                # recursive call to minimax with the new board
                score = self.minimax(new_board, depth + 1, True)
                # update the best score
                best_score = min(score, best_score)
            return best_score
        
    def minimax_alpha_beta(self, board, depth, is_maximizing, alpha=float('-inf'), beta=float('inf')):
        # Count unique states visited
        board_tuple = tuple(board)
        if board_tuple not in self.minimax_stats['unique_states']:
            self.minimax_stats['unique_states'].add(board_tuple)
            self.minimax_stats['states_visited'] += 1
        
        # Base case - check for terminal states
        if self.check_winner(board) == self.ai_player:
            return 1
        if self.check_winner(board) == self.random:
            return -1
        if self.is_board_full(board):
            return 0
        
        # Count branches explored only for non-terminal states
        available_moves = self.available_moves(board)
        
        if is_maximizing:
            best_score = float('-inf')
            for move in available_moves:
                # Make the move on a copy of the board
                new_board = board.copy()
                new_board[move] = self.ai_player
                # Recursive call with updated alpha
                score = self.minimax_alpha_beta(new_board, depth + 1, False, alpha, beta)
                best_score = max(score, best_score)
                # Update alpha (best for maximizer)
                alpha = max(alpha, best_score)
                # Prune if we can't improve the minimizer's best option
                if beta <= alpha:
                    break
            return best_score

        else:
            best_score = float('inf')
            for move in available_moves:
                # Make the move on a copy of the board
                new_board = board.copy()
                new_board[move] = self.random
                # Recursive call with updated beta
                score = self.minimax_alpha_beta(new_board, depth + 1, True, alpha, beta)
                best_score = min(score, best_score)
                # Update beta (best for minimizer)
                beta = min(beta, best_score)
                # Prune if we can't improve the maximizer's best option
                if beta <= alpha:
                    break
            return best_score

    def get_ai_move(self):
        """ Get the AI's move """
        # Start timing
        start_time = time.time()
        
        # start with the best score as negative infinity
        best_score = float('-inf')
        # start with no move
        move = None
        # check each available move
        available_moves = self.available_moves(self.board)
        # alpha and beta for alpha-beta pruning
        alpha = float('-inf')
        beta = float('inf')
        
        for i in available_moves:
            # make the move on a copy of the board
            new_board = self.board.copy()
            new_board[i] = self.ai_player
            score = self.minimax_alpha_beta(new_board, 0, False, alpha, beta)
            if score > best_score:
                best_score = score
                move = i
            # Update alpha as we find better moves
            alpha = max(alpha, best_score)
        
        # Record decision time
        end_time = time.time()
        self.minimax_stats['decision_times'].append(end_time - start_time)
        
        return move

    def get_ai_move_original(self):
        """ Get the AI's move using original minimax without pruning """
        # Start timing
        start_time = time.time()
        
        best_score = float('-inf')
        move = None
        available_moves = self.available_moves(self.board)
        
        for i in available_moves:
            new_board = self.board.copy()
            new_board[i] = self.ai_player
            score = self.minimax(new_board, 0, False)  # Use original minimax
            if score > best_score:
                best_score = score
                move = i
        
        # Record decision time
        end_time = time.time()
        self.minimax_stats['decision_times'].append(end_time - start_time)
        
        return move

    def play(self, opponent_type="random", ai_goes_first=None):
        """Play a game with specified opponent type
        opponent_type: 'random', 'strategic', or 'minimax'
        ai_goes_first: True if AI goes first, False if opponent goes first, None for random
        """
        import random
        move_count = 0
        
        if ai_goes_first is None:
            # Random starting player if not specified
            current_player = random.choice([self.random, self.ai_player])
        else:
            # Set starting player based on ai_goes_first parameter
            current_player = self.ai_player if ai_goes_first else self.random
        
        while not self.game_over(self.board):
            move_count += 1
            if current_player == self.ai_player:
                move = self.get_ai_move()
                self.make_move(move, self.ai_player, self.board)
            else:
                # Different opponent strategies
                if opponent_type == "random":
                    move = random.choice(self.available_moves(self.board))
                elif opponent_type == "strategic":
                    move = self.get_strategic_move(self.board, self.random)
                elif opponent_type == "minimax":
                    # For simplicity, opponent uses the same minimax but with opposite player perspective
                    temp_ai = self.ai_player
                    self.ai_player = self.random
                    self.random = temp_ai
                    move = self.get_ai_move()
                    # Restore original player settings
                    temp_ai = self.ai_player
                    self.ai_player = self.random
                    self.random = temp_ai
                else:
                    # Default to random
                    move = random.choice(self.available_moves(self.board))
                    
                self.make_move(move, self.random, self.board)
            
            # Switch players
            current_player = self.ai_player if current_player == self.random else self.random
            
        # Record moves count
        self.minimax_stats['moves_per_game'].append(move_count)
        return move_count

    def reset_board(self):
        """Reset the board to empty state"""
        self.board = [" " for _ in range(9)]

    def game_simulation(self, num_games=100, opponent_type="random", ai_goes_first=None):
        self.minimax_stats = {
            'states_visited': 0,
            'total_games': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'unique_states': set(),  # Reset unique states set
            'decision_times': [],     # Reset decision times
            'moves_per_game': []      # Reset moves per game
        }
        """ Simulate the game """
        ai_position = "random" if ai_goes_first is None else ("first" if ai_goes_first else "second")
        print(f"Simulating {num_games} games against {opponent_type} opponent (AI goes {ai_position})...")
        for i in range(num_games): 
            self.reset_board()  # Reset the board before each game
            moves = self.play(opponent_type, ai_goes_first)
            if self.check_winner(self.board) == self.ai_player:
                self.minimax_stats['wins'] += 1
            elif self.check_winner(self.board) == self.random:
                self.minimax_stats['losses'] += 1
            else:
                self.minimax_stats['draws'] += 1
            print(f"Game {i+1} complete - {moves} moves")
            self.minimax_stats['total_games'] += 1
        
        print("\nMinimax Statistics:")
        print(f"Total games: {self.minimax_stats['total_games']}")
        print(f"Wins: {self.minimax_stats['wins']} ({self.minimax_stats['wins']/self.minimax_stats['total_games']*100:.2f}%)")
        print(f"Losses: {self.minimax_stats['losses']} ({self.minimax_stats['losses']/self.minimax_stats['total_games']*100:.2f}%)")
        print(f"Draws: {self.minimax_stats['draws']} ({self.minimax_stats['draws']/self.minimax_stats['total_games']*100:.2f}%)")
        print(f"Total unique states visited: {self.minimax_stats['states_visited']}")
        print(f"Average unique states per game: {self.minimax_stats['states_visited']/self.minimax_stats['total_games']:.2f}")
        print(f"Average decision time: {sum(self.minimax_stats['decision_times'])/len(self.minimax_stats['decision_times'])*1000:.2f} ms")
        print(f"Average moves per game: {sum(self.minimax_stats['moves_per_game'])/len(self.minimax_stats['moves_per_game']):.2f}")

    def play_demo_game(self, opponent_type="strategic"):
        """Play a single game and print each step to verify game logic"""
        self.reset_board()
        print("Starting a demo game")
        current_player = random.choice([self.random, self.ai_player])
        print(f"First player: {'AI' if current_player == self.ai_player else 'Opponent'}")
        
        move_count = 0
        while not self.game_over(self.board):
            move_count += 1
            print(f"\nMove {move_count}:")
            if current_player == self.ai_player:
                print("AI's turn")
                move = self.get_ai_move()
                self.make_move(move, self.ai_player, self.board)
                print(f"AI chose position {move}")
            else:
                print("Opponent's turn")
                if opponent_type == "random":
                    move = random.choice(self.available_moves(self.board))
                    print(f"Random opponent chose position {move}")
                elif opponent_type == "strategic":
                    move = self.get_strategic_move(self.board, self.random)
                    print(f"Strategic opponent chose position {move}")
                else:
                    move = random.choice(self.available_moves(self.board))
                    print(f"Default opponent chose position {move}")
                self.make_move(move, self.random, self.board)
            
            # Print board after each move
            self.print_board(self.board)
            
            # Switch players
            current_player = self.ai_player if current_player == self.random else self.random
        
        # Print game result
        if self.check_winner(self.board) == self.ai_player:
            print("\nAI wins!")
        elif self.check_winner(self.board) == self.random:
            print("\nOpponent wins!")
        else:
            print("\nIt's a draw!")

class Q_learning(TicTacToe):
    def __init__(self):
        super().__init__()
        self.q_table = {}
        self.states = []
        self.actions = []
        self.board = [" " for _ in range(9)]
        self.current_player = self.ai_player #self.player1
        # AI goes first
        self.training_stats = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'moves_per_game': []
        }
        # Add timing metrics
        self.q_decision_times = []
        self.training_time = 0
    
    def get_current_state(self):
        """ Get the current state of the board """
        return tuple(self.board)
    
    def get_possible_actions(self):
        """ Get the possible actions for the current state """
        return [i for i, spot in enumerate(self.board) if spot == " "]
    
    def get_q_move(self):
        """ Get a move based on Q-table values """
        # Start timing
        start_time = time.time()
        
        current_state = self.get_current_state()
        
        # If state exists in Q-table and has values
        if current_state in self.q_table and len(self.q_table[current_state]) > 0:
            action = max(self.q_table[current_state], key=self.q_table[current_state].get)
        else:
            # If state not in Q-table, choose random action
            action = random.choice(self.get_possible_actions())
        
        # Record decision time
        end_time = time.time()
        self.q_decision_times.append(end_time - start_time)
        
        return action
    
    def train(self, episodes=5000000, opponent_type="random"):
        """ Implement Q-learning """
        # Hyperparameters
        learning_rate = 0.1    # Moderate learning rate # 0.15
        discount_factor = 0.90   # Balanced discount
        exploration_rate = 0.3   # Moderate exploration # 0.3 
        exploration_decay = 0.999997 #0.99997  # Moderate decay

        # Start timing training
        training_start = time.time()

        print(f"Starting Q-learning training for {episodes} episodes against {opponent_type} opponent...")
        for episode in range(episodes):
            self.reset_board()
            game_over = False
            moves_this_game = 0

            while not game_over:
                # get current state
                moves_this_game += 1
                current_state = self.get_current_state()
                
                # AI's turn
                # Choose action
                if random.uniform(0, 1) < exploration_rate:
                    # Explore: choose random action
                    action = random.choice(self.get_possible_actions())
                else:
                    # Exploit: Choose best action based on Q-table
                    if current_state in self.q_table and len(self.q_table[current_state]) > 0:
                        action = max(self.q_table[current_state], key=self.q_table[current_state].get)
                    else:
                        # If current state not in Q-table, initialize with random actions
                        action = random.choice(self.get_possible_actions())

                # Make the move
                self.make_move(action, self.current_player, self.board)
                
                # Get reward and check if game is over
                reward = 0  # Default reward for non-terminal states
                if self.check_winner(self.board) == self.ai_player:
                    reward = 1.0  # Reward for win
                    game_over = True
                    self.training_stats['wins'] += 1
                elif self.check_winner(self.board) == self.random:
                    reward = -1.0  # Negative reward for loss
                    game_over = True
                    self.training_stats['losses'] += 1
                elif self.is_board_full(self.board):
                    reward = 1.0  # Small reward for draw
                    game_over = True
                    self.training_stats['draws'] += 1

                # Get next state and update Q-table
                next_state = self.get_current_state()

                # initialize Q-value if state not in Q-table
                if current_state not in self.q_table:
                    self.q_table[current_state] = {}
                if next_state not in self.q_table:
                    self.q_table[next_state] = {}
                
                # Get max future Q-value
                if game_over:
                    max_future_q = 0
                else:
                    if self.q_table[next_state]:    
                        max_future_q = max(self.q_table[next_state].values())
                    else:
                        max_future_q = 0

                # Update Q-value using equation
                # Q(s,a) = (1-α)Q(s,a) + α[R + γ max Q(s',a')]
                current_q = self.q_table[current_state].get(action, 0)
                sample = reward + discount_factor * max_future_q
                new_q = (1 - learning_rate) * current_q + learning_rate * sample
                self.q_table[current_state][action] = new_q

                # if game not over, let opponent move based on type
                if not game_over:
                    if opponent_type == "random" or (episode % 2 == 0 and opponent_type == "mixed"):
                        # Random opponent
                        random_move = random.choice(self.get_possible_actions())
                        self.make_move(random_move, self.random, self.board)
                    elif opponent_type == "strategic": #or (episode % 2 == 0 and opponent_type == "mixed"):
                        # Strategic opponent
                        strategic_move = self.get_strategic_move(self.board, self.random)
                        self.make_move(strategic_move, self.random, self.board)
                    elif opponent_type == "minimax" or (episode % 2 != 0 and opponent_type == "mixed"):
                        # Minimax opponent - use alpha-beta pruning for efficiency
                        # Temporarily swap player roles for minimax calculation
                        temp_ai = self.ai_player
                        self.ai_player = self.random
                        self.random = temp_ai
                        move = self.get_ai_move()
                        # Restore original player settings
                        temp_ai = self.ai_player
                        self.ai_player = self.random
                        self.random = temp_ai
                        self.make_move(move, self.random, self.board)
                    
                    # Check for game over after opponent's move
                    if self.check_winner(self.board) == self.random or self.check_winner(self.board) == self.ai_player2:
                        game_over = True
                        self.training_stats['losses'] += 1
                    elif self.check_winner(self.board) == self.ai_player:
                        game_over = True
                        self.training_stats['wins'] += 1
                    elif self.is_board_full(self.board):
                        game_over = True
                        self.training_stats['draws'] += 1

            # End of game updates
            self.training_stats['moves_per_game'].append(moves_this_game)
            exploration_rate *= exploration_decay

            # Print progress every 100 episodes
            if episode % 100 == 0:
                avg_moves = sum(self.training_stats['moves_per_game'][-100:]) / min(100, len(self.training_stats['moves_per_game']))
                win_rate = self.training_stats['wins'] / (episode + 1) * 100
                print(f"Episode {episode}")
                print(f"Win Rate: {win_rate:.2f}%")
                print(f"Average moves per game: {avg_moves:.2f}")
                print(f"Exploration rate: {exploration_rate:.2f}")
                print("-------------------")
        
        # Record total training time
        training_end = time.time()
        self.training_time = training_end - training_start
        
        # Print final statistics
        total_games = self.training_stats['wins'] + self.training_stats['losses'] + self.training_stats['draws']
        print("\nFinal Training Results:")
        print(f"Total games played: {total_games}")
        print(f"Wins: {self.training_stats['wins']} ({self.training_stats['wins']/total_games*100:.2f}%)")
        print(f"Losses: {self.training_stats['losses']} ({self.training_stats['losses']/total_games*100:.2f}%)")
        print(f"Draws: {self.training_stats['draws']} ({self.training_stats['draws']/total_games*100:.2f}%)")
        print(f"Average moves per game: {sum(self.training_stats['moves_per_game'])/len(self.training_stats['moves_per_game']):.2f}")
        print(f"Training time: {self.training_time:.2f} seconds")
        print(f"Q-table size: {len(self.q_table)} unique states")
        print(f"Total state-action pairs: {sum(len(actions) for actions in self.q_table.values())}")

    def play_against(self, opponent_type="strategic", num_games=100, ai_goes_first=None):
        """Test Q-learning agent against different opponents
        opponent_type: 'random', 'strategic', or 'minimax'
        ai_goes_first: True if AI goes first, False if opponent goes first, None for random
        """
        wins = 0
        losses = 0
        draws = 0
        moves_per_game = []
        decision_times = []
        
        print(f"\nTesting Q-learning against {opponent_type} opponent ({num_games} games)...")
        
        for i in range(num_games):
            self.reset_board()
            move_count = 0
            game_over = False
            
            if ai_goes_first is None:
                # Random starting player if not specified
                current_player = random.choice([self.ai_player, self.random])
            else:
                # Set starting player based on ai_goes_first parameter
                current_player = self.ai_player if ai_goes_first else self.random
            
            while not game_over:
                move_count += 1
                
                if current_player == self.ai_player:
                    # Q-learning agent's turn
                    start_time = time.time()
                    move = self.get_q_move()
                    end_time = time.time()
                    decision_times.append(end_time - start_time)
                    
                    self.make_move(move, self.ai_player, self.board)
                else:
                    # Opponent's turn
                    if opponent_type == "random":
                        move = random.choice(self.get_possible_actions())
                    elif opponent_type == "strategic":
                        move = self.get_strategic_move(self.board, self.random)
                    elif opponent_type == "minimax":
                        # Minimax opponent
                        temp_ai = self.ai_player
                        self.ai_player = self.random
                        self.random = temp_ai
                        move = self.get_ai_move()
                        # Restore original settings
                        temp_ai = self.ai_player
                        self.ai_player = self.random
                        self.random = temp_ai
                    
                    self.make_move(move, self.random, self.board)
                
                # Check if game is over
                if self.check_winner(self.board) == self.ai_player:
                    game_over = True
                    wins += 1
                elif self.check_winner(self.board) == self.random:
                    game_over = True
                    losses += 1
                elif self.is_board_full(self.board):
                    game_over = True
                    draws += 1
                
                # Switch players
                current_player = self.ai_player if current_player == self.random else self.random
            
            moves_per_game.append(move_count)
            if (i+1) % 10 == 0:
                print(f"Completed {i+1} games")
        
        # Print results
        print("\nQ-learning vs. %s Results:" % opponent_type)
        print(f"Total games: {num_games}")
        print(f"Wins: {wins} ({wins/num_games*100:.2f}%)")
        print(f"Losses: {losses} ({losses/num_games*100:.2f}%)")
        print(f"Draws: {draws} ({draws/num_games*100:.2f}%)")
        print(f"Average moves per game: {sum(moves_per_game)/len(moves_per_game):.2f}")
        print(f"Average decision time: {sum(decision_times)/len(decision_times)*1000:.2f} ms")
        
        return wins, losses, draws

# Add comparison function
def compare_algorithms():
    print("\n=== ALGORITHM COMPARISON ===\n")
    
    # Test minimax without pruning
    print("Testing Minimax without pruning...")
    game_standard = TicTacToe()
    game_standard.minimax_stats = {
        'states_visited': 0,
        'total_games': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'unique_states': set(),
        'decision_times': [],
        'moves_per_game': []
    }
    
    # Play 10 games with standard minimax against strategic opponent
    game_standard.game_simulation(num_games=10, opponent_type="strategic")
    
    # Test minimax with alpha-beta pruning
    print("Testing Minimax with alpha-beta pruning...")
    game_ab = TicTacToe()
    game_ab.minimax_stats = {
        'states_visited': 0,
        'total_games': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'unique_states': set(),
        'decision_times': [],
        'moves_per_game': []
    }
    
    # Play 10 games with alpha-beta pruning against strategic opponent
    game_ab.game_simulation(num_games=10, opponent_type="strategic")
    
    # Test Q-learning
    print("Testing Q-learning...")
    q_agent = Q_learning()
    
    # Train the agent
    print("Training Q-learning agent against mixed minimax and random opponents...")
    q_agent.train(episodes=100000, opponent_type="mixed")
    
    # Test the trained agent against different opponents
    print("\n=== Testing Trained Q-Learning Agent ===\n")
    print("Testing against random opponent:")
    q_agent.play_against(opponent_type="random", num_games=10, ai_goes_first=True)
    
    print("\nTesting against strategic opponent:")
    q_agent.play_against(opponent_type="strategic", num_games=10, ai_goes_first=True)
    
    print("\nTesting against minimax opponent (with alpha-beta pruning):")
    q_agent.play_against(opponent_type="minimax", num_games=10, ai_goes_first=True)
    
    # Get Q-learning metrics
    q_table_size = len(q_agent.q_table)
    total_state_actions = sum(len(actions) for actions in q_agent.q_table.values())
    
    # Print comparison results
    print("\n=== COMPARISON RESULTS ===\n")
    
    print("Minimax (Standard) vs Strategic Opponent:")
    print(f"  Win rate: {game_standard.minimax_stats['wins']/game_standard.minimax_stats['total_games']*100:.2f}%")
    print(f"  Average states visited per game: {game_standard.minimax_stats['states_visited']/game_standard.minimax_stats['total_games']:.2f}")
    print(f"  Average decision time: {sum(game_standard.minimax_stats['decision_times'])/len(game_standard.minimax_stats['decision_times'])*1000:.2f} ms")
    print(f"  Average moves per game: {sum(game_standard.minimax_stats['moves_per_game'])/len(game_standard.minimax_stats['moves_per_game']):.2f}")
    
    print("\nMinimax (Alpha-Beta Pruning) vs Strategic Opponent:")
    print(f"  Win rate: {game_ab.minimax_stats['wins']/game_ab.minimax_stats['total_games']*100:.2f}%")
    print(f"  Average states visited per game: {game_ab.minimax_stats['states_visited']/game_ab.minimax_stats['total_games']:.2f}")
    print(f"  Average decision time: {sum(game_ab.minimax_stats['decision_times'])/len(game_ab.minimax_stats['decision_times'])*1000:.2f} ms")
    print(f"  Average moves per game: {sum(game_ab.minimax_stats['moves_per_game'])/len(game_ab.minimax_stats['moves_per_game']):.2f}")
    print(f"  Pruning efficiency: {(1 - game_ab.minimax_stats['states_visited']/max(1, game_standard.minimax_stats['states_visited']))*100:.2f}%")
    
    print("\nQ-Learning:")
    print(f"  Q-table size (unique states): {q_table_size}")
    print(f"  Total state-action pairs: {total_state_actions}")
    print(f"  Training time: {q_agent.training_time:.2f} seconds")
    if q_agent.q_decision_times:
        print(f"  Average decision time: {sum(q_agent.q_decision_times)/len(q_agent.q_decision_times)*1000:.2f} ms")
    print(f"  Average moves per game: {sum(q_agent.training_stats['moves_per_game'])/len(q_agent.training_stats['moves_per_game']):.2f}")
    total_games = q_agent.training_stats['wins'] + q_agent.training_stats['losses'] + q_agent.training_stats['draws']
    print(f"  Training win rate: {q_agent.training_stats['wins']/total_games*100:.2f}%")

# main method
if __name__ == "__main__":
    # Run standard minimax for 100 games against strategic opponent
    # print("\n=== Running Standard Minimax (without pruning) vs Strategic Opponent ===\n")
    # game_standard = TicTacToe()
    # # Use original minimax
    # original_get_ai_move = game_standard.get_ai_move
    # game_standard.get_ai_move = game_standard.get_ai_move_original
    # game_standard.game_simulation(num_games=50, opponent_type="strategic", ai_goes_first=True)

    # # Run standard minimax for 100 games against strategic opponent
    # print("\n=== Running Standard Minimax (without pruning) vs Strategic Opponent ===\n")
    # game_standard = TicTacToe()
    # # Use original minimax
    # original_get_ai_move = game_standard.get_ai_move
    # game_standard.get_ai_move = game_standard.get_ai_move_original
    # game_standard.game_simulation(num_games=50, opponent_type="strategic", ai_goes_first=False)
     

  #  Run minimax with alpha-beta pruning for 100 games vs strategic (AI first)
    # print("\n=== Running Minimax with Alpha-Beta Pruning vs Strategic Opponent (AI first) ===\n")
    # game_ab = TicTacToe()
    # game_ab.game_simulation(num_games=5, opponent_type="random", ai_goes_first=True)
    
    # # Run minimax with alpha-beta pruning for 100 games vs strategic (AI second)
    # print("\n=== Running Minimax with Alpha-Beta Pruning vs Strategic Opponent (AI second) ===\n")
    # game_ab = TicTacToe()
    # game_ab.game_simulation(num_games=5, opponent_type="strategic", ai_goes_first=False)
    
    
    # Uncomment to train and run Q-learning
    q_agent = Q_learning()
    print("Training Q-learning agent against mixed minimax and random opponents...")
    q_agent.train(episodes=5000000, opponent_type="strategic")
    
    # # Test the trained agent against different opponents
    # print("\n=== Testing Trained Q-Learning Agent ===\n")
    # print("Testing against random opponent:")
    q_agent.play_against(opponent_type="random", num_games=100, ai_goes_first=True)
    
    # print("\nTesting against strategic opponent:")
    q_agent.play_against(opponent_type="strategic", num_games=100, ai_goes_first=True)
    
    print("\nTesting against minimax opponent (with alpha-beta pruning):")
    q_agent.play_against(opponent_type="minimax", num_games=100, ai_goes_first=True)
    
    
    # Uncomment to play a demo game
    # game = TicTacToe()
    # game.play_demo_game(opponent_type="strategic")
