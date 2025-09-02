# Q-Learning

import random
from Constant import *
from Coin import Coin
from board import Board
from exceptions import ColumnFullException

class GameLogic():

    """A class that handles win conditions and determines winner"""
    WIN_SEQUENCE_LENGTH = 4

    def __init__(self, board):
        """
        Initialize the GameLogic object with a reference to the game board
        """
        self.board = board
        (num_rows, num_columns) = self.board.get_dimensions()
        #print(num_rows, num_columns)
        self.board_rows = num_rows
        self.board_cols = num_columns
        self.winner_value = 0

    def check_game_over(self):
        """
        Check whether the game is over which can be because of a tie or one
        of two players have won
        """
        (last_visited_nodes, player_value) = self.board.get_last_filled_information()
        representation = self.board.get_representation()
        player_won = self.search_win(last_visited_nodes, representation)
        if player_won:
            self.winner_value = player_value

        return ( player_won or self.board.check_board_filled() )



    def search_win(self, last_visited_nodes, representation):
        """
        Determine whether one of the players have won
        """
        for indices in last_visited_nodes:
            current_node = representation[indices[0]][indices[1]]
            if ( current_node.top_left_score == GameLogic.WIN_SEQUENCE_LENGTH or
                 current_node.top_score == GameLogic.WIN_SEQUENCE_LENGTH or
                 current_node.top_right_score == GameLogic.WIN_SEQUENCE_LENGTH or
                 current_node.left_score == GameLogic.WIN_SEQUENCE_LENGTH or
                 current_node.right_score == GameLogic.WIN_SEQUENCE_LENGTH or
                 current_node.bottom_left_score == GameLogic.WIN_SEQUENCE_LENGTH or
                 current_node.bottom_score == GameLogic.WIN_SEQUENCE_LENGTH or
                 current_node.bottom_right_score == GameLogic.WIN_SEQUENCE_LENGTH ):
                return True

        return False

    def determine_winner_name(self):
        """
        Return the winner's name
        """
        if (self.winner_value == 1):
            return "BLUE"
        elif (self.winner_value == 2):
            return "RED"
        else:
            return "TIE"

    def get_winner(self):
        """
        Return the winner coin type value
        """
        return self.winner_value
    


class Player():
    """A class that represents a player in the game"""

    def __init__(self, coin_type):
        """
        Initialize a player with their coin type
        """
        self.coin_type = coin_type

    def complete_move(self):
        """
        A method to make a move and update any learning parameters if any
        """
        pass

    def get_coin_type(self):
        """
        Return the coin type of the player
        """
        return self.coin_type

    def set_coin_type(self, coin_type):
        """
        Set the coin type of a player
        """
        self.coin_type = coin_type


class HumanPlayer(Player):
    """A class that represents a human player in the game"""
    def __init__(self, coin_type):
        Player.__init__(self, coin_type)

class RandomPlayer(Player):
    """A class that represents a computer that selects random moves"""
    def __init__(self, coin_type):
        Player.__init__(self, coin_type)
        
    def choose_action(self, state, actions):
        return random.choice(actions)

class SmartRandomPlayer(Player):
    """A player that plays winning moves when available and blocks opponent wins"""
    
    def __init__(self, coin_type):
        Player.__init__(self, coin_type)
    
    def choose_action(self, state, actions):
        # Check for winning move
        for action in actions:
            if self.would_win(state, action, self.coin_type):
                return action
                
        # # Check for opponent winning move to block
        opponent_type = 1 if self.coin_type == 2 else 2
        for action in actions:
            if self.would_win(state, action, opponent_type):
                return action
        
        # If no winning/blocking moves, play random
        return random.choice(actions)
    
    def would_win(self, state, action, player_type):
        """Simulate a move and check if it would result in a win"""
        temp_board = Board(6, 7)
        temp_game_logic = GameLogic(temp_board)
        
        # Recreate current state
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] != 0:
                    temp_coin = Coin(state[i][j])
                    temp_coin.set_column(j)
                    try:
                        temp_board.insert_coin(temp_coin, None, temp_game_logic)
                    except ColumnFullException:
                        continue
        
        # Try the move
        test_coin = Coin(player_type)
        test_coin.set_column(action)
        
        try:
            game_over = temp_board.insert_coin(test_coin, None, temp_game_logic)
            if game_over:
                winner = temp_game_logic.get_winner()
                return winner == player_type
        except ColumnFullException:
            return False
        
        return False
    
class MinimaxPlayer(Player):
    """A class that represents a Connect4 player using standard minimax"""
    
    def __init__(self, coin_type):
        Player.__init__(self, coin_type)
        self.max_depth = 5  # Standard minimax can't go as deep
        self.opponent_type = 1 if coin_type == 2 else 2
        # Add metrics for minimax
        self.minimax_stats = {
            'states_visited': 0,
            'branches_explored': 0,
            'unique_states': set(),
            'decision_times': [],
            'moves_per_game': [],
            'current_game_moves': 0,
            'current_game_states': set()  # Track unique states for current game
        }
    
    def reset_stats(self):
        """Reset minimax statistics for a new move"""
        # Don't reset the whole stats dictionary, just the counters for this move
        self.minimax_stats['states_visited'] = 0
        self.minimax_stats['branches_explored'] = 0
    
    def start_new_game(self):
        """Start tracking a new game"""
        self.minimax_stats['current_game_moves'] = 0
        self.minimax_stats['current_game_states'] = set()
    
    def choose_action(self, state, actions):
        """Choose the best move using minimax"""
        import time
        start_time = time.time()
        
        self.reset_stats()
        
        if not actions:
            return None
            
        # Prefer center columns (better strategy in Connect4)
        center_col = 3  # Center column in a 7-column board
        ordered_actions = sorted(actions, key=lambda x: abs(x - center_col))
        
        best_score = float('-inf')
        best_move = ordered_actions[0]  # Default to first available move
        
        for col in ordered_actions:
            # Create a temporary board to simulate the move
            temp_board = Board(6, 7)
            temp_game_logic = GameLogic(temp_board)
            
            # Copy current state to temp board
            self.copy_state_to_board(state, temp_board)
            
            # Make the move
            coin = Coin(self.coin_type)
            coin.set_column(col)
            
            try:
                temp_board.insert_coin(coin, None, temp_game_logic)
                
                # Call minimax to evaluate this move
                score = self.minimax(temp_board, temp_game_logic, 0, False)
                
                if score > best_score:
                    best_score = score
                    best_move = col
                    
            except ColumnFullException:
                continue
        
        # Track decision time in milliseconds
        end_time = time.time()
        decision_time = (end_time - start_time) * 1000  # Convert to ms
        self.minimax_stats['decision_times'].append(decision_time)
        
        # Increment move counter for this game
        self.minimax_stats['current_game_moves'] += 1
                
        return best_move
    
    def copy_state_to_board(self, state, board):
        """Copy a state representation to a board"""
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] != 0:
                    coin = Coin(state[i][j])
                    coin.set_column(j)
                    try:
                        # Insert without checking for game over
                        board.insert_coin(coin, None, GameLogic(board))
                    except ColumnFullException:
                        continue
    
    def minimax(self, board, game_logic, depth, is_maximizing):
        """
        Standard minimax algorithm for Connect4
        
        Args:
            board: Current board state
            game_logic: Game logic object to check for terminal states
            depth: Current depth in search tree (starts at 0)
            is_maximizing: True if maximizing player, False if minimizing
            
        Returns:
            Best score for this position
        """
        # Count unique states visited
        state_tuple = board.get_state()
        
        # Add to all-time unique states
        if state_tuple not in self.minimax_stats['unique_states']:
            self.minimax_stats['unique_states'].add(state_tuple)
            self.minimax_stats['states_visited'] += 1
        
        # Add to current game unique states
        if state_tuple not in self.minimax_stats['current_game_states']:
            self.minimax_stats['current_game_states'].add(state_tuple)
        
        # Check if game is over
        game_over = game_logic.check_game_over()
        if game_over:
            winner = game_logic.get_winner()
            if winner == self.coin_type:
                return 100  # Win
            elif winner == self.opponent_type:
                return -100  # Loss
            else:
                return 50  # Draw (board full)
        
        # Check depth limit
        if depth >= self.max_depth:
            return self.evaluate_board(board)
        
        # Get available actions
        actions = board.get_available_actions()
        
        # Count branches at root level
        if depth == 0:
            self.minimax_stats['branches_explored'] += len(actions)
        
        if is_maximizing:
            best_score = float('-inf')
            for col in actions:
                # Create a copy of the board
                temp_board = Board(6, 7)
                temp_game_logic = GameLogic(temp_board)
                self.copy_state_to_board(board.get_state(), temp_board)
                
                # Make the move
                coin = Coin(self.coin_type)
                coin.set_column(col)
                
                try:
                    temp_board.insert_coin(coin, None, temp_game_logic)
                    # Recursive call to minimax
                    score = self.minimax(temp_board, temp_game_logic, depth + 1, False)
                    best_score = max(score, best_score)
                except ColumnFullException:
                    continue
                    
            return best_score
        
        else:  # Minimizing player (opponent)
            best_score = float('inf')
            for col in actions:
                # Create a copy of the board
                temp_board = Board(6, 7)
                temp_game_logic = GameLogic(temp_board)
                self.copy_state_to_board(board.get_state(), temp_board)
                
                # Make the opponent's move
                coin = Coin(self.opponent_type)
                coin.set_column(col)
                
                try:
                    temp_board.insert_coin(coin, None, temp_game_logic)
                    # Recursive call to minimax
                    score = self.minimax(temp_board, temp_game_logic, depth + 1, True)
                    best_score = min(score, best_score)
                except ColumnFullException:
                    continue
                    
            return best_score
    
    def evaluate_window(self, window, piece):
        """Evaluate a window of 4 slots"""
        score = 0
        empty_value = 0
        opponent_piece = self.opponent_type
        
        # Winning windows
        if window.count(piece) == 4:
            score += 100
        
        # Strong threat - 3 pieces and an empty slot
        elif window.count(piece) == 3 and window.count(empty_value) == 1:
            score += 5
        
        # Developing threat - 2 pieces and 2 empty slots
        elif window.count(piece) == 2 and window.count(empty_value) == 2:
            score += 2
            
        # Opponent threats - 3 opponent pieces and an empty slot
        if window.count(opponent_piece) == 3 and window.count(empty_value) == 1:
            score -= 60  # Higher priority to block opponent threats
            
        return score
    
    def evaluate_board(self, board):
        """Evaluate the current position using sliding windows"""
        score = 0
        state = board.get_state()
        rows = len(state)
        cols = len(state[0])
        
        # Score center column (strategic advantage)
        center_col = cols // 2
        center_array = [state[r][center_col] for r in range(rows)]
        center_count = center_array.count(self.coin_type)
        score += center_count * 3
        
        # Score horizontal windows
        for r in range(rows):
            for c in range(cols - 3):
                window = [state[r][c+i] for i in range(4)]
                score += self.evaluate_window(window, self.coin_type)
        
        # Score vertical windows
        for c in range(cols):
            for r in range(rows - 3):
                window = [state[r+i][c] for i in range(4)]
                score += self.evaluate_window(window, self.coin_type)
        
        # Score positive diagonal windows
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = [state[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, self.coin_type)
        
        # Score negative diagonal windows
        for r in range(3, rows):
            for c in range(cols - 3):
                window = [state[r-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, self.coin_type)
        
        return score

    def record_game_end(self):
        """Record stats at the end of a game"""
        if self.minimax_stats['current_game_moves'] > 0:
            self.minimax_stats['moves_per_game'].append(self.minimax_stats['current_game_moves'])
            # Also track unique states seen in this game
            unique_states_this_game = len(self.minimax_stats['current_game_states'])
            # Reset for next game
            self.start_new_game()
    
    def get_stats_summary(self):
        """Get a summary of minimax performance statistics"""
        total_states = len(self.minimax_stats['unique_states'])
        avg_states = total_states / max(1, len(self.minimax_stats['moves_per_game']))
        avg_time = sum(self.minimax_stats['decision_times']) / max(1, len(self.minimax_stats['decision_times']))
        avg_moves = sum(self.minimax_stats['moves_per_game']) / max(1, len(self.minimax_stats['moves_per_game']))
        
        return {
            'total_unique_states': total_states,
            'avg_states_per_game': avg_states,
            'avg_decision_time_ms': avg_time,
            'avg_moves_per_game': avg_moves,
            'total_decisions': len(self.minimax_stats['decision_times'])
        }

class MinimaxAlphaBetaPlayer(MinimaxPlayer):
    """A class that represents a Connect4 player using minimax with alpha-beta pruning"""
    
    def __init__(self, coin_type):
        MinimaxPlayer.__init__(self, coin_type)
        self.max_depth = 5  # Alpha-beta pruning can go deeper
    
    def choose_action(self, state, actions):
        """Choose the best move using minimax with alpha-beta pruning"""
        import time
        start_time = time.time()
        
        self.reset_stats()
        
        if not actions:
            return None
            
        # Prefer center columns (better strategy in Connect4)
        center_col = 3  # Center column in a 7-column board
        ordered_actions = sorted(actions, key=lambda x: abs(x - center_col))
        
        best_score = float('-inf')
        best_move = ordered_actions[0]  # Default to first available move
        
        # Alpha-beta pruning parameters
        alpha = float('-inf')
        beta = float('inf')
        
        for col in ordered_actions:
            # Create a temporary board to simulate the move
            temp_board = Board(6, 7)
            temp_game_logic = GameLogic(temp_board)
            
            # Copy current state to temp board
            self.copy_state_to_board(state, temp_board)
            
            # Make the move
            coin = Coin(self.coin_type)
            coin.set_column(col)
            
            try:
                temp_board.insert_coin(coin, None, temp_game_logic)
                
                # Call minimax with alpha-beta pruning to evaluate this move
                score = self.minimax_alpha_beta(temp_board, temp_game_logic, 0, False, alpha, beta)
                
                if score > best_score:
                    best_score = score
                    best_move = col
                
                # Update alpha for pruning
                alpha = max(alpha, best_score)
                    
            except ColumnFullException:
                continue
        
        # Track decision time in milliseconds
        end_time = time.time()
        decision_time = (end_time - start_time) * 1000  # Convert to ms
        self.minimax_stats['decision_times'].append(decision_time)
        
        # Increment move counter for this game
        self.minimax_stats['current_game_moves'] += 1
                
        return best_move
    
    def minimax_alpha_beta(self, board, game_logic, depth, is_maximizing, alpha, beta):
        """
        Minimax algorithm with alpha-beta pruning
        
        Args:
            board: Current board state
            game_logic: Game logic object to check for terminal states
            depth: Current depth in search tree
            is_maximizing: True if maximizing player, False if minimizing
            alpha: Best score for maximizing player
            beta: Best score for minimizing player
            
        Returns:
            Best score for this position
        """
        # Count unique states visited
        state_tuple = board.get_state()
        
        # Add to all-time unique states
        if state_tuple not in self.minimax_stats['unique_states']:
            self.minimax_stats['unique_states'].add(state_tuple)
            self.minimax_stats['states_visited'] += 1
        
        # Add to current game unique states
        if state_tuple not in self.minimax_stats['current_game_states']:
            self.minimax_stats['current_game_states'].add(state_tuple)
        
        # Check if game is over
        game_over = game_logic.check_game_over()
        if game_over:
            winner = game_logic.get_winner()
            if winner == self.coin_type:
                return 100  # Win
            elif winner == self.opponent_type:
                return -100  # Loss
            else:
                return 20  # Draw (board full)
        
        # Check depth limit
        if depth >= self.max_depth:
            return self.evaluate_board(board)
        
        # Get available actions
        actions = board.get_available_actions()
        
        # Count branches at root level
        if depth == 0:
            self.minimax_stats['branches_explored'] += len(actions)
        
        # Order columns with center-first for better pruning
        center_col = 3
        ordered_actions = sorted(actions, key=lambda x: abs(x - center_col))
        
        if is_maximizing:
            best_score = float('-inf')
            for col in ordered_actions:
                # Create a copy of the board
                temp_board = Board(6, 7)
                temp_game_logic = GameLogic(temp_board)
                self.copy_state_to_board(board.get_state(), temp_board)
                
                # Make the move
                coin = Coin(self.coin_type)
                coin.set_column(col)
                
                try:
                    temp_board.insert_coin(coin, None, temp_game_logic)
                    # Recursive call with alpha-beta pruning
                    score = self.minimax_alpha_beta(temp_board, temp_game_logic, depth + 1, False, alpha, beta)
                    best_score = max(score, best_score)
                    
                    # Update alpha
                    alpha = max(alpha, best_score)
                    
                    # Prune if possible
                    if beta <= alpha:
                        break
                        
                except ColumnFullException:
                    continue
                    
            return best_score
        
        else:  # Minimizing player (opponent)
            best_score = float('inf')
            for col in ordered_actions:
                # Create a copy of the board
                temp_board = Board(6, 7)
                temp_game_logic = GameLogic(temp_board)
                self.copy_state_to_board(board.get_state(), temp_board)
                
                # Make the opponent's move
                coin = Coin(self.opponent_type)
                coin.set_column(col)
                
                try:
                    temp_board.insert_coin(coin, None, temp_game_logic)
                    # Recursive call with alpha-beta pruning
                    score = self.minimax_alpha_beta(temp_board, temp_game_logic, depth + 1, True, alpha, beta)
                    best_score = min(score, best_score)
                    
                    # Update beta
                    beta = min(beta, best_score)
                    
                    # Prune if possible
                    if beta <= alpha:
                        break
                        
                except ColumnFullException:
                    continue
                    
            return best_score

class QLearningPlayer(Player):
    """A class that represents an AI using Q-learning algorithm"""

    def __init__(self, coin_type, epsilon=0.2, alpha=0.1, gamma=0.9):
        """
        Initialize a Q-learner with parameters epsilon, alpha and gamma
        and its coin type
        """
        Player.__init__(self, coin_type)
        self.q = {}
        self.epsilon = epsilon  # e-greedy chance of random exploration
        self.alpha = alpha     # learning rate
        self.gamma = gamma     # discount factor for future rewards
        self.initial_q_value = 1.0  # Optimistic initial value

    def getQ(self, state, action):
        """
        Return a probability for a given state and action where the greater
        the probability the better the move
        """
        # Convert state to tuple for dictionary key
        state = tuple(map(tuple, state))
        
        # Initialize state in Q-table if not present
        if state not in self.q:
            self.q[state] = {}
            
        # Initialize action in state's Q-values if not present
        if action not in self.q[state]:
            self.q[state][action] = 0.0
            
        return self.q[state][action]
    
    def choose_action(self, state, actions):
        """
        Return an action based on the best move recommendation by the current
        Q-Table with a epsilon chance of trying out a new move
        """
        # Explore: choose random action with probability epsilon
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)
        
        # Exploit: choose best action based on Q-table
        # Get Q-values for all possible actions
        q_values = {action: self.getQ(state, action) for action in actions}
        
        # Find maximum Q-value
        max_q = max(q_values.values()) if q_values else 0
        
        # Find all actions with the maximum Q-value
        best_actions = [action for action, q_value in q_values.items() if q_value == max_q]
        
        # If multiple best actions, choose randomly among them
        return random.choice(best_actions)

    def learn(self, board, actions, chosen_action, game_over, game_logic):
        """
        Determine the reward based on its current chosen action and update
        the Q table using the reward received and the maximum future reward
        based on the resulting state due to the chosen action
        """
        # Get reward based on game outcome
        reward = 0
        if game_over:
            winner = game_logic.get_winner()
            if winner == self.coin_type:
                reward = 10.0  # Win
            elif winner == 0:
                reward = 5.0   # Draw (neutral)
            else:
                reward = -10.0  # Loss (symmetric with win)
        else:
            # Strategic rewards for non-terminal states
            representation = board.get_representation()
            opponent_type = 1 if self.coin_type == 2 else 2
            
            # Check for my threats
            for i in range(len(representation)):
                for j in range(len(representation[0])):
                    node = representation[i][j]
                    if node.value == self.coin_type:
                        # Reward for creating threats
                        if (node.top_score == 3 or node.right_score == 3 or 
                            node.bottom_score == 3 or node.left_score == 3 or
                            node.top_right_score == 3 or node.bottom_right_score == 3 or
                            node.bottom_left_score == 3 or node.top_left_score == 3):
                            reward = 3.0  # Creating 3-in-a-row (increased from 2.0)
                        elif (node.top_score == 2 or node.right_score == 2 or 
                              node.bottom_score == 2 or node.left_score == 2 or
                              node.top_right_score == 2 or node.bottom_right_score == 2 or
                              node.bottom_left_score == 2 or node.top_left_score == 2):
                            reward = 1.0  # Creating 2-in-a-row (increased from 0.5)

            # Check for opponent threats and penalize
            for i in range(len(representation)):
                for j in range(len(representation[0])):
                    node = representation[i][j]
                    if node.value == opponent_type:
                        # Penalize for allowing opponent threats
                        if (node.top_score == 3 or node.right_score == 3 or 
                            node.bottom_score == 3 or node.left_score == 3 or
                            node.top_right_score == 3 or node.bottom_right_score == 3 or
                            node.bottom_left_score == 3 or node.top_left_score == 3):
                            reward -= 4.0  # Opponent has 3-in-a-row - serious threat!
                        elif (node.top_score == 2 or node.right_score == 2 or 
                              node.bottom_score == 2 or node.left_score == 2 or
                              node.top_right_score == 2 or node.bottom_right_score == 2 or
                              node.bottom_left_score == 2 or node.top_left_score == 2):
                            reward -= 0.5  # Opponent has 2-in-a-row - potential threat

        prev_state = board.get_prev_state()
        current_state = board.get_state()
        
        # Get current Q-value
        current_q = self.getQ(prev_state, chosen_action)
        
        # Get max future Q-value
        if game_over:
            max_future_q = 0
        else:
            future_q_values = [self.getQ(current_state, action) for action in actions]
            max_future_q = max(future_q_values) if future_q_values else 0
        
        # Update Q-value using Q-learning equation
        # Q(s,a) = (1-α)Q(s,a) + α[R + γ max Q(s',a')]
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        
        # Store updated Q-value
        prev_state = tuple(map(tuple, prev_state))
        if prev_state not in self.q:
            self.q[prev_state] = {}
        self.q[prev_state][chosen_action] = new_q


    
def train_q_learning(num_episodes=50000, opponent_type='smart'):
    """Train Q-learning agent with improved parameters and opponent"""
    import time
    start_time = time.time()
    
    # Initialize with optimized parameters
    initial_epsilon = 0.8  # Reduced from 0.5 for more focused exploration
    final_epsilon = 0.0001  # Keep this low for final exploitation
    epsilon_decay = (final_epsilon / initial_epsilon) ** (1.0 / num_episodes)  # Standard decay
    # Q player as player 1 (goes first)
    q_player = QLearningPlayer(coin_type=1, epsilon=initial_epsilon, alpha=0.3, gamma=0.85)  # Adjusted learning parameters

    # Create both opponent types
    if opponent_type == 'smart':
        regular_opponent = SmartRandomPlayer(coin_type=2)
    else:  # Default to random if not smart
        regular_opponent = RandomPlayer(coin_type=2)
        
    # Create minimax opponent with reduced depth for faster training
    minimax_opponent = MinimaxPlayer(coin_type=2)
    minimax_opponent.max_depth = 4  # Reduced depth for faster training
    
    stats = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'win_rates': [],
        'minimax_wins': 0,
        'minimax_losses': 0,
        'regular_wins': 0,
        'regular_losses': 0,
        'total_rewards': 0,
        'recent_rewards': [],
        'recent_wins': [],  # Track recent games (last 100)
        'recent_games': 0,
        'elapsed_time': 0
    }
    
    for episode in range(num_episodes):
        # Alternate between regular opponent and minimax
        # if episode % 2 == 0:
        #     opponent = minimax_opponent
        #     opponent_type_current = "minimax"
        # else:
        opponent = regular_opponent
        opponent_type_current = opponent_type
            
        board = Board(6, 7)
        game_logic = GameLogic(board)
        game_over = False
        episode_reward = 0
        
        while not game_over:
            # Q-player move first
            actions = board.get_available_actions()
            if not actions:
                stats['draws'] += 1
                game_over = True
                break
                
            # Get current state before making move
            current_state = board.get_state()
            
            # Choose and make move
            chosen_action = q_player.choose_action(current_state, actions)
            coin = Coin(q_player.coin_type)
            coin.set_column(chosen_action)
            
            try:
                # Make move and get reward
                game_over = board.insert_coin(coin, None, game_logic)
                
                # Learn from this move and track reward
                old_q = q_player.getQ(current_state, chosen_action)
                q_player.learn(board, actions, chosen_action, game_over, game_logic)
                new_q = q_player.getQ(current_state, chosen_action)
                # Approximate reward from Q-value change
                reward = (new_q - old_q) / q_player.alpha
                episode_reward += reward
                
                if game_over:
                    winner = game_logic.get_winner()
                    if winner == q_player.coin_type:
                        stats['wins'] += 1
                        stats['recent_wins'].append(1)
                        if opponent_type_current == "minimax":
                            stats['minimax_wins'] += 1
                        else:
                            stats['regular_wins'] += 1
                    elif winner == 0:
                        stats['draws'] += 1
                        stats['recent_wins'].append(0)
                    else:
                        stats['losses'] += 1
                        stats['recent_wins'].append(0)
                        if opponent_type_current == "minimax":
                            stats['minimax_losses'] += 1
                        else:
                            stats['regular_losses'] += 1
                    break

                # Opponent move second
                actions = board.get_available_actions()
                if not actions:
                    stats['draws'] += 1
                    stats['recent_wins'].append(0)
                    game_over = True
                    
                    # Learn from this move after finding out board is full
                    #q_player.learn(board, actions, chosen_action, game_over, game_logic)
                    break
                    
                opponent_action = opponent.choose_action(board.get_state(), actions)
                opponent_coin = Coin(opponent.coin_type)
                opponent_coin.set_column(opponent_action)
                
                # Make opponent's move
                game_over = board.insert_coin(opponent_coin, None, game_logic)
                
                # NOW learn from Q-player's move AFTER seeing opponent's response
                # This is the key change - learning happens here instead of before opponent's move
               # q_player.learn(board, board.get_available_actions(), chosen_action, game_over, game_logic)
                
                if game_over:
                    winner = game_logic.get_winner()
                    if winner == q_player.coin_type:
                        stats['wins'] += 1
                        stats['recent_wins'].append(1)
                        if opponent_type_current == "minimax":
                            stats['minimax_wins'] += 1
                        else:
                            stats['regular_wins'] += 1
                    elif winner == 0:
                        stats['draws'] += 1
                        stats['recent_wins'].append(0)
                    else:
                        stats['losses'] += 1
                        stats['recent_wins'].append(0)
                        if opponent_type_current == "minimax":
                            stats['minimax_losses'] += 1
                        else:
                            stats['regular_losses'] += 1
                
            except ColumnFullException:
                continue
        
        # Keep only recent game results (last 100)
        stats['recent_wins'] = stats['recent_wins'][-100:]
        stats['recent_rewards'].append(episode_reward)
        stats['recent_rewards'] = stats['recent_rewards'][-100:]
        stats['total_rewards'] += episode_reward
        stats['recent_games'] = min(stats['recent_games'] + 1, 100)
        
        # Decay epsilon after each episode
        q_player.epsilon = max(final_epsilon, q_player.epsilon * epsilon_decay)
        
        # Track win rate
        current_win_rate = stats['wins'] / (episode + 1) * 100
        stats['win_rates'].append(current_win_rate)
        
        # Only print stats every 10 episodes to reduce output volume
        if episode % 10 == 0:
            # Calculate recent win rate
            recent_win_rate = sum(stats['recent_wins']) / len(stats['recent_wins']) * 100 if stats['recent_wins'] else 0
            
            # Calculate average reward
            avg_reward = sum(stats['recent_rewards']) / len(stats['recent_rewards']) if stats['recent_rewards'] else 0
            
            # Calculate elapsed time
            current_time = time.time()
            elapsed = current_time - start_time
            stats['elapsed_time'] = elapsed
            
            # Q-table size
            num_states = len(q_player.q)
            total_state_actions = sum(len(actions) for actions in q_player.q.values())
            
            print(f"\nEpisode {episode}/{num_episodes} (Time: {elapsed:.1f}s)")
            print(f"Wins: {stats['wins']} ({current_win_rate:.2f}%)")
            print(f"Recent Win Rate: {recent_win_rate:.2f}%")
            print(f"Avg Reward: {avg_reward:.2f}")
            print(f"Q-table: {num_states} states, {total_state_actions} state-actions")
            print(f"Epsilon: {q_player.epsilon:.4f}")
            
    # Record final time
    stats['elapsed_time'] = time.time() - start_time
    
    return q_player, stats

def test_game_logic(player1=None, player2=None, show_every_move=True):
    """
    Run a Connect 4 game without animations to test the game logic.
    
    Args:
        player1: First player (defaults to RandomPlayer if None)
        player2: Second player (defaults to SmartRandomPlayer if None)
        show_every_move: Whether to print the board after every move
        
    Returns:
        Winner of the game (0 for draw, 1 for player1, 2 for player2)
    """
    # Set up default players if not provided
    if player1 is None:
        player1 = RandomPlayer(coin_type=1)
    
    if player2 is None:
        player2 = MinimaxAlphaBetaPlayer(coin_type=2)
    
    # Initialize game
    board = Board(6, 7)
    game_logic = GameLogic(board)
    game_over = False
    current_player = player1
    move_count = 0
    
    # Print initial empty board
    print("\nStarting game...")
    print_board_state(board.get_state())
    
    # Main game loop
    while not game_over:
        move_count += 1
        print(f"\nMove {move_count}, {get_player_name(current_player.coin_type)}'s turn")
        
        # Get available actions
        actions = board.get_available_actions()
        if not actions:
            print("Board is full - game is a draw")
            return 0
        
        # Player makes a move
        chosen_action = current_player.choose_action(board.get_state(), actions)
        coin = Coin(current_player.coin_type)
        coin.set_column(chosen_action)
        
        print(f"{get_player_name(current_player.coin_type)} places in column {chosen_action}")
        
        try:
            # Make move
            game_over = board.insert_coin(coin, None, game_logic)
            
            # Print board after move
            if show_every_move:
                print_board_state(board.get_state())
            
            # Check if game is over
            if game_over:
                winner = game_logic.get_winner()
                if winner == 0:
                    print("\nGame ended in a draw")
                else:
                    print(f"\n{get_player_name(winner)} wins in {move_count} moves!")
                
                # Always show final board
                if not show_every_move:
                    print_board_state(board.get_state())
                
                return winner
            
            # Switch players
            current_player = player2 if current_player == player1 else player1
            
        except ColumnFullException:
            print(f"Column {chosen_action} is full, trying again")
            continue
    
    return 0

def print_board_state(state):
    """Print a text representation of the board"""
    print("  0 1 2 3 4 5 6")  # Column numbers
    print(" +-+-+-+-+-+-+-+")
    
    for row in range(len(state)):
        print(f"{row}|", end="")
        for col in range(len(state[0])):
            if state[row][col] == 0:
                print(" |", end="")
            elif state[row][col] == 1:
                print("B|", end="")  # Blue
            elif state[row][col] == 2:
                print("R|", end="")  # Red
        print()
    
    print(" +-+-+-+-+-+-+-+")

def get_player_name(coin_type):
    """Convert coin_type to player name"""
    if coin_type == 1:
        return "BLUE"
    elif coin_type == 2:
        return "RED"
    return "NONE"

def simulate(player1=None, player2=None, games=100):
    """Simulate games between two players and track results"""
    # Set up default players if not provided
    if player1 is None:
        player1 = RandomPlayer(coin_type=1)
    
    if player2 is None:
        player2 = MinimaxAlphaBetaPlayer(coin_type=2)

    # Track results
    stats = {
        'player1_wins': 0,
        'player2_wins': 0,
        'draws': 0
    }
    
    # Initialize new games for players with statistics
    if hasattr(player1, 'start_new_game'):
        player1.start_new_game()
    if hasattr(player2, 'start_new_game'):
        player2.start_new_game()

    for game in range(games):
        if game % 1 == 0:
            print(f"Simulating game {game+1}/{games}")
            
        board = Board(6, 7)
        game_logic = GameLogic(board)
        game_over = False
        
        # Reset move counters for this game
        move_count = 0
        
        # Start with player1 (minimax)
        current_player = player1
        
        # Clear game-specific stats for new game
        if hasattr(player1, 'start_new_game'):
            player1.start_new_game()
        if hasattr(player2, 'start_new_game'):
            player2.start_new_game()

        while not game_over:
            # Get available actions
            actions = board.get_available_actions()
            if not actions:
                stats['draws'] += 1
                game_over = True
                break

            # Player makes a move
            chosen_action = current_player.choose_action(board.get_state(), actions)
            coin = Coin(current_player.coin_type)
            coin.set_column(chosen_action)
            
            # Count the move
            move_count += 1
            
            try:
                # Make move
                game_over = board.insert_coin(coin, None, game_logic)
                
                # Check if game is over
                if game_over:
                    winner = game_logic.get_winner()
                    if winner == 0:
                        stats['draws'] += 1
                    elif winner == player1.coin_type:
                        stats['player1_wins'] += 1
                    else:
                        stats['player2_wins'] += 1
                        
                    # Record game ending for stats if players support it
                    if hasattr(player1, 'record_game_end'):
                        player1.record_game_end()
                    if hasattr(player2, 'record_game_end'):
                        player2.record_game_end()
                    break
                
                # Switch players
                current_player = player2 if current_player == player1 else player1
                
            except ColumnFullException:
                continue
    
    # Print results
    print("\nSimulation Results:")
    print(f"Total Games: {games}")
    print(f"{player1.__class__.__name__} (Player 1) Wins: {stats['player1_wins']} ({stats['player1_wins']/games*100:.2f}%)")
    print(f"{player2.__class__.__name__} (Player 2) Wins: {stats['player2_wins']} ({stats['player2_wins']/games*100:.2f}%)")
    print(f"Draws: {stats['draws']} ({stats['draws']/games*100:.2f}%)")
    
    return stats

# Add this new function to run minimax against different opponents
def run_minimax_simulations(use_pruning=True, opponent_type='smart', num_games=50):
    """
    Run minimax simulations against specified opponent
    
    Args:
        use_pruning: Whether to use alpha-beta pruning
        opponent_type: Type of opponent ('smart', 'random', or 'minimax')
        num_games: Number of games to simulate
        
    Returns:
        stats dictionary with results
    """
    import time
    overall_start = time.time()
    
    # Create the minimax player with or without pruning
    if use_pruning:
        minimax_player = MinimaxAlphaBetaPlayer(coin_type=1)
        player_name = "MinimaxAlphaBeta"
    else:
        minimax_player = MinimaxPlayer(coin_type=1)
        player_name = "Minimax"
    
    # Create the opponent
    if opponent_type == 'smart':
        opponent = SmartRandomPlayer(coin_type=2)
        opponent_name = "SmartRandom"
    elif opponent_type == 'minimax':
        opponent = MinimaxAlphaBetaPlayer(coin_type=2)
        opponent_name = "MinimaxAlphaBeta"
    else:
        opponent = RandomPlayer(coin_type=2)
        opponent_name = "Random"
    
    # Run simulations
    stats = simulate(minimax_player, opponent, games=num_games)
    
    # Get minimax performance stats
    minimax_stats = minimax_player.get_stats_summary()
    
    # Calculate overall time
    overall_time = time.time() - overall_start
    
    # Print detailed results
    print(f"\n===== {player_name} vs {opponent_name} ({num_games} games) =====")
    print(f"Total time: {overall_time:.2f} seconds")
    print(f"Win rate: {stats['player1_wins']/num_games*100:.2f}%")
    print(f"Total unique states visited: {minimax_stats['total_unique_states']}")
    print(f"Average unique states per game: {minimax_stats['avg_states_per_game']:.2f}")
    print(f"Average decision time: {minimax_stats['avg_decision_time_ms']:.2f} ms")
    print(f"Average moves per game: {minimax_stats['avg_moves_per_game']:.2f}")
    
    return {
        'game_stats': stats,
        'minimax_stats': minimax_stats,
        'time': overall_time
    }

def test_saved_qtable(qtable_file="q_table_radom.pkl", num_games=10):
    """
    Test a saved Q-table against different opponents
    
    Args:
        qtable_file: Path to the saved Q-table pickle file
        num_games: Number of games to simulate against each opponent
    """
    import pickle
    
    # Load the Q-table
    try:
        with open(qtable_file, 'rb') as f:
            q_table = pickle.load(f)
            print(f"Loaded Q-table with {len(q_table)} states")
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return
    
    # Create a QLearningPlayer with the loaded Q-table
    q_player = QLearningPlayer(coin_type=1, epsilon=0.01)  # Low epsilon for some exploration
    q_player.q = q_table
    
    # Test against RandomPlayer
    random_opponent = RandomPlayer(coin_type=2)
    print(f"\nTesting Q-Learning vs RandomPlayer ({num_games} games)...")
    random_results = simulate(q_player, random_opponent, games=num_games)
    win_rate_random = random_results['player1_wins'] / num_games * 100
    
    # Test against SmartRandomPlayer
    smart_opponent = SmartRandomPlayer(coin_type=2)
    print(f"\nTesting Q-Learning vs SmartRandomPlayer ({num_games} games)...")
    smart_results = simulate(q_player, smart_opponent, games=num_games)
    win_rate_smart = smart_results['player1_wins'] / num_games * 100
    
    # Test against MinimaxAlphaBetaPlayer
    minimax_opponent = MinimaxAlphaBetaPlayer(coin_type=2)
    print(f"\nTesting Q-Learning vs MinimaxAlphaBetaPlayer ({num_games} games)...")
    minimax_results = simulate(q_player, minimax_opponent, games=num_games)
    win_rate_minimax = minimax_results['player1_wins'] / num_games * 100
    
    # Print summary
    print("\n===== Q-LEARNING PERFORMANCE SUMMARY =====")
    print(f"Q-table size: {len(q_player.q)} states")
    print(f"Win rate vs Random: {win_rate_random:.2f}%")
    print(f"Win rate vs SmartRandom: {win_rate_smart:.2f}%")
    print(f"Win rate vs MinimaxAlphaBeta: {win_rate_minimax:.2f}%")
    
    return {
        'random_results': random_results,
        'smart_results': smart_results,
        'minimax_results': minimax_results
    }

if __name__ == "__main__":
    # # Run simulation between MinimaxAlphaBeta and SmartRandom
    # print("\nRunning MinimaxAlphaBeta vs Random simulation...")
    # minimax_player = MinimaxAlphaBetaPlayer(coin_type=1)
    # random_player = RandomPlayer(coin_type=2)
    # num_games = 10

    # stats = simulate(minimax_player, random_player, games=num_games)
    # # Print detailed breakdown
    # print("\nDetailed Results:")
    # print(f"MinimaxAlphaBeta wins: {stats['player1_wins']} ({stats['player1_wins']/num_games*100:.2f}%)")
    # print(f"SmartRandom wins: {stats['player2_wins']} ({stats['player2_wins']/num_games*100:.2f}%)")
    # print(f"Draws: {stats['draws']} ({stats['draws']/num_games*100:.2f}%)")
    
    # # First train against smart opponent
    # print("\nTraining against random opponent...")
    # trained_agent, training_stats = train_q_learning(num_episodes=500000, opponent_type='random') #, opponent_type='smart')
    # print("Training complete")

    # # Save the Q-table
    # import pickle
    # with open('q_table_random.pkl', 'wb') as f:
    #     pickle.dump(trained_agent.q, f)
    # print("Q-table saved as q_table_random.pkl")

    # Test the saved Q-table
    # test_saved_qtable("q_table_random.pkl", num_games=100)

    # # # simuale a game
    test_game_logic(player1=RandomPlayer(coin_type=1), player2=MinimaxAlphaBetaPlayer(coin_type=2), show_every_move=True)

    # Uncomment to run minimax simulations
    # print("\nRunning Minimax (with pruning) vs SmartRandom")
    # run_minimax_simulations(use_pruning=True, opponent_type='smart', num_games=10)
    
    # print("\nRunning Minimax (without pruning) vs SmartRandom")
    # run_minimax_simulations(use_pruning=False, opponent_type='smart', num_games=5)
    
