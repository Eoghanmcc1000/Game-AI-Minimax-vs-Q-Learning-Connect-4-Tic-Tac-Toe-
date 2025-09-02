# Game AI: Minimax vs Q-Learning (Connect 4 & Tic-Tac-Toe)

This project implements **classical search algorithms** (Minimax with and without Alpha-Beta pruning) and **Reinforcement Learning** (Q-Learning) for two games: **Connect 4** and **Tic-Tac-Toe**.  
The implementations are benchmarked on **win rates, computational efficiency, and scalability**.

---

## 📂 Project Structure
├── Connect4/
│ ├── QLearning.py # Minimax & Q-Learning implementations + simulations
│ ├── board.py # Connect4 board logic
│ ├── Coin.py # Game piece definitions
│ ├── Constant.py # Game constants
│ ├── exceptions.py # Custom exceptions
│
├── TicTacToe/
│ ├── test.py # Minimax and Q-Learning for Tic-Tac-Toe
│
├── results/ # Experimental results & plots
├── README.md # You are here

---

## ⚙️ Requirements

- Python **3.9+**
- Standard libraries only (`random`, `pickle`)

---

## ▶️ Running the Code

### Connect 4

**Train a Q-Learning agent**
```bash
python Connect4/QLearning.py
(uncomment the training block inside QLearning.py)
Test a trained agent

python Connect4/QLearning.py
(uncomment the testing block with test_saved_qtable and ensure .pkl file exists)

Run Minimax simulations
python Connect4/QLearning.py
(uncomment either the run_minimax_simulations block for with/without pruning)

### Tic-Tac-Toe
Run minimax and Q-learning experiments
python TicTacToe/test.py
(enable/disable scenarios inside main by commenting/uncommenting the relevant blocks)

📊 Results Summary
Tic-Tac-Toe
Minimax (optimal): ~88% win rate as first player
Alpha-Beta Pruning: >95% reduction in states explored with same win rate
Q-Learning: ~90% vs Random, but drops sharply vs strategic opponents
Head-to-Head: Minimax always outperforms Q-Learning
Connect 4
Minimax: ~15.78s per move (depth-limited), ~80% win vs SmartRandom
Alpha-Beta Pruning: 97.5% faster (0.39s per move), strong performance
Q-Learning (500k episodes):
90% win vs Random
80% win vs SmartRandom
70% win vs Minimax AB
Real-time play: ~0.0005s per move
