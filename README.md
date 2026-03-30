# DQN Assignment 3

## Student Information

**Name:** Lesley Wanjiku Kamamo

**Student ID:** 8984971

---

## Assignment Overview

This assignment focuses on implementing a **Deep Q-Network (DQN)** to solve the Pong environment using reinforcement learning. The objective was to train an agent using image-based inputs (stacked frames) and evaluate its performance under different hyperparameter configurations.

The implementation includes:

* A Convolutional Neural Network (CNN) for approximating Q-values
* Experience Replay Buffer for stable learning
* Target Network for reducing instability
* Epsilon-greedy exploration strategy

Two key experiments were conducted:

1. **Batch Size Comparison** (8 vs 16)
2. **Target Network Update Frequency** (3 vs 10 episodes)

Performance was evaluated using:

* Score per episode
* Average reward over the last 5 episodes

Results showed that:

* Batch size 16 provided smoother learning and better performance
* Target update every 3 episodes enabled slightly faster learning
* Overall performance was limited due to short training duration

---

## ⚙️ Requirements

Install dependencies before running:

```bash
pip install gym[atari]
pip install ale-py
pip install numpy pandas matplotlib torch
```

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone <https://github.com/Lesley-w19/DQN_Assignment3.git>
cd DQN_Assignment3
```

2. Open the notebook:

```bash
jupyter notebook DQNAssignment3.ipynb
```

3. Run all cells in order:

* This will:

  * Train the DQN agent
  * Run experiments (batch size & target update)
  * Generate plots
  * Save results as CSV files

4. Run the DQAssign3.py file to simulate pong environment

```bash
python DQAssign3.py
```

---

## Output

The following outputs are generated:

* Training plots (Score & Average Reward)
* CSV files with experiment results
* Summary table comparing configurations

---

## Future Improvements

* Increase training episodes for better convergence
* Implement Double DQN to reduce overestimation bias
* Improve exploration strategies (e.g., epsilon scheduling, softmax)

---

## Notes

* Training may take significant time depending on hardware
* GPU is recommended but not required
* Results may vary slightly due to randomness
