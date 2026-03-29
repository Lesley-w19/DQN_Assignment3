import gymnasium as gym
import numpy as np
import random
import collections
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from assignment3_utils import process_frame, transform_reward

# =========================
# HYPERPARAMETERS
# =========================
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

LR = 0.00025
MEMORY_SIZE = 50000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CNN MODEL
# =========================
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = x.float()
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =========================
# REPLAY BUFFER
# =========================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)

        return (
            np.concatenate(s),
            a,
            r,
            np.concatenate(ns),
            d
        )

    def __len__(self):
        return len(self.buffer)

# =========================
# AGENT
# =========================
class Agent:
    def __init__(self, action_size, batch_size, target_update):

        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon = EPSILON

        self.policy_net = DQN(action_size).to(DEVICE)
        self.target_net = DQN(action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.policy_net.fc[-1].out_features)

        state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            return torch.argmax(self.policy_net(state)).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        s, a, r, ns, d = self.memory.sample(self.batch_size)

        s = torch.tensor(s, dtype=torch.float32).to(DEVICE)
        ns = torch.tensor(ns, dtype=torch.float32).to(DEVICE)

        a = torch.tensor(a, dtype=torch.long).to(DEVICE)
        r = torch.tensor(r, dtype=torch.float32).to(DEVICE)
        d = torch.tensor(d, dtype=torch.float32).to(DEVICE)

        q = self.policy_net(s)
        next_q = self.target_net(ns)

        q_val = q.gather(1, a.unsqueeze(1)).squeeze(1)
        max_next_q = torch.max(next_q, dim=1)[0]

        target = r + GAMMA * max_next_q * (1 - d)

        loss = nn.MSELoss()(q_val, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# =========================
# FRAME STACKING
# =========================
def stack_frames(frames, frame, is_new):

    processed = process_frame(frame, (84, 80))
    processed = processed.squeeze(0)

    if is_new:
        frames = collections.deque([processed]*4, maxlen=4)
    else:
        frames.append(processed)

    stacked = np.stack(frames, axis=0)
    stacked = np.squeeze(stacked, axis=-1)

    return np.expand_dims(stacked, axis=0), frames

# =========================
# TRAIN FUNCTION
# =========================
def train_dqn(batch_size, target_update, episodes=500):

    env = gym.make("PongDeterministic-v4")

    action_size = env.action_space.n
    agent = Agent(action_size, batch_size, target_update)

    scores = []
    avg_scores = []
    steps_per_episode = []

    for ep in range(episodes):

        state, _ = env.reset()
        frames = collections.deque(maxlen=4)
        state, frames = stack_frames(frames, state, True)

        done = False
        total_reward = 0
        steps = 0

        while not done:

            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward = transform_reward(reward)

            next_state, frames = stack_frames(frames, next_state, False)

            agent.memory.push(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward
            steps += 1

        scores.append(total_reward)
        avg_scores.append(np.mean(scores[-5:]))
        steps_per_episode.append(steps)

        # Epsilon decay
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        # Target update
        if ep % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(f"[Batch {batch_size} | Update {target_update}] Ep {ep+1} | Score: {total_reward} | Avg5: {avg_scores[-1]:.2f}")

    env.close()
    return scores, avg_scores, steps_per_episode

# =========================
# PLOTTING FUNCTION
# =========================
def plot_experiment(results, title):

    plt.figure(figsize=(12,5))

    # Score per episode
    plt.subplot(1,2,1)
    for label, (scores, _, _) in results.items():
        plt.plot(scores, label=label)
    plt.title("Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()

    # Avg last 5
    plt.subplot(1,2,2)
    for label, (_, avg, _) in results.items():
        plt.plot(avg, label=label)
    plt.title("Avg Reward (Last 5)")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.legend()

    plt.suptitle(title)
    plt.show()

# =========================
# MAIN EXPERIMENTS
# =========================
def main():

    # =====================
    # EXPERIMENT 1: Batch Size
    # =====================
    print("\n=== Experiment: Batch Size ===")

    results_batch = {}

    results_batch["Batch 8"] = train_dqn(8, 10)
    results_batch["Batch 16"] = train_dqn(16, 10)

    plot_experiment(results_batch, "Batch Size Comparison")

    # =====================
    # EXPERIMENT 2: Target Update
    # =====================
    print("\n=== Experiment: Target Update ===")

    results_target = {}

    results_target["Update 10"] = train_dqn(8, 10)
    results_target["Update 3"] = train_dqn(8, 3)

    plot_experiment(results_target, "Target Network Update Comparison")


if __name__ == "__main__":
    main()