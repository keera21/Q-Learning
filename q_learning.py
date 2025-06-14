import pickle
import os
import numpy as np
import random

class QLearningAgent:
    def __init__(self, grid_size, actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.98, min_epsilon=0.01):
        self.grid_size = grid_size
        self.actions = actions
        self.q_table = np.zeros((grid_size, grid_size, actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_action(self, state):
        x, y = state
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.actions - 1)
        return np.argmax(self.q_table[x, y])

    def update(self, state, action, reward, next_state, done):
        x, y = state
        nx, ny = next_state
        best_next = np.max(self.q_table[nx, ny])
        target = reward + (0 if done else self.gamma * best_next)
        self.q_table[x, y, action] = (1 - self.alpha) * self.q_table[x, y, action] + self.alpha * target

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def save_q_table(q_table, filename='q_table.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)
    print("Q-table saved to file.")

def load_q_table(filename='q_table.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            q_table = pickle.load(f)
        print("Q-table loaded from file.")
    else:
        q_table = np.zeros((10, 10, 4))
        print("No saved Q-table found. Starting fresh.")
    return q_table

def train_agent(env, episodes=300):
    agent = QLearningAgent(grid_size=10, actions=4)

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

    save_q_table(agent.q_table)
    return agent.q_table

def test_agent(env, q_table, episodes=5):
    agent = QLearningAgent(grid_size=10, actions=4)
    agent.q_table = q_table

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Q-Learning Test")
    clock = pygame.time.Clock()

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = np.argmax(agent.q_table[state[0], state[1]])
            state, reward, done = env.step(action)
            total_reward += reward

            env.render(screen)
            clock.tick(10)

        print(f"Test Episode {ep + 1}: Reward = {total_reward}")

    pygame.quit()
