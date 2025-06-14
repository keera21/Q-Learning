import pygame
import time
from enviroment import GridWorld
from q_learning import QLearningAgent

# Constants
EPISODES = 300
ACTIONS = 4
FPS = 100  # Slower so you can see movement clearly

def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Self-Driving Car with Q-Learning")
    clock = pygame.time.Clock()

    env = GridWorld()
    agent = QLearningAgent(grid_size=10, actions=ACTIONS)

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            env.render(screen)
            clock.tick(FPS)

        print(f"Episode {episode+1:3d}: Total Reward = {total_reward:4.0f}, Epsilon = {agent.epsilon:.2f}")

        # Watch it behave like a pro in last 50 episodes
        if episode > 250:
            time.sleep(0.1)

    pygame.quit()

if __name__ == "__main__":
    main()



from qlearning import train_agent
from enviroment import Env  # Or however your environment is named

def main():
    env = Env()
    Q = train_agent(env, episodes=300)

if __name__ == "__main__":
    main()


from qlearning import load_q_table, test_agent
from enviroment import Env

def main():
    env = Env()
    
    Q = load_q_table()  # Load trained Q-table
    
    test_agent(env, Q, episodes=5)  # Run to see how agent performs

if __name__ == "__main__":
    main()


