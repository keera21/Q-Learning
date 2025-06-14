from enviroment import GridWorld
from q_learning import train_agent

def main():
    env = GridWorld()
    train_agent(env, episodes=300)

if __name__ == "__main__":
    main()
