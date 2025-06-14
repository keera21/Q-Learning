from enviroment import GridWorld
from q_learning import load_q_table, test_agent

def main():
    env = GridWorld()
    q_table = load_q_table()
    test_agent(env, q_table, episodes=5)

if __name__ == "__main__":
    main()
