
from environment import TradingEnv
from agent import DQNAgent
from data_loader import load_data
import config
import torch

def train_agent():
    # Load and preprocess data
    data = load_data(config.DATA_PATH)

    # Split data into training and testing sets
    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)
    training_data = data.iloc[:split_index]
    testing_data = data.iloc[split_index:]

    # Create environment
    env = TradingEnv(training_data, initial_balance=config.INITIAL_BALANCE)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, config.AGENT_PARAMS)

    episodes = config.NUM_EPISODES

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
            agent.step_count += 1

            if agent.step_count % agent.update_target_every == 0:
                agent.update_target_model()

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Save the trained model
    torch.save(agent.model.state_dict(), 'dqn_trading_model.pth')

    # Evaluate the agent
    evaluate_agent(agent, testing_data)

def evaluate_agent(agent, testing_data):
    # Create test environment
    test_env = TradingEnv(testing_data, initial_balance=config.INITIAL_BALANCE)
    state = test_env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state, evaluate=True)  # Disable exploration during evaluation
        next_state, reward, done, _ = test_env.step(action)
        state = next_state
        total_reward += reward

    # Calculate performance metrics
    cumulative_return = (test_env.total_asset - test_env.initial_balance) / test_env.initial_balance
    print(f'\nEvaluation Results:')
    print(f'Cumulative Return: {cumulative_return * 100:.2f}%')

    # You can add more evaluation metrics as discussed earlier

if __name__ == "__main__":
    train_agent()
