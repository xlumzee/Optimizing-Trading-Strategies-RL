from environment import TradingEnv
from reinforce_agent import REINFORCEAgent
from data_loader import load_data
import config
import torch

def train_reinforce_agent():
    data = load_data(config.DATA_PATH)

    # Split data 
    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)
    training_data = data.iloc[:split_index]
    testing_data = data.iloc[split_index:]

    # Env creation
    env = TradingEnv(training_data, initial_balance=config.INITIAL_BALANCE)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCEAgent(state_size, action_size, learning_rate=config.AGENT_PARAMS['learning_rate'])

    num_episodes = config.NUM_EPISODES

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)
            state = next_state
            total_reward += reward

        # Update policy after each episode
        agent.update_policy()

        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}")

    # Save the trained policy
    torch.save(agent.policy_network.state_dict(), 'reinforce_trading_model.pth')

    # Evaluate the agent
    evaluate_reinforce_agent(agent, testing_data)

def evaluate_reinforce_agent(agent, testing_data):
    # Create test environment
    test_env = TradingEnv(testing_data, initial_balance=config.INITIAL_BALANCE)
    state = test_env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = test_env.step(action)
        state = next_state
        total_reward += reward

    # performance metrics
    cumulative_return = (test_env.total_asset - test_env.initial_balance) / test_env.initial_balance
    print(f'\nEvaluation Results:')
    print(f'Cumulative Return: {cumulative_return * 100:.2f}%')

    # Plot the portfolio value over time
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,6))
    plt.plot(test_env.portfolio_values, label='REINFORCE Strategy')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.title('Portfolio Value Over Time')
    plt.show()

if __name__ == "__main__":
    train_reinforce_agent()
