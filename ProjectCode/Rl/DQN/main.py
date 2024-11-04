
from environment import TradingEnv
from agent import DQNAgent
from data_loader import load_data
import config
import torch
import matplotlib.pyplot as plt
import pickle

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

    total_rewards = []  # List to store total rewards per episode

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

        total_rewards.append(total_reward)  # Store the total reward

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")


    # Save the total rewards for plotting
    with open('dqn_total_rewards.pkl', 'wb') as f:
        pickle.dump(total_rewards, f)

    # Save the trained model
    torch.save(agent.model.state_dict(), 'dqn_trading_model.pth')

    # Plot total rewards per episode
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Progress: Total Rewards per Episode')
    plt.savefig('dqn_training_rewards.png')
    plt.show()

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

    # Map actions to labels
    action_mapping = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    actions = test_env.actions_memory # retrieves actions taken during evaluation

    # Calculate performance metrics
    cumulative_return = (test_env.total_asset - test_env.initial_balance) / test_env.initial_balance
    print(f'\nEvaluation Results:')
    print(f'Cumulative Return: {cumulative_return * 100:.2f}%')

    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(test_env.portfolio_values, label='DQN Strategy')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.savefig('dqn_portfolio_value.png')
    plt.show()

    # plot actions over time
    plt.figure(figsize=(12, 3))
    plt.plot(actions, label='Actions', marker='o', linestyle='')
    plt.yticks([0, 1, 2], ['Hold', 'Buy', 'Sell'])
    plt.xlabel('Time Steps')
    plt.ylabel('Action')
    plt.title('Actions Taken Over Time')
    plt.savefig('dqn_actions.png')
    plt.show()


if __name__ == "__main__":
    train_agent()


