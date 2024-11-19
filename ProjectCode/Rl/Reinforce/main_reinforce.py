from environment import TradingEnv
from reinforce_agent import REINFORCEAgent
from data_loader import load_data
import config
import torch

import matplotlib.pyplot as plt
import pickle


def train_reinforce_agent(data):
    # throwing error while UI is generated
    #data = load_data(config.DATA_PATH)

    # Split data 
    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)
    training_data = data.iloc[:split_index]
    testing_data = data.iloc[split_index:]

    ######################## -- throwing error
    # # Env creation
    env = TradingEnv(training_data, initial_balance=config.INITIAL_BALANCE)
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n

    # Dynamically determine state and action sizes from the environment
    state_size = env.observation_space.shape[0]  # Number of features in the state
    action_size = env.action_space.n  # Number of possible actions

    agent = REINFORCEAgent(state_size, action_size, learning_rate=config.AGENT_PARAMS['learning_rate'])
    ######################### ------------------------------------------------------------
    num_episodes = config.NUM_EPISODES

    total_rewards = []  # List to store total rewards per episode

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
        total_rewards.append(total_reward)  # Store the total reward

        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}")

    # Save the total rewards for plotting
    with open('reinforce_total_rewards.pkl', 'wb') as f:
        pickle.dump(total_rewards, f)

    # Save the trained policy
    torch.save(agent.policy_network.state_dict(), 'reinforce_trading_model.pth')

    # Plot total rewards per episode
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE Training Progress: Total Rewards per Episode')
    plt.savefig('reinforce_training_rewards.png')
    plt.show()

    


    # Evaluate the agent
    evaluate_reinforce_agent(agent, testing_data)


    return total_rewards, agent

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

    # Map actions to labels
    action_mapping = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    actions = test_env.actions_memory

    # Plot actions over time
    plt.figure(figsize=(12, 3))
    plt.plot(actions, label='Actions', marker='o', linestyle='')
    plt.yticks([0, 1, 2], ['Hold', 'Buy', 'Sell'])
    plt.xlabel('Time Steps')
    plt.ylabel('Action')
    plt.title('Actions Taken Over Time')
    plt.savefig('reinforce_actions.png')
    plt.show()

    # for UI
    return {
        'cumulative_return': cumulative_return * 100,
        'portfolio_values': test_env.portfolio_values,
        'actions': test_env.actions_memory,
    }

if __name__ == "__main__":
    train_reinforce_agent()
