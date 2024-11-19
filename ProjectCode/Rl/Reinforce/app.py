import streamlit as st
import pandas as pd
import pickle
import torch
from reinforce_agent import REINFORCEAgent
from environment import TradingEnv
from main_reinforce import train_reinforce_agent, evaluate_reinforce_agent
import config
import matplotlib.pyplot as plt

# title
st.title("REINFORCE Trading Algorithm Dashboard")

# Sidebar: Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload your Feature-Engineered Dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Preprocessing
    st.write(f"Dataset Length: {len(data)} rows")

    # Start training button
    if st.button("Start Training"):
        with st.spinner("Training in progress..."): # progess bar
            # Train REINFORCE agent
            train_rewards, trained_agent = train_reinforce_agent(data)

            # Plot training rewards
            st.write("### Training Rewards per Episode")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(train_rewards)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.set_title("Training Progress")
            st.pyplot(fig)

        # Save the trained agent for later evaluation
        with open("trained_agent.pkl", "wb") as f:
            pickle.dump(trained_agent, f)

        st.success("Training Completed! The agent has been saved.")

    # Evaluate the agent
    if st.button("Evaluate Agent"):
        if not uploaded_file:
            st.error("Please upload a dataset for evaluation.")
        else:
            with st.spinner("Evaluating the agent..."):
                # Load trained agent
                with open("trained_agent.pkl", "rb") as f:
                    trained_agent = pickle.load(f)

                # Evaluate the agent
                evaluation_results = evaluate_reinforce_agent(trained_agent, data)

                # Display cumulative return
                st.write(f"### Cumulative Return: {evaluation_results['cumulative_return']:.2f}%")

                # Portfolio Value Over Time
                st.write("### Portfolio Value Over Time")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(evaluation_results['portfolio_values'], label="Portfolio Value")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Portfolio Value")
                ax.legend()
                st.pyplot(fig)

                # Actions Taken Over Time
                st.write("### Actions Taken Over Time")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(evaluation_results['actions'], marker='o', linestyle='', label="Actions")
                ax.set_yticks([0, 1, 2])
                ax.set_yticklabels(['Hold', 'Buy', 'Sell'])
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Action")
                ax.legend()
                st.pyplot(fig)
