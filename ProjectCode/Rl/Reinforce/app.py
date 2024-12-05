import streamlit as st
import pandas as pd
import pickle
from data_loader import load_data  # Import the preprocessing function
from reinforce_agent import REINFORCEAgent
from main_reinforce import train_reinforce_agent, evaluate_reinforce_agent
import matplotlib.pyplot as plt

# Title
st.title("REINFORCE Trading Algorithm Dashboard")

# Sidebar: Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload Feature-Engineered Dataset (CSV)", type=["csv"])

if uploaded_file:
    try:
        # Preprocess data using the data_loader.py function
        data = load_data(uploaded_file)
        st.write("### Preprocessed Dataset Preview")
        st.dataframe(data.head())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Training Hyperparameters
    st.sidebar.write("### Training Hyperparameters")
    episodes = st.sidebar.number_input("Number of Episodes", min_value=10, max_value=1000, value=50)
    # giving error - fix it
    #learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)
    gamma = st.sidebar.slider("Discount Factor (Gamma)", min_value=0.0, max_value=1.0, value=0.99)

    # Train Button
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            train_rewards, trained_agent = train_reinforce_agent(data)
            st.success("Training Completed!")

            # Plot Training Rewards
            st.write("### Training Rewards per Episode")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(train_rewards)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.set_title("Training Progress")
            st.pyplot(fig)

            # Save trained agent
            with open("trained_agent.pkl", "wb") as f:
                pickle.dump(trained_agent, f)

    # Evaluate Button
    if st.button("Evaluate Agent"):
        try:
            with open("trained_agent.pkl", "rb") as f:
                trained_agent = pickle.load(f)
            evaluation_results = evaluate_reinforce_agent(trained_agent, data)

            # Display Metrics and Results
            st.write(f"### Cumulative Return: {evaluation_results['cumulative_return']:.2f}%")
            st.write("### Portfolio Value Over Time")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(evaluation_results['portfolio_values'], label="Portfolio Value")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Portfolio Value")
            ax.legend()
            st.pyplot(fig)

        except FileNotFoundError:
            st.error("No trained agent found. Please train the agent first.")
