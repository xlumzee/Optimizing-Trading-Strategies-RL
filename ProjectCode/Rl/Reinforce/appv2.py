import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data  
from reinforce_agent import REINFORCEAgent
from main_reinforce import train_reinforce_agent, evaluate_reinforce_agent
import config

# for wide layout and removing padding
st.set_page_config(layout="wide")

# Remove default Streamlit padding/margins
st.markdown(
    """
    <style>
    /* Remove extra padding from the top */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    /* Optional: reduce side padding as well */
    .block-container, .main, .viewerBadge_container__1QSob {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("REINFORCE Trading Algorithm Dashboard")

# Sidebar: Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload Feature-Engineered Dataset (CSV)", type=["csv"])

st.sidebar.write("### Training Hyperparameters")
config.NUM_EPISODES = st.sidebar.number_input(
    "Number of Episodes",
    min_value=10,
    max_value=1000,
    value=config.NUM_EPISODES
)

def max_drawdown(portfolio_values):
    """
    portfolio_values: list or array of cumulative portfolio values
    Returns the maximum drawdown as a fraction (e.g., 0.20 = 20%)
    """
    arr = np.array(portfolio_values)
    cumulative_max = np.maximum.accumulate(arr)
    drawdowns = (cumulative_max - arr) / cumulative_max
    return np.max(drawdowns)

if uploaded_file:
    try:
        data = load_data(uploaded_file)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Convert "date" to datetime if present
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Create a two-column layout: left (1) and right (2)
    left_col, right_col = st.columns([1, 2])

   #left side
    with left_col:
        st.subheader("Preprocessed Dataset Preview")
        # Show only first 10 rows to fit better
        st.dataframe(data.head(10))

        # Display basic statistics - integrate the following code snippet with the feature engineering pipeline and it should display dynamically
        # needs to be updated

        # st.write("**New Feature-Engineered Variables**:")
        # st.markdown("""
        # - **RET**: Daily returns
        # - **VOL_CHANGE**: Volume change ratio
        # - **BA_SPREAD**: Bid-Ask spread
        # - **ILLIQUIDITY**: Amihud illiquidity measure
        # - **RET_ema_12, RET_ema_26**: Exponential moving averages of returns
        # - **day_of_week**: Numeric day indicator (0=Mon, 6=Sun)
        # """)

   #EDA on the right partition
    with right_col:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Create a 2x2 layout inside the right column
        row_top_left, row_top_right = st.columns(2)
        row_bot_left, row_bot_right = st.columns(2)

        # 1) Time-Series (PRC over Date)
        with row_top_left:
            st.write("**PRC over Date**")
            if "date" in data.columns and "PRC" in data.columns:
                df_sorted = data.sort_values(by="date")
                fig_ts, ax_ts = plt.subplots()
                ax_ts.plot(df_sorted["date"], df_sorted["PRC"], label="Price")
                ax_ts.set_xlabel("Date")
                ax_ts.set_ylabel("PRC")
                ax_ts.set_title("Time Series of PRC")
                plt.xticks(rotation=30)
                ax_ts.legend()
                st.pyplot(fig_ts)
            else:
                st.write("Missing 'date' or 'PRC' column.")

        # 2) Histogram of Returns
        with row_top_right:
            st.write("**Histogram of Returns (RET)**")
            if "RET" in data.columns:
                fig_ret, ax_ret = plt.subplots()
                ax_ret.hist(data["RET"].dropna(), bins=30)
                ax_ret.set_xlabel("RET")
                ax_ret.set_ylabel("Frequency")
                ax_ret.set_title("Distribution of RET")
                st.pyplot(fig_ret)
            else:
                st.write("No 'RET' column found.")

        # 3) Correlation Heatmap
        with row_bot_left:
            st.write("**Correlation Heatmap**")
            if len(numeric_cols) > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(4, 3))
                corr_matrix = data[numeric_cols].corr()
                sns.heatmap(corr_matrix, ax=ax_corr, annot=False, cmap="coolwarm")
                ax_corr.set_title("Correlation")
                st.pyplot(fig_corr)
            else:
                st.write("Not enough numeric columns for correlation.")

        # 4) Box Plot: Returns by Day of Week
        with row_bot_right:
            st.write("**Box Plot: RET by day_of_week**")
            if "day_of_week" in data.columns and "RET" in data.columns:
                fig_box, ax_box = plt.subplots()
                sns.boxplot(x="day_of_week", y="RET", data=data, ax=ax_box)
                ax_box.set_title("RET by day_of_week")
                ax_box.set_xlabel("Day of Week")
                ax_box.set_ylabel("RET")
                st.pyplot(fig_box)
            else:
                st.write("Need 'day_of_week' and 'RET' columns to plot.")

   #--------------------
   # TRAINING & EVALUATION
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            train_rewards, trained_agent = train_reinforce_agent(data)
            st.success("Training Completed!")

            # Plot training rewards
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

    #Evaluation - throwing error for the addn metric - fix this
    if st.button("Evaluate Agent"):
        try:
            with open("trained_agent.pkl", "rb") as f:
                trained_agent = pickle.load(f)

            evaluation_results = evaluate_reinforce_agent(trained_agent, data)

            st.write(f"### Cumulative Return: {evaluation_results['cumulative_return']:.2f}%")
            st.write("### Portfolio Value Over Time")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(evaluation_results['portfolio_values'], label="Portfolio Value")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Portfolio Value")
            ax.legend()
            st.pyplot(fig)

            # Metric selection
            metrics_list = [
                "Sharpe Ratio",
                "Maximum Drawdown",
                "Volatility (Std Dev of Returns)",
                "Distribution of Daily Returns",
                "Win Rate"
            ]
            selected_metric = st.selectbox("Select a metric to display:", metrics_list)

            # Compute & display metric
            if selected_metric == "Sharpe Ratio":
                if 'daily_returns' in evaluation_results:
                    daily_returns = evaluation_results['daily_returns']
                    avg_return = np.mean(daily_returns)
                    std_return = np.std(daily_returns)
                    risk_free_rate = 0.0
                    sharpe_ratio = (avg_return - risk_free_rate) / (std_return + 1e-9)
                    st.write(f"**Sharpe Ratio**: {sharpe_ratio:.2f}")
                else:
                    st.write("**daily_returns** not found in evaluation_results.")

            elif selected_metric == "Maximum Drawdown":
                if 'portfolio_values' in evaluation_results:
                    mdd = max_drawdown(evaluation_results['portfolio_values'])
                    st.write(f"**Maximum Drawdown**: {mdd*100:.2f}%")
                else:
                    st.write("**portfolio_values** not found in evaluation_results.")

            elif selected_metric == "Volatility (Std Dev of Returns)":
                if 'daily_returns' in evaluation_results:
                    vol = np.std(evaluation_results['daily_returns'])
                    st.write(f"**Volatility** (Std Dev): {vol:.4f}")
                else:
                    st.write("**daily_returns** not found in evaluation_results.")

            elif selected_metric == "Distribution of Daily Returns":
                if 'daily_returns' in evaluation_results:
                    daily_returns = evaluation_results['daily_returns']
                    fig2, ax2 = plt.subplots()
                    ax2.hist(daily_returns, bins=30)
                    ax2.set_title("Distribution of Daily Returns")
                    ax2.set_xlabel("Return")
                    ax2.set_ylabel("Frequency")
                    st.pyplot(fig2)
                else:
                    st.write("**daily_returns** not found in evaluation_results.")

            elif selected_metric == "Win Rate":
                if 'trade_profits' in evaluation_results:
                    trade_profits = evaluation_results['trade_profits']
                    num_wins = sum(1 for p in trade_profits if p > 0)
                    win_rate = (num_wins / len(trade_profits)) * 100
                    st.write(f"**Win Rate**: {win_rate:.2f}%")
                else:
                    st.write("**trade_profits** not found in evaluation_results.")

        except FileNotFoundError:
            st.error("No trained agent found. Please train the agent first.")

else:
    st.write("Please upload a CSV file to begin.")
