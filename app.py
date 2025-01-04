import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from nba_api.stats.endpoints import leaguegamefinder
import matplotlib.pyplot as plt
import seaborn as sns
import io
from dotenv import load_dotenv
import google.generativeai as genai
import os
import textwrap

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load models
model_winner = joblib.load("saved_models/xgb_classifier_model.pkl")
model_pts = joblib.load('saved_models/linear_regression_model_player.pkl')

# Load game data
@st.cache_data
def load_game_data():
    gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable='01/31/2020', league_id_nullable='00')
    games = gamefinder.get_data_frames()[0]
    games['Home'] = games['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    games['AST-TOV Ratio'] = games['AST'] / (games['TOV'] + 1e-5)  # Avoid division by zero
    games['eFG%'] = (games['FGM'] + 0.5 * games['FG3M']) / games['FGA']
    games['TrueShooting%'] = games['PTS'] / (2 * (games['FGA'] + 0.44 * games['FTA']))
    return games

games = load_game_data()

# Load player data
@st.cache_data
def load_player_data():
    joined_data = pd.read_csv('combined.csv').set_index('Player')
    return joined_data

player_data = load_player_data()

# Load dataset for analysis
@st.cache_data
def load_analysis_data():
    return pd.read_csv("data/nba_2024_per_game_stats.csv")

df = load_analysis_data()

# Functions
def predict_winner(team1, team2):
    team1_data = games[games['TEAM_NAME'] == team1].iloc[0]
    team2_data = games[games['TEAM_NAME'] == team2].iloc[0]

    team1_features = team1_data[['Home', 'eFG%', 'TrueShooting%', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'AST-TOV Ratio']].values.reshape(1, -1)
    team2_features = team2_data[['Home', 'eFG%', 'TrueShooting%', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'AST-TOV Ratio']].values.reshape(1, -1)

    team1_prob = model_winner.predict_proba(team1_features)[:, 1]
    team2_prob = model_winner.predict_proba(team2_features)[:, 1]

    if team1_prob > team2_prob:
        return team1, team1_prob
    else:
        return team2, team2_prob

# Helper functions for LLM
def to_markdown(text):
    text = text.replace('\u2022', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def get_gemini_response(question):
    """Fetch response from Gemini model."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

def predict_winner_with_llm(home_team, away_team):
    """Predict winner between home and away team using LLM."""
    prompt = f"Based on NBA historical data and team performance, who is likely to win: {home_team} (home) vs {away_team} (away)?"
    return get_gemini_response(prompt)

def player_points_prediction_with_llm(player_name):
    """Get points prediction for the player using LLM."""
    prompt = (f"Provide a detailed analysis of {player_name}'s performance, including points scored in the 2022-2023 and 2023-2024 seasons, "
              "and the current season 2024-2025. Include a brief analysis of trends and expectations.")
    return get_gemini_response(prompt)

def predict_mvp_with_llm():
    """Predict the most likely MVP for the current NBA season using LLM."""
    prompt = "Based on current NBA performance and historical trends, who is most likely to win the MVP award for the 2024-2025 season?"
    return get_gemini_response(prompt)

# Pages
def nba_game_winner_prediction():
    st.title("NBA Game Winner Prediction")
    st.markdown("Select two teams to predict the winner of their upcoming game.")

    teams = games['TEAM_NAME'].unique()
    team1 = st.selectbox("Select The Home Team", teams)
    team2 = st.selectbox("Select The Away Team", teams)

    if team1 == team2:
        st.warning("Please select different teams for Team 1 and Team 2.")
    else:
        if st.button("Predict Winner"):
            winner, prob = predict_winner(team1, team2)
            st.write(f"The predicted winner is **{winner}** with a probability of **{prob[0]:.2f}**")

def nba_player_points_prediction():
    st.title("NBA Player Points Prediction")
    st.markdown("Enter a player's name to predict their points for the 2023-2024 season.")

    player_name = st.text_input("Player Name", "")

    if player_name:
        if player_name in player_data.index:
            player_row = player_data.loc[player_name]
            player_features = player_row.drop(
                ['PTS_22', 'Pos_22', 'Pos_23', 'Team_22', 'Team_23', 'Rk_22', 'Rk_23'], axis=1
            )

            pts_predicted = model_pts.predict(player_features.values.reshape(1, -1))[0]
            st.write(f"**Predicted Points for {player_name} (2023-2024):** {pts_predicted:.2f}")
            pts_actual = player_row['PTS_22']
            st.write(f"Actual Points in 2022-2023: {pts_actual}")
        else:
            st.error("Player not found. Please enter a valid player's name.")

def nba_insights_with_llm():
    st.title("NBA Insights with LLM")
    st.markdown("Get advanced insights from the LLM by providing a question or context.")

    prompt = st.text_area("Enter your prompt", "Analyze the performance of the Lakers in the last 5 games.")
    if st.button("Get Insights"):
        result = get_gemini_response(prompt)
        st.write("**LLM Response:**")
        st.write(result)


def nba_analysis():
    pages = [
        "Introduction",
        "Dataset Overview",
        "Statistical Summary",
        "Player Metrics",
        "Team Analysis",
        "Visualizations",
        "Advanced Metrics"
    ]

    selected_page = st.sidebar.selectbox("Select Analysis Page", pages)

    if selected_page == "Introduction":
        st.title("NBA Data Analysis 2024")
        st.write("Welcome to the NBA Data Analysis app! Explore player and team performance, visualize key stats, and uncover insights from the 2024 NBA season.")

    elif selected_page == "Dataset Overview":
        st.title("Dataset Overview")
        st.write(df.head())
        st.write("### Dataset Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    elif selected_page == "Statistical Summary":
        st.title("Statistical Summary")
        st.write("### Descriptive Statistics")
        st.write(df.describe())
        st.write("### Top Scorers")
        st.write(df.nlargest(5, 'PTS')[['Player', 'PTS']])

    elif selected_page == "Player Metrics":
        st.title("Player Metrics")
        if 'PPG' not in df.columns:  # Ensure PPG column is computed
            df['PPG'] = df['PTS'] / df['G']
        st.write("### Points Per Game (PPG)")
        st.write(df[['Player', 'PPG']].sort_values(by='PPG', ascending=False).head(10))

    elif selected_page == "Team Analysis":
        st.title("Team Analysis")
        team_stats = df.groupby('Team').sum()[['PTS', 'TRB', 'AST']].sort_values(by='PTS', ascending=False)
        st.write("### Team Statistics")
        st.write(team_stats)
        st.bar_chart(team_stats[['PTS']])

    elif selected_page == "Visualizations":
        st.title("Visualizations")

        # Scoring and Play Time Analysis
        st.write("### Scoring and Play Time Analysis")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='MP', y='PTS', color='orange', ax=ax1)
        ax1.set_title("Impact of Minutes Played on Scoring")
        st.pyplot(fig1)

        # Fouls vs Turnovers
        st.write("### Fouls vs Turnovers")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='PF', y='TOV', hue='Team', alpha=0.7, ax=ax2)
        ax2.set_title("Fouls vs Turnovers")
        st.pyplot(fig2)
        
        # 3-Point % vs. 2-Point %
        st.write("### 3-Point % vs. 2-Point %")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='3P%', y='2P%', hue='Player', ax=ax1)
        st.pyplot(fig1)

        # Age vs Points Scored
        st.write("### Age vs Points Scored")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.regplot(data=df, x='Age', y='PTS', scatter_kws={'alpha': 0.6}, ax=ax2)
        st.pyplot(fig2)

        # Defensive Stats
        st.write("### Defensive Stats")
        defensive_stats = df.groupby('Player')[['STL', 'BLK']].sum().sort_values(by='STL', ascending=False)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        defensive_stats.head(10).plot(kind='bar', stacked=True, title="Top Defensive Players", ax=ax3)
        ax3.set_ylabel("Total Stats")
        st.pyplot(fig3)

        # Shot Distribution
        st.write("### Shot Type Distribution")
        shot_distribution = df[['3P', '2P', 'FT']].sum()
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        shot_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax4)
        ax4.set_ylabel("")
        ax4.set_title("Shot Type Distribution")
        st.pyplot(fig4)

        # Consistent Scorers
        st.write("### Consistent Scorers")
        df['PTS_std_dev'] = df.groupby('Player')['PTS'].transform('std')
        consistent_players = df.groupby('Player')['PTS_std_dev'].mean().sort_values()
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        consistent_players.head(10).plot(kind='bar', color='green', title="Top Consistent Scorers", ax=ax5)
        ax5.set_ylabel("Standard Deviation in Points")
        st.pyplot(fig5)

        # Points Per Game Distribution
        if 'PPG' not in df.columns:  # Compute PPG if missing
            df['PPG'] = df['PTS'] / df['G']
        st.write("### Points Per Game Distribution")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.histplot(df['PPG'], kde=True, bins=20, color='blue', ax=ax3)
        st.pyplot(fig3)

    elif selected_page == "Advanced Metrics":
        st.title("Advanced Metrics")
        df['MVP_Score'] = (0.5 * df['PTS']) + (0.3 * df['AST']) + (0.2 * df['TRB'])
        st.write("### MVP Score")
        st.write(df.nlargest(5, 'MVP_Score')[['Player', 'MVP_Score']])
# Main App
st.sidebar.title("NBA App Navigation")
main_pages = [
    "Game Winner Prediction",
    "Player Points Prediction",
    "LLM Insights",
    "Data Analysis"
]

selected_main_page = st.sidebar.radio("Select a Main Page", ["Game Winner Prediction", "Player Points Prediction", "LLM Insights", "Data Analysis"])

if selected_main_page == "Game Winner Prediction":
    nba_game_winner_prediction()
elif selected_main_page == "Player Points Prediction":
    nba_player_points_prediction()
elif selected_main_page == "LLM Insights":
    nba_insights_with_llm()
elif selected_main_page == "Data Analysis":
    nba_analysis()

    




