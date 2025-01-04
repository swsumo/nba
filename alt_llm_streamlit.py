# NBA Expert Model
from dotenv import load_dotenv
import streamlit as st
import os
import textwrap
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Helper functions
def to_markdown(text):
    text = text.replace('\u2022', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def get_gemini_response(question):
    """Fetch response from Gemini model."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

def predict_winner(home_team, away_team):
    """Predict winner between home and away team."""
    prompt = f"Based on NBA historical data and team performance, who is likely to win: {home_team} (home) vs {away_team} (away)?"
    return get_gemini_response(prompt)

def player_points_prediction(player_name):
    """Get points prediction for the player."""
    prompt = (f"Provide a detailed analysis of {player_name}'s performance, including points scored in the 2022-2023 and 2023-2024 seasons, "
              "and the current season 2024-2025. Include a brief analysis of trends and expectations.")
    return get_gemini_response(prompt)

def predict_mvp():
    """Predict the most likely MVP for the current NBA season."""
    prompt = "Based on current NBA performance and historical trends, who is most likely to win the MVP award for the 2024-2025 season?"
    return get_gemini_response(prompt)

# Streamlit app setup
st.set_page_config(page_title="NBA Expert Model")
st.header("NBA Expert Model")

# User interaction options
options = ["Winner Prediction", "Player Points Prediction", "MVP Prediction"]
choice = st.radio("Select a feature:", options)

if choice == "Winner Prediction":
    st.subheader("Winner Prediction")
    home_team = st.text_input("Enter Home Team:")
    away_team = st.text_input("Enter Away Team:")
    if st.button("Predict Winner"):
        if home_team and away_team:
            result = predict_winner(home_team, away_team)
            st.write(f"Prediction: {result}")
        else:
            st.error("Please enter both home and away team names.")

elif choice == "Player Points Prediction":
    st.subheader("Player Points Prediction")
    player_name = st.text_input("Enter Player Name:")
    if st.button("Predict Points"):
        if player_name:
            points_result = player_points_prediction(player_name)
            st.write(f"Prediction: {points_result}")
        else:
            st.error("Please enter a player name.")

elif choice == "MVP Prediction":
    st.subheader("MVP Prediction")
    if st.button("Predict MVP"):
        mvp_result = predict_mvp()
        st.write(f"Prediction: {mvp_result}")
