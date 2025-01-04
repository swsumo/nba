# NBA Analysis: Player and Team Performance Prediction

## Overview
This project provides an AI-powered system that predicts the outcome of NBA games and player performance. It leverages data fetched from the **NBA API** and **Football Reference** website to analyze and predict key insights. The system is designed to predict:
1. **Winning Team Prediction**: Based on historical data, the model predicts which team is more likely to win a match.
2. **Player Performance Prediction**: It forecasts how many points a player will score in the next game based on their historical performance.

Additionally, the project includes:
- Visualizations that help understand patterns in the data.
- **Google Gemini LLM** integration for detailed Q&A on NBA-related topics.

## Features
- **Winning Team Prediction Model**: Predicts the outcome of NBA games based on historical performance data of teams.
- **Player Performance Model**: Predicts how many points a player will score in their next match.
- **Exploratory Data Analysis (EDA)**: Visualizations and analysis of NBA data, including player and team statistics.
- **Google Gemini Integration**: A dedicated model (via `alt_llm.py`) powered by Google Gemini for answering NBA-related questions.
- **Streamlit Application**: A web-based interface to interact with the models and predictions via `app.py`.

## Data Sources
The data for this project is fetched from the following sources:
1. **NBA API**: Provides data on players, teams, and game statistics.
2. **Football Reference**: Additional data from the **Basketball Reference** site for detailed team performance and player stats.

For how the data is fetched, refer to the `data_fetch.py` script.

## Files & Structure

### Main Files
- **`app.py`**: The main Streamlit web application that allows users to interact with the models and view the predictions and visualizations.
- **`alt_llm.py`**: An alternative interface specifically for querying NBA-related questions using the **Google Gemini** LLM.
- **`data_fetch.py`**: Fetches data from external sources like the NBA API and Basketball Reference, preparing it for the models.
- **`EDA.ipynb`**: Jupyter notebook containing exploratory data analysis (EDA), providing visual insights into the dataset and key statistics of players and teams.

### Models
- **`saved_models/`**: Contains the saved models in the form of `.pkl` files, which are used for predictions.
- **`saved_jupyter_notebooks/`**: Jupyter notebooks where the models were originally trained, with code for training and testing the models.

### Folder Structure
```bash
NBA_Analysis/
│
├── app.py                     # Main Streamlit application for the interface
├── alt_llm.py                 # Alternative interface for querying NBA-related info
├── data_fetch.py              # Script for fetching data from external APIs
├── EDA.ipynb                  # Exploratory Data Analysis notebook
├── saved_models/              # Folder containing saved models (Pickle files)
│   ├── player_performance_model.pkl
│   └── team_performance_model.pkl
├── saved_jupyter_notebooks/   # Folder containing Jupyter notebooks for model training
│   ├── player_performance_model_training.ipynb
│   └── team_performance_model_training.ipynb
└── req.txt           # Required Python packages for the project
```

## Installation

### Requirements
Ensure that you have Python 3.x installed. You can install the required dependencies using `pip`.

1. Clone the repository:
    ```bash
    git clone https://github.com/swsumo/Nba_analysis2.git
    cd Nba_analysis2
    ```

2. Install dependencies:
    ```bash
    pip install -r req.txt
    ```

3. Ensure you have access to the necessary APIs (NBA API and Basketball Reference), and set up any environment variables or credentials as needed.

### Running the App
To start the Streamlit web app:

1. Navigate to the project folder in your terminal.
2. Run the app using Streamlit:
    ```bash
    streamlit run app.py
    ```
3. The web interface should open in your default browser, where you can interact with the models and view the predictions.

### Running the Alternative LLM Site
To use the **Google Gemini**-powered Q&A interface:

1. Run the alternative site:
    ```bash
    streamlit run alt_llm.py
    ```
2. This will launch a separate Streamlit interface dedicated to answering NBA-related questions.

## Models & Prediction

### Player Performance Model
- **Description**: This model predicts how many points a player is likely to score in their next match.
- **Training**: The model is trained using historical player performance data from the NBA games dataset.

### Team Performance Model
- **Description**: This model predicts which team is more likely to win a given match based on historical team performance.
- **Training**: The model uses historical NBA team data and game outcomes to make its predictions.

## Visualizations

The `EDA.ipynb` notebook contains various visualizations for analyzing the dataset, including:
- Distribution of points scored by players.
- Team performance over different seasons.
- Player and team statistics comparisons.

These visualizations provide deeper insights into the data and help understand the patterns influencing game outcomes and player performance.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
