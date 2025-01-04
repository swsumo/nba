#to get the games data since 2020
'''
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

# Fetch games starting from January 31, 2020
gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable='01/31/2020', league_id_nullable='00')

# Get the game data as a dataframe
games = gamefinder.get_data_frames()[0]

# Define the CSV file name
csv_file = "nba_games_since_01_31_2020.csv"

# Save the data to the CSV file
games.to_csv(csv_file, index=False)

print(f"Game data saved to {csv_file}")

# Example: Display first few rows of the data
print(games.head())
'''


#to get the teams csv file
'''
import csv
from nba_api.stats.static import teams

# Fetch all teams
team_dict = teams.get_teams()

# Define the CSV file name
csv_file = "nba_teams.csv"

# Save the data to the CSV file
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(["ID", "Full Name", "Abbreviation", "Nickname", "City", "State", "Year Founded", "Is NBA Team"])
    
    # Write team data
    for team in team_dict:
        writer.writerow([
            team.get('id', 'Unknown'),
            team.get('full_name', 'Unknown'),
            team.get('abbreviation', 'Unknown'),
            team.get('nickname', 'Unknown'),
            team.get('city', 'Unknown'),
            team.get('state', 'Unknown'),
            team.get('year_founded', 'Unknown'),
            team.get('is_nba_team', 'Unknown')  # Use .get() to avoid KeyError
        ])

print(f"Team data saved to {csv_file}")

'''
'''
import requests
import pandas as pd
from bs4 import BeautifulSoup

# URL of the page with the table
# Add new urls acc and change the csv file name 
#url = 'https://www.basketball-reference.com/leagues/NBA_2024_per_game.html#per_game_stats'
url='https://www.basketball-reference.com/leagues/NBA_2023_per_game.html#per_game_stats'

# Send a request to the website
response = requests.get(url)

# Parse the page content
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table with the id 'per_game_stats'
table = soup.find('table', {'id': 'per_game_stats'})

# Read the table into a pandas DataFrame
df = pd.read_html(str(table))[0]

# Save the DataFrame to a CSV file
df.to_csv('nba_2023_per_game_stats.csv', index=False)

print("Table saved as nba_2023_per_game_stats.csv")
'''

import pandas as pd

# Load the data from the two CSV files
nba_2023 = pd.read_csv('nba_2023_per_game_stats.csv')
nba_2024 = pd.read_csv('nba_2024_per_game_stats.csv')

# Rename the columns of the 2023 dataset to match the requested format
nba_2023.columns = [col + '_22' if col != 'Player' else col for col in nba_2023.columns]

# Rename the columns of the 2024 dataset to match the requested format
nba_2024.columns = [col + '_23' if col != 'Player' else col for col in nba_2024.columns]

# Merge the two dataframes on the 'Player' column
combined = pd.merge(nba_2023, nba_2024, on='Player')

# Save the result to a new CSV file
combined.to_csv('combined.csv', index=False)

print("The combined CSV has been saved as 'combined.csv'.")

