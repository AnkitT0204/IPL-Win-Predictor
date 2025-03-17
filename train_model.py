import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
allrounders = pd.read_csv('data/allrounders_ratings.csv')
batsmen = pd.read_csv('data/batsmen_ratings.csv')
bowlers = pd.read_csv('data/bowlers_ratings.csv')
matches = pd.read_csv('data/match_data.csv')

# Merge player ratings with match data
def merge_player_ratings(matches, players, rating_column, player_type):
    players = players.rename(columns={rating_column: f'{player_type}_rating'})
    matches = matches.merge(players[['Player', f'{player_type}_rating']], how='left', left_on='Top_Batsman_T1', right_on='Player')
    matches = matches.rename(columns={f'{player_type}_rating': f'{player_type}_rating_T1'})
    matches = matches.drop(columns=['Player'])  # Drop the 'Player' column to avoid duplication

    matches = matches.merge(players[['Player', f'{player_type}_rating']], how='left', left_on='Top_Batsman_T2', right_on='Player')
    matches = matches.rename(columns={f'{player_type}_rating': f'{player_type}_rating_T2'})
    matches = matches.drop(columns=['Player'])  # Drop the 'Player' column to avoid duplication

    return matches

# Merge ratings for batsmen, bowlers, and all-rounders
matches = merge_player_ratings(matches, batsmen, 'Rating', 'batsman')
matches = merge_player_ratings(matches, bowlers, 'Bowler Rating', 'bowler')
matches = merge_player_ratings(matches, allrounders, 'All-Rounder Rating', 'allrounder')

# Fill NaN values with 0 (for players without ratings)
matches = matches.fillna(0)

# Select relevant features for the model
features = ['batsman_rating_T1', 'batsman_rating_T2', 'bowler_rating_T1', 'bowler_rating_T2', 'allrounder_rating_T1', 'allrounder_rating_T2']
X = matches[features]
y = matches['Winner']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/ipl_model.pkl')
print("Model trained and saved successfully!")