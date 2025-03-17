import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="wide"
)

# Function to load and prepare data
def load_data():
    try:
        # Check if processed data exists
        if os.path.exists('processed_match_data.csv'):
            match_data = pd.read_csv('processed_match_data.csv')
            if os.path.exists('batsmen_ratings.csv'):
                batsmen = pd.read_csv('batsmen_ratings.csv')
            else:
                batsmen = pd.DataFrame(columns=['Player', 'Rating'])
            
            if os.path.exists('bowlers_ratings.csv'):
                bowlers = pd.read_csv('bowlers_ratings.csv')
            else:
                bowlers = pd.DataFrame(columns=['Player', 'Bowler Rating'])
                
            if os.path.exists('allrounders_ratings.csv'):
                allrounders = pd.read_csv('allrounders_ratings.csv')
            else:
                allrounders = pd.DataFrame(columns=['Player', 'All-Rounder Rating'])
                
            return match_data, batsmen, bowlers, allrounders
            
        # If not, load from provided files
        match_data = pd.read_csv('match_data.csv')
        
        # Try to load player ratings
        try:
            batsmen = pd.read_csv('batsmen_ratings.csv')
        except:
            batsmen = pd.DataFrame(columns=['Player', 'Rating'])
            
        try:
            bowlers = pd.read_csv('bowlers_ratings.csv')
        except:
            bowlers = pd.DataFrame(columns=['Player', 'Bowler Rating'])
            
        try:
            allrounders = pd.read_csv('allrounders_ratings.csv')
        except:
            allrounders = pd.DataFrame(columns=['Player', 'All-Rounder Rating'])
            
        return match_data, batsmen, bowlers, allrounders
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create empty dataframes if files don't exist
        match_data = pd.DataFrame()
        batsmen = pd.DataFrame(columns=['Player', 'Rating'])
        bowlers = pd.DataFrame(columns=['Player', 'Bowler Rating'])
        allrounders = pd.DataFrame(columns=['Player', 'All-Rounder Rating'])
        
        return match_data, batsmen, bowlers, allrounders

# Data preprocessing function
def preprocess_data(match_data, batsmen, bowlers, allrounders):
    if match_data.empty:
        return pd.DataFrame(), []
    
    # Create a copy to avoid modifying the original
    df = match_data.copy()
    
    # Handle missing values
    df['Toss_Winner'] = df['Toss_Winner'].fillna('Unknown')
    df['Toss_Decision'] = df['Toss_Decision'].fillna('Unknown')
    
    # Create team strength features
    team1_list = df['Team_1'].unique().tolist()
    team2_list = df['Team_2'].unique().tolist()
    teams = pd.Series(team1_list + team2_list).unique()
    
    # Create team rating dictionaries based on player ratings
    team_batsmen_ratings = {team: 0 for team in teams}
    team_bowler_ratings = {team: 0 for team in teams}
    team_allrounder_ratings = {team: 0 for team in teams}
    
    # Extract top players from each team based on match data
    for team in teams:
        team_top_batsmen = df[df['Team_1'] == team]['Top_Batsman_T1'].tolist() + df[df['Team_2'] == team]['Top_Batsman_T2'].tolist()
        team_top_batsmen = pd.Series(team_top_batsmen).dropna().unique()
        
        for player in team_top_batsmen:
            if player in batsmen['Player'].values:
                team_batsmen_ratings[team] += batsmen[batsmen['Player'] == player]['Rating'].values[0]
        
        team_top_bowlers = df[df['Team_1'] == team]['Best_Bowler_T1'].tolist() + df[df['Team_2'] == team]['Best_Bowler_T2'].tolist()
        team_top_bowlers = pd.Series(team_top_bowlers).dropna().unique()
        
        for player in team_top_bowlers:
            if player in bowlers['Player'].values:
                team_bowler_ratings[team] += bowlers[bowlers['Player'] == player]['Bowler Rating'].values[0]
        
        for player in pd.Series(team_top_batsmen).tolist() + pd.Series(team_top_bowlers).tolist():
            if player in allrounders['Player'].values:
                team_allrounder_ratings[team] += allrounders[allrounders['Player'] == player]['All-Rounder Rating'].values[0]
    
    # Normalize team ratings
    max_bat_rating = max(team_batsmen_ratings.values()) if team_batsmen_ratings else 1
    max_bowl_rating = max(team_bowler_ratings.values()) if team_bowler_ratings else 1
    max_all_rating = max(team_allrounder_ratings.values()) if team_allrounder_ratings else 1
    
    for team in teams:
        team_batsmen_ratings[team] = team_batsmen_ratings[team] / max_bat_rating if max_bat_rating > 0 else 0
        team_bowler_ratings[team] = team_bowler_ratings[team] / max_bowl_rating if max_bowl_rating > 0 else 0
        team_allrounder_ratings[team] = team_allrounder_ratings[team] / max_all_rating if max_all_rating > 0 else 0
    
    # Add team strength features to dataframe
    df['Team1_Batting_Strength'] = df['Team_1'].map(team_batsmen_ratings)
    df['Team1_Bowling_Strength'] = df['Team_1'].map(team_bowler_ratings)
    df['Team1_Allrounder_Strength'] = df['Team_1'].map(team_allrounder_ratings)
    
    df['Team2_Batting_Strength'] = df['Team_2'].map(team_batsmen_ratings)
    df['Team2_Bowling_Strength'] = df['Team_2'].map(team_bowler_ratings)
    df['Team2_Allrounder_Strength'] = df['Team_2'].map(team_allrounder_ratings)
    
    # Add venue statistics - home advantage
    venue_win_rates = {}
    for venue in df['Venue'].unique():
        venue_matches = df[df['Venue'] == venue]
        for team in teams:
            team_matches = venue_matches[(venue_matches['Team_1'] == team) | (venue_matches['Team_2'] == team)]
            if not team_matches.empty:
                team_wins = team_matches[team_matches['Winner'] == team].shape[0]
                venue_win_rates[(venue, team)] = team_wins / team_matches.shape[0]
            else:
                venue_win_rates[(venue, team)] = 0.5
    
    df['Team1_Venue_Advantage'] = df.apply(lambda row: venue_win_rates.get((row['Venue'], row['Team_1']), 0.5), axis=1)
    df['Team2_Venue_Advantage'] = df.apply(lambda row: venue_win_rates.get((row['Venue'], row['Team_2']), 0.5), axis=1)
    
    # Add toss advantage
    toss_impact = {}
    for team in teams:
        toss_wins = df[df['Toss_Winner'] == team]
        if not toss_wins.empty:
            wins_after_toss = toss_wins[toss_wins['Winner'] == team].shape[0]
            toss_impact[team] = wins_after_toss / len(toss_wins)
        else:
            toss_impact[team] = 0.5
    
    df['Team1_Toss_Advantage'] = df['Toss_Winner'].map(toss_impact)
    df['Team2_Toss_Advantage'] = df['Toss_Winner'].map(toss_impact)
    
    # Encode toss decision
    df['Toss_Decision_Bat'] = (df['Toss_Decision'] == 'Bat').astype(int)
    
    # Create target variable
    df['Team1_Win'] = (df['Winner'] == df['Team_1']).astype(int)
    
    # Select features for model
    features = [
        'Team1_Batting_Strength', 'Team1_Bowling_Strength', 'Team1_Allrounder_Strength',
        'Team2_Batting_Strength', 'Team2_Bowling_Strength', 'Team2_Allrounder_Strength',
        'Team1_Venue_Advantage', 'Team2_Venue_Advantage',
        'Team1_Toss_Advantage', 'Team2_Toss_Advantage', 'Toss_Decision_Bat'
    ]
    
    # Save processed data
    df.to_csv('processed_match_data.csv', index=False)
    
    return df, features

# Train the model
def train_model(df, features):
    if df.empty or not features:
        return None
    
    X = df[features]
    y = df['Team1_Win']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    joblib.dump(pipeline, 'ipl_model.pkl')
    
    return pipeline, accuracy, features

# Predict function
def predict_match(model, features, team1, team2, venue, toss_winner, toss_decision):
    if model is None:
        return None
    
    match_data, batsmen, bowlers, allrounders = load_data()
    
    if match_data.empty:
        return None
    
    new_match = pd.DataFrame({
        'Team_1': [team1],
        'Team_2': [team2],
        'Venue': [venue],
        'Toss_Winner': [toss_winner],
        'Toss_Decision': [toss_decision],
        'Winner': [team1]  # Placeholder
    })
    
    combined_data = pd.concat([match_data, new_match], ignore_index=True)
    
    processed_data, _ = preprocess_data(combined_data, batsmen, bowlers, allrounders)
    
    if processed_data.empty:
        return None
        
    new_match_processed = processed_data.iloc[-1]
    
    X_new = pd.DataFrame(index=[0])
    
    for feature in features:
        if feature in new_match_processed.index:
            try:
                value = new_match_processed[feature]
                if pd.isna(value):
                    X_new[feature] = 0.0
                elif isinstance(value, (np.bool_, bool)):
                    X_new[feature] = float(value)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    X_new[feature] = float(value)
                else:
                    try:
                        X_new[feature] = float(value)
                    except:
                        X_new[feature] = 0.0
            except Exception as e:
                X_new[feature] = 0.0
                print(f"Error with feature {feature}: {e}")
        else:
            X_new[feature] = 0.0
    
    try:
        prediction_proba = model.predict_proba(X_new)[0]
        
        team1_win_prob = prediction_proba[1]
        team2_win_prob = prediction_proba[0]
        
        feature_importances = {}
        rf_model = model.named_steps['classifier']
        for idx, feature in enumerate(features):
            if idx < len(rf_model.feature_importances_):
                feature_importances[feature] = rf_model.feature_importances_[idx]
            else:
                feature_importances[feature] = 0.0
        
        features_values = {feature: X_new[feature].iloc[0] for feature in features}
        
        result = {
            'team1': team1,
            'team2': team2,
            'team1_win_prob': team1_win_prob * 100,
            'team2_win_prob': team2_win_prob * 100,
            'prediction': team1 if team1_win_prob > team2_win_prob else team2,
            'feature_importances': feature_importances,
            'features_values': features_values
        }
        
        return result
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Main app
def main():
    st.title("🏏 IPL Match Winner Predictor")
    st.markdown("""
    This app predicts the winner of IPL cricket matches based on team statistics, venue information, and toss outcomes.
    """)
    
    with st.spinner('Loading data...'):
        match_data, batsmen, bowlers, allrounders = load_data()
    
    if not match_data.empty:
        st.success(f"✅ Loaded {len(match_data)} matches")
        
        with st.spinner('Processing data...'):
            df, features = preprocess_data(match_data, batsmen, bowlers, allrounders)
        
        if df.empty or not features:
            st.error("Data preprocessing failed. Please check your data files.")
            return
            
        model = None
        if os.path.exists('ipl_model.pkl'):
            try:
                with st.spinner('Loading trained model...'):
                    model = joblib.load('ipl_model.pkl')
                    X = df[features]
                    y = df['Team1_Win']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.success(f"Model accuracy: {accuracy:.2f}")
            except Exception as e:
                st.warning(f"Couldn't load existing model: {e}. Training a new one.")
                model = None
                
        if model is None:
            with st.spinner('Training model...'):
                model_result = train_model(df, features)
                if model_result:
                    model, accuracy, features = model_result
                    st.success(f"Model accuracy: {accuracy:.2f}")
                else:
                    st.error("Failed to train model")
                    return
        
        # Analysis tab
        with st.expander("📊 Data Analysis", expanded=False):
            st.subheader("Match Statistics")
            
            st.write("Team Win Rates")
            teams = pd.concat([match_data['Team_1'], match_data['Team_2']]).unique()
            win_rates = {}
            for team in teams:
                team_matches = match_data[(match_data['Team_1'] == team) | (match_data['Team_2'] == team)]
                team_wins = match_data[match_data['Winner'] == team].shape[0]
                win_rates[team] = team_wins / len(team_matches) if len(team_matches) > 0 else 0
            
            win_rate_df = pd.DataFrame({
                'Team': list(win_rates.keys()),
                'Win Rate': list(win_rates.values())
            }).sort_values(by='Win Rate', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Win Rate', y='Team', data=win_rate_df, ax=ax)
            plt.xlabel('Win Rate')
            plt.ylabel('Team')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.write("Toss Impact Analysis")
            toss_wins = df.groupby('Toss_Winner')['Team1_Win'].mean()
            fig, ax = plt.subplots(figsize=(10, 6))
            toss_wins.plot(kind='bar')
            plt.ylabel('Win Rate After Toss')
            plt.xlabel('Team')
            plt.title('Win Rate When Winning Toss')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Prediction section
        st.subheader("Match Prediction")
        
        teams = pd.concat([match_data['Team_1'].astype(str), match_data['Team_2'].astype(str)]).unique()
        venues = match_data['Venue'].unique()
        
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select Team 1", options=sorted(teams))
            venue = st.selectbox("Select Venue", options=sorted(venues))
        
        with col2:
            team2 = st.selectbox("Select Team 2", options=sorted(teams), index=1 if len(teams) > 1 else 0)
            toss_winner = st.selectbox("Select Toss Winner", options=[team1, team2])
            toss_decision = st.selectbox("Toss Decision", options=['Bat', 'Bowl'])
        
        if st.button("Predict Match Winner"):
            if team1 == team2:
                st.error("Please select different teams")
            elif toss_winner not in [team1, team2]:
                st.error("Toss winner must be one of the selected teams")
            else:
                with st.spinner('Predicting match outcome...'):
                    result = predict_match(model, features, team1, team2, venue, toss_winner, toss_decision)
                
                if result:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Prediction")
                        st.markdown(f"### 🏆 {result['prediction']} is predicted to win!")
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        teams_prob = [result['team1'], result['team2']]
                        probs = [result['team1_win_prob'], result['team2_win_prob']]
                        colors = ['#4CAF50' if teams_prob[i] == result['prediction'] else '#F44336' for i in range(2)]
                        
                        ax.bar(teams_prob, probs, color=colors)
                        ax.set_ylabel('Win Probability (%)')
                        ax.set_title('Match Win Probability')
                        
                        for i, v in enumerate(probs):
                            ax.text(i, v + 1, f"{v:.1f}%", ha='center')
                        
                        plt.ylim(0, 100)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Key Factors")
                        
                        feature_imp = result['feature_importances']
                        top_features = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:10])
                        
                        clean_names = {
                            'Team1_Batting_Strength': f"{team1} Batting",
                            'Team1_Bowling_Strength': f"{team1} Bowling",
                            'Team1_Allrounder_Strength': f"{team1} All-rounders",
                            'Team2_Batting_Strength': f"{team2} Batting",
                            'Team2_Bowling_Strength': f"{team2} Bowling",
                            'Team2_Allrounder_Strength': f"{team2} All-rounders",
                            'Team1_Venue_Advantage': f"{team1} Venue Advantage",
                            'Team2_Venue_Advantage': f"{team2} Venue Advantage",
                            'Team1_Toss_Advantage': f"{team1} Toss Advantage",
                            'Team2_Toss_Advantage': f"{team2} Toss Advantage",
                            'Toss_Decision_Bat': 'Toss Decision (Bat)'
                        }
                        
                        display_features = {}
                        for f, v in top_features.items():
                            display_features[clean_names.get(f, f.replace('_', ' ').title())] = v
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        y_pos = range(len(display_features))
                        
                        ax.barh(y_pos, list(display_features.values()))
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(list(display_features.keys()))
                        ax.set_xlabel('Importance')
                        ax.set_title('Top Factors Influencing Prediction')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with st.expander("📈 Detailed Analysis", expanded=False):
                        st.subheader("Team Comparison")
                        
                        team_metrics = {
                            'Batting': [result['features_values']['Team1_Batting_Strength'], 
                                       result['features_values']['Team2_Batting_Strength']],
                            'Bowling': [result['features_values']['Team1_Bowling_Strength'], 
                                      result['features_values']['Team2_Bowling_Strength']],
                            'All-rounders': [result['features_values']['Team1_Allrounder_Strength'], 
                                           result['features_values']['Team2_Allrounder_Strength']],
                            'Venue Advantage': [result['features_values']['Team1_Venue_Advantage'], 
                                              result['features_values']['Team2_Venue_Advantage']],
                            'Toss Advantage': [result['features_values']['Team1_Toss_Advantage'], 
                                             result['features_values']['Team2_Toss_Advantage']]
                        }
                        
                        categories = list(team_metrics.keys())
                        N = len(categories)
                        
                        angles = [n / float(N) * 2 * np.pi for n in range(N)]
                        angles += angles[:1]
                        
                        fig = plt.figure(figsize=(8, 8))
                        ax = plt.subplot(111, polar=True)
                        
                        plt.xticks(angles[:-1], categories, size=12)
                        
                        values1 = [team_metrics[cat][0] for cat in categories]
                        values1 += values1[:1]
                        ax.plot(angles, values1, linewidth=2, linestyle='solid', label=team1)
                        ax.fill(angles, values1, alpha=0.25)
                        
                        values2 = [team_metrics[cat][1] for cat in categories]
                        values2 += values2[:1]
                        ax.plot(angles, values2, linewidth=2, linestyle='solid', label=team2)
                        ax.fill(angles, values2, alpha=0.25)
                        
                        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                        plt.title('Team Strength Comparison', size=15)
                        st.pyplot(fig)
                else:
                    st.error("Prediction failed. Please check your inputs.")
    else:
        st.error("No match data available. Please upload match data files.")
        
        st.subheader("Upload Data Files")
        
        match_file = st.file_uploader("Upload match data CSV", type=["csv"])
        batsmen_file = st.file_uploader("Upload batsmen ratings CSV", type=["csv"])
        bowlers_file = st.file_uploader("Upload bowlers ratings CSV", type=["csv"])
        allrounders_file = st.file_uploader("Upload all-rounders ratings CSV", type=["csv"])
        
        if match_file is not None:
            with open('match_data.csv', 'wb') as f:
                f.write(match_file.getbuffer())
            
            if batsmen_file is not None:
                with open('batsmen_ratings.csv', 'wb') as f:
                    f.write(batsmen_file.getbuffer())
            
            if bowlers_file is not None:
                with open('bowlers_ratings.csv', 'wb') as f:
                    f.write(bowlers_file.getbuffer())
                
            if allrounders_file is not None:
                with open('allrounders_ratings.csv', 'wb') as f:
                    f.write(allrounders_file.getbuffer())
            
            st.success("Files uploaded successfully. Please reload the app.")

# Run the app
if __name__ == "__main__":
    main()