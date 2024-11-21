import pandas as pd
def get_team_data(team_data_path, team_name):
    team_data = pd.read_csv(team_data_path)
    team_data['team'] = team_data['team'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    matching_teams = [team for team in team_data['team'].unique() if team_name.lower() in team.lower()]
    
    # Group by team and aggregate useful metrics
    team_summary = team_data.groupby('team').agg({
        'goals': 'sum',
        'xg': 'mean',
        'shots': 'sum',
        'sot': 'sum',
        'pass_success_rate': 'mean',
        'possession': 'mean',
        'losses': 'sum',
        'recoveries': 'sum',
        'tot_duels': 'sum',
        'tot_duels_won': 'sum',
        'tot_duels_win_rate': 'mean',
        'corners': 'sum',
        'fouls': 'sum',
        'yellows': 'sum',
        'reds': 'sum',
        'shots_against': 'sum',
        'goals_against': 'sum',
        'def_duels': 'sum',
        'interceptions': 'sum',
        'clearences': 'sum',
        'offsides': 'sum'
    }).reset_index()

    return team_summary[team_summary['team'] == matching_teams[0]]

def generate_prompt(team_summary, home_attack_side, avg_formation):
    team_data = team_summary.iloc[0]
    # Prepare the team statistics as a string
    team_stats_str = f"""
    - Team: {team_data['team']}
    - Goals: {team_data['goals']}
    - Expected Goals (xG): {team_data['xg']}
    - Shots: {team_data['shots']}
    - Shots on Target (SOT): {team_data['sot']}
    - Pass Success Rate: {team_data['pass_success_rate']}%
    - Possession: {team_data['possession']}%
    - Losses: {team_data['losses']}
    - Recoveries: {team_data['recoveries']}
    - Total Duels: {team_data['tot_duels']}
    - Corners: {team_data['corners']}
    - Fouls: {team_data['fouls']}
    - Yellow Cards: {team_data['yellows']}
    - Red Cards: {team_data['reds']}
    - Shots Against: {team_data['shots_against']}
    - Goals Against: {team_data['goals_against']}
    - Defensive Duels: {team_data['def_duels']}
    - Interceptions: {team_data['interceptions']}
    - Clearances: {team_data['clearences']}
    - Offsides: {team_data['offsides']}
    """

    # Attack side stats
    attack_side_str = f"""
    Attack Side Breakdown:
    - Left Side: {home_attack_side['left_attempts'].values[0]} attempts, xG: {home_attack_side['left_XG'].values[0]}
    - Middle: {home_attack_side['mid_attempts'].values[0]} attempts, xG: {home_attack_side['mid_XG'].values[0]}
    - Right Side: {home_attack_side['right_attempts'].values[0]} attempts, xG: {home_attack_side['right_XG'].values[0]}
    """

    # Average formation stats
    formation_str = f"""
    Average Formation Possession by Time Period:
    - Total: {avg_formation[avg_formation['index']=='Total'][0].values[0]}%
    """
    
    # Combine all the parts into one prompt
    prompt = f"""
    Analyze the strengths and weaknesses of the opposing team based on the following statistics:

    Team Stats:
    {team_stats_str}

    Attack Side Breakdown:
    {attack_side_str}

    Average Formation Possession:
    {formation_str}

    Provide insights on the team's weaknesses and strategies for exploiting them.
    """
    
    system_message = "You are a soccer head coach."
    
    return prompt, system_message
