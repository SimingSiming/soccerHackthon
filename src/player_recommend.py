import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mplsoccer import VerticalPitch

def calculate_weighted_scores(data, team_name, topk=3):
    """
    Calculate weighted scores for metrics based on given weights.

    :param data: DataFrame containing player statistics
    :param total_weights: Dictionary of weights for metrics
    :param metric_types: List of metric types (defensive, attacking, passing)
    :param team_name: Team name to filter
    :param topk: Number of top players to return
    :return: DataFrame with computed weighted scores
    """
    metric_types = ["defensive", "attacking","passing"]
    total_weights = {
        "defensive": {
            'aerial_duels_won': 1.2,
            'interceptions':1.8,
            'recoveries': 1.5,
            'defensive_duels_won':2,
            'clearances': 1.8
        },
        
        "attacking": {
            'goals': 2.0,
            'assists': 1.8,
            'xg': 1.5,
            'successful_dribbles': 1.2,
            'xa': 1.2
        },
        
        "passing" : {
            'passes_into_final_third': 2.0,
            'passes_into_box': 1.8,
            'forward_passes': 1.5,
            'crosses': 1.5
        }
    }
    weighted_scores = data.copy()

    # Calculate weighted scores for each metric
    for metric_type, weights in total_weights.items():
        for metric, weight in weights.items():
            col_name = f"{metric_type}_{metric}_weighted"
            weighted_scores[col_name] = (weighted_scores[metric] / weighted_scores['Minutes played']) * 90 * weight

        # Sum all weighted metrics to compute the total score for this metric type
        total_col = f"{metric_type}_total_weighted_score"
        weighted_scores[total_col] = weighted_scores[
            [f"{metric_type}_{metric}_weighted" for metric in weights.keys()]
        ].sum(axis=1)
        
    score_columns = [col for col in weighted_scores.columns if "total_weighted_score" in col]
    
    # Group by player_name, team, and position
    grouped_scores = weighted_scores.groupby(['player_name', 'team'])[score_columns].mean().reset_index()

    for metric_type in metric_types:
        grouped_scores[f"{metric_type}_percentile"] = grouped_scores[f"{metric_type}_total_weighted_score"].rank(pct = True)
    
    grouped_scores = grouped_scores[grouped_scores.team == team_name]
    # Return the top players by total score for each metric type
    all_topk = []
    for metric_type in metric_types:
        topk_scores = grouped_scores.sort_values(
            by=f"{metric_type}_total_weighted_score", ascending=False
        ).head(topk)
        topk_scores["aspect"] = metric_type
        all_topk.append(topk_scores)

    # Combine all top-k results
    final_topk_scores = pd.concat(all_topk, ignore_index=True)

    return final_topk_scores

# Process to determine most played position
def get_most_played_position(data):
    position_dict = {
    "AMF": ["AMF", "LAMF", "RAMF"],
    "CF": ["CF"], 
    "GK": ["GK"],
    "CB": ["CB", "LCB", "LCBB", "RCB", "RCBB", "LCB3", "RCB3"],
    "DMF": ["DMF", "LDMF", "RDMF"],
    "LB": ["LB", "LB5"], 
    "RB": ["RB", "RB5"], 
    "LW": ["LW", "LWB", "LWF"], 
    "RW": ["RW", "RWB", "RWF"], 
    "LCM": ["LCM", "LCMF3", "LCMF"],
    "RCM": ["RCM", "RCMF3", "RCMF"]
    }
    position_dict = {pos: key for key, values in position_dict.items() for pos in values}
    
    # Create a column for each position's appearance count
    data['Position_list'] = data['Position'].str.split(', ')  # Split multiple positions into a list
    data = data[data.Position != "0"].copy()
    # Expand the positions into multiple rows
    exploded = data.explode('Position_list')

    # Group by player and team, and count each position
    position_counts = exploded.groupby(['player_name', 'team', 'Position_list'])['Minutes played'].sum().reset_index()

    # Determine the position with the highest total minutes for each player-team
    most_played_position = position_counts.loc[
        position_counts.groupby(['player_name', 'team'])['Minutes played'].idxmax()
    ]

    # Merge back the most played position into the original DataFrame
    merged_data = data.merge(
        most_played_position[['player_name', 'team', 'Position_list']],
        on=['player_name', 'team'],
        how='left'
    )

    # Rename the column for clarity
    merged_data = merged_data.rename(columns={'Position_list_y': 'Most_played_position'})
    final_data = merged_data[["player_name", "team", 'Most_played_position']].copy()
    
    # Map the most played position using the dictionary
    final_data['Most_played_position'] = final_data['Most_played_position'].map(position_dict)

    final_data.drop_duplicates(inplace = True)
    return final_data

def get_opponent(other_crucial_player, nu_crucial_player): 
    against_dict = {
    "AMF": "DMF",
    "CF": "CB", 
    "GK": "GK", 
    "CB":"CF",
    "DMF": "AMF",
    "LB": "RW",
    "RW": "LB",
    "RB": "LW",
    "LW": "RB",
    "LCM": "RCM", 
    "RCM": "LCM"
    }
    
    match_dict = {
        "defensive": "attacking", 
        "attacking": "defensive", 
        "passing": "passing"
    }
    # Prepare a result DataFrame
    matches = []
    
    # Iterate through crucial players
    for _, player in other_crucial_player.iterrows():
        player_position = player['Most_played_position']
        metric_type = player["aspect"]
        matching_position = against_dict.get(player_position)
        matching_metric_type = match_dict.get(metric_type)
        
        # Filter opponents with the matching position and find the best match
        position_opponents = nu_crucial_player[nu_crucial_player['Most_played_position'] == matching_position]

        # Find the opponent with the highest opposing score
        best_opponent = position_opponents.loc[
            position_opponents[f"{matching_metric_type}_total_weighted_score"].idxmax()
        ]

        # Add the match details
        matches.append({
            "Crucial Player Name": player['player_name'],
            "Crucial Player Team": player['team'],
            "Opponent Player Name": best_opponent['player_name'],
            "Opponent Player Team": best_opponent['team'],
            "Opponent Player Position": best_opponent['Most_played_position'] 
        })
    
    # Convert the matches into a DataFrame
    matches_df = pd.DataFrame(matches)
    return matches_df


# Updated function to plot a soccer pitch based on a DataFrame of players
def plot_pitch_from_dataframe(df, logo_path):
    """
    Plot a soccer pitch with player indicators based on a DataFrame.

    :param df: DataFrame containing columns:
        'player1', 'player2', 'nu_scores', 'other_scores',
        'nu_position', 'other_position'.
    """
    # Create the pitch
    pitch = VerticalPitch(pitch_color='grass', line_color='white', half=False)
    fig, ax = pitch.draw(figsize=(12, 8))
    

    logo_img = mpimg.imread(logo_path)
    ax.imshow(logo_img, extent=[30, 50, 50, 70], zorder=10, alpha = 0.7)  

    # Position mapping for player1 positions
    position_mapping = {
        "GK": (40, 0),
        "CB": (40, 25),
        "LB": (10, 20),
        "RB": (70, 20),
        "DMF": (40, 50),
        "AMF": (40, 70),
        "CF": (40, 100),
        "LW": (10, 90),
        "RW": (70, 90),
        "LCM": (10, 60),
        "RCM": (70, 60)
    }

    # Keep track of plotted positions to avoid duplicates
    plotted_positions = set()
    
    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        nu_player = row['Opponent Player Name']
        other_player = row['Crucial Player Name']
        nu_scores = (
            f"{row['passing_percentile_y'] * 100:.1f}%",
            f"{row['attacking_percentile_y'] * 100:.1f}%",
            f"{row['defensive_percentile_y'] * 100:.1f}%"
        )
        other_scores = (
            f"{row['passing_percentile_x'] * 100:.1f}%",
            f"{row['attacking_percentile_x'] * 100:.1f}%",
            f"{row['defensive_percentile_x'] * 100:.1f}%"
        )
        nu_position = row['Opponent Player Position']
        
        # Get positions
        nu_pos = position_mapping.get(nu_position)
        other_pos = (nu_pos[0], nu_pos[1] + 5)  # Offset for player2
        
        # Plot player1 if not already plotted
        if nu_pos not in plotted_positions:
            ax.scatter(*nu_pos, color='purple', s=200)
            ax.annotate(
                f"P: {nu_scores[0]}\nA: {nu_scores[1]}\nD: {nu_scores[2]}",
                xy=nu_pos,
                xytext=(nu_pos[0] + 3, nu_pos[1]-3),
                color='purple',
                fontsize=8
            )
            ax.annotate(
                nu_player,
                xy=nu_pos,
                xytext=(nu_pos[0] - 3, nu_pos[1] - 7),
                color='black',
                fontsize=10,
                weight='bold'
            )
            plotted_positions.add(nu_pos)

            
            ax.scatter(*other_pos, color='red', s=200, marker='v')
            ax.annotate(
                f"P: {other_scores[0]}\nA: {other_scores[1]}\nD: {other_scores[2]}",
                xy=other_pos,
                xytext=(other_pos[0] + 3, other_pos[1]),
                color='red',
                fontsize=8
            )
            ax.annotate(
                other_player,
                xy=other_pos,
                xytext=(other_pos[0] - 3, other_pos[1] + 7),
                color='black',
                fontsize=10,
                weight='bold'
            )
    
    # Add legend and title
    plt.title("Key Player Match-Up", fontsize=16)
    
    # Display the pitch
    plt.show()