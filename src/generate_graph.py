import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import easyocr
import fitz  # PyMuPDF
from PIL import Image
import re

def process_folder_with_fixed_pixel_grid(folder_path, cell_size=(10, 10), threshold_fraction=0.2):
    """
    Processes all PNG images in a folder, dividing each image into fixed pixel-size grids
    and accumulating binary masks based on thresholded regions.

    Parameters:
    - folder_path: Path to the folder containing images.
    - cell_size: Tuple (cell_height, cell_width) defining the size of each grid cell in pixels.
    - threshold_fraction: Fraction of pixels in a grid that must be black/white to mark the grid.

    Returns:
    - accumulated_mask: A heatmap showing the accumulated mask across images.
    """
    accumulated_mask = None

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):  # Process only PNG files
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)
            if image is None:
                print(f"Could not read file: {file_path}")
                continue

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Define thresholds for "close to white" and "close to black"
            white_threshold = 255  # Adjust this threshold for white pixels
            black_threshold = 70   # Adjust this threshold for black pixels

            # Create masks for white and black pixels
            white_mask = cv2.inRange(gray, white_threshold, 255)  # Pixels >= white_threshold
            black_mask = cv2.inRange(gray, 0, black_threshold)    # Pixels <= black_threshold

            # Combine the masks
            binary_mask = cv2.bitwise_or(white_mask, black_mask)
            binary_mask = binary_mask // 255  # Normalize mask to 0 and 1

            # Get the dimensions of the binary mask
            img_height, img_width = binary_mask.shape
            cell_height, cell_width = cell_size

            # Calculate the number of rows and columns based on cell size
            grid_rows = img_height // cell_height
            grid_cols = img_width // cell_width

            # Create a grid-based mask
            grid_mask = np.zeros((grid_rows, grid_cols), dtype=np.float32)

            # Loop through each grid cell
            for i in range(grid_rows):
                for j in range(grid_cols):
                    # Calculate cell boundaries
                    start_row, end_row = i * cell_height, (i + 1) * cell_height
                    start_col, end_col = j * cell_width, (j + 1) * cell_width

                    # Extract the cell from the binary mask
                    cell = binary_mask[start_row:end_row, start_col:end_col]

                    # Mark the cell if the fraction of black/white pixels exceeds the threshold
                    if np.sum(cell) > (cell.shape[0] * cell.shape[1]) * threshold_fraction:
                        grid_mask[i, j] += 1

            # Initialize the accumulator on the first image
            if accumulated_mask is None:
                accumulated_mask = np.zeros_like(grid_mask, dtype=np.float32)

            # Accumulate the grid mask
            accumulated_mask += grid_mask

    return accumulated_mask

def scale_mask(accumulated_mask, method="log"):
    """
    Scales the accumulated mask for better visualization.
    Supported methods: "log", "normalize".
    """
    if method == "log":
        # Apply logarithmic scaling (add a small value to avoid log(0))
        scaled_mask = np.log1p(accumulated_mask)
    elif method == "normalize":
        # Normalize to range [0, 1]
        scaled_mask = (accumulated_mask - accumulated_mask.min()) / (
            accumulated_mask.max() - accumulated_mask.min()
        )
    else:
        raise ValueError("Unsupported scaling method. Use 'log' or 'normalize'.")
    return scaled_mask

def plot_heatmap(accumulated_mask, title="Heatmap"):
    # Scale the mask for better visualization
    scaled_mask = scale_mask(accumulated_mask, method="log")

    # Plot the accumulated grid-based mask as a heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(scaled_mask, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Scaled Values")
    plt.title(title)
    plt.axis("off")  # Remove axes for a cleaner look
    plt.show()

def extract_xg_value(text):
    """
    Extract the XG value from a string in the format: 'XX / YY XG'.
    Handles cases with '<' before the XG value.
    
    Parameters:
        text (str): The input text string.

    Returns:
        float: The extracted XG value.
    """
    match = re.search(r'/\s*([<]?\d*\.?\d+)\s*XG', text)
    if match:
        return float(match.group(1).replace('<', ''))
    else:
        raise ValueError(f"Unable to extract XG value from text: '{text}'")

def extract_attempts(text):
    """
    Extract the number of attempts from a string in the format: 'XX / YY XG'.
    
    Parameters:
        text (str): The input text string.

    Returns:
        int: The extracted number of attempts.
    """
    try:
        return int(text.split('/')[0].strip())
    except ValueError:
        raise ValueError(f"Unable to extract attempts from text: '{text}'")

def extract_stats_from_image(image_path):
    """
    Extract stats from an image containing text with shooting attempts and XG values.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        dict: A dictionary containing extracted stats for left, mid, and right positions.
    """
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    sorted_results = sorted(results, key=lambda x: x[0][0][0])

    stats_dict = {
        'left_attempts': None,
        'left_XG': None,
        'left_percent': None,
        'mid_attempts': None,
        'mid_XG': None,
        'mid_percent': None,
        'right_attempts': None,
        'right_XG': None,
        'right_percent': None
    }
    
    current_position = 'left'
    for detection in sorted_results:
        text = detection[1]
        try:
            if '/' in text and 'XG' in text:
                attempts = extract_attempts(text)
                xg_value = extract_xg_value(text)
                
                if current_position == 'left':
                    stats_dict['left_attempts'] = attempts
                    stats_dict['left_XG'] = xg_value
                    current_position = 'mid'
                elif current_position == 'mid':
                    stats_dict['mid_attempts'] = attempts
                    stats_dict['mid_XG'] = xg_value
                    current_position = 'right'
                else:
                    stats_dict['right_attempts'] = attempts
                    stats_dict['right_XG'] = xg_value
            
            elif '%' in text:
                percent_value = int(text.replace('%', '').strip())
                if stats_dict['left_percent'] is None:
                    stats_dict['left_percent'] = percent_value
                elif stats_dict['mid_percent'] is None:
                    stats_dict['mid_percent'] = percent_value
                else:
                    stats_dict['right_percent'] = percent_value
        
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping text due to parsing error: {text}. Error: {e}")
    
    return stats_dict


def process_folder_attack_side(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            stats_dict = extract_stats_from_image(image_path)
            stats_dict['team_image'] = filename
            data.append(stats_dict)
    
    df = pd.DataFrame(data)
    cols = ['team_image'] + [col for col in df.columns if col != 'team_image']
    df = df[cols]
    
    columns_to_average = ['left_attempts', 'left_XG', 'mid_attempts', 'mid_XG', 'right_attempts', 'right_XG']

    # Calculate the mean of the specified columns
    mean_values = df[columns_to_average].mean()

    # Alternatively, if you want the mean as a DataFrame:
    mean_df = mean_values.to_frame().T  # Transpose to keep it as a single-row DataFrame
    
    return mean_df

def plot_attack_side(df, title="Attack Side"):
    # Calculate total XG and percentages
    total_XG = df['left_XG'].sum() + df['mid_XG'].sum() + df['right_XG'].sum()
    percentage_left = (df['left_XG'].sum() / total_XG) * 100
    percentage_mid = (df['mid_XG'].sum() / total_XG) * 100
    percentage_right = (df['right_XG'].sum() / total_XG) * 100

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Setup the plot
    x_positions = [0.25, 0.5, 0.75]
    arrow_colors = ['black', 'black', 'black']
    # Calculate total attempts
    total_attempts = df['left_attempts'].sum() + df['mid_attempts'].sum() + df['right_attempts'].sum()

    # Scale the arrow length and width based on number of attempts
    arrow_lengths = [df['left_attempts'].values[0] / total_attempts * 0.75, 
                     df['mid_attempts'].values[0] / total_attempts * 0.75, 
                     df['right_attempts'].values[0] / total_attempts * 0.75]

    # Scale the arrow width (thickness) based on the number of attempts
    arrow_widths = [df['left_attempts'].values[0] / total_attempts * 10, 
                    df['mid_attempts'].values[0] / total_attempts * 10, 
                    df['right_attempts'].values[0] / total_attempts * 10]

    # Draw arrows with length and width proportional to the number of attempts
    for i in range(3):
        ax.annotate('', 
                    xy=(x_positions[i], 0.2 + arrow_lengths[i]), 
                    xytext=(x_positions[i], 0.2),
                    arrowprops=dict(facecolor=arrow_colors[i], 
                                    edgecolor='black', 
                                    lw=arrow_widths[i],  # Adjust arrow width
                                    shrink=0.05))

    # Add text annotations
    attempts = [df['left_attempts'].values[0], df['mid_attempts'].values[0], df['right_attempts'].values[0]]
    XGs = [df['left_XG'].values[0], df['mid_XG'].values[0], df['right_XG'].values[0]]
    percentages = [percentage_left, percentage_mid, percentage_right]

    for i in range(3):
        ax.text(x_positions[i], 0.05, 
                f"{int(attempts[i])} / {XGs[i]:.2f} xG\n{percentages[i]:.0f}%",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Style the plot
    ax.set_facecolor('#ccffcc')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add rotated goal area (now facing up)
    ax.plot([0.25, 0.75], [0.7, 0.7], color='white', lw=3)  # Bottom horizontal line moved up
    ax.plot([0.25, 0.25], [0.7, 1], color='white', lw=3)  # Left vertical line moved up
    ax.plot([0.75, 0.75], [0.7, 1], color='white', lw=3)  # Right vertical line moved up


    plt.title(title, fontsize=14, fontweight='bold')
    plt.show()


def extract_image_data(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Split into top and bottom sections
    height = image.shape[0]
    width = image.shape[1]
    top_section = image[0:int(height*0.37), :]  # Table section
    bottom_section = image[int(height*0.3):, int(width*0.1):int(width*0.9)]  # Graph section

    # Extract table data using OCR
    reader = easyocr.Reader(['en'])
    table_text = reader.readtext(top_section)

    # Process graph by color
    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(bottom_section, cv2.COLOR_BGR2HSV)

    # Black line mask
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 100])
    black_mask = cv2.inRange(hsv, black_lower, black_upper)

    # Blue line mask
    blue_lower = np.array([0, 50, 150])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # Extract numbers for each line
    black_numbers = reader.readtext(black_mask)
    blue_numbers = reader.readtext(blue_mask)
    
    # Sort numbers by x-coordinate
    black_numbers = sorted(black_numbers, key=lambda x: x[0][0][0])
    blue_numbers = sorted(blue_numbers, key=lambda x: x[0][0][0])
    
    return {
        'table_data': table_text,
        'black_line_values': [num[1] for num in black_numbers if num[1].isdigit()],
        'blue_line_values': [num[1] for num in blue_numbers if num[1].isdigit()]
    }

def process_table_data(table_text, black_line_values, blue_line_values, team_name):
    # Initialize dictionary for storing data dynamically
    df_data = {}

    # Flags to detect when we're processing a team row
    current_team = None

    # Column labels for stats (can be adjusted based on structure)
    columns = ['Total', '1st half', '2nd half']
    current_column_index = 0  # This will track which column we're filling

    # Process each OCR result
    for detection in table_text:
        text = detection[1]  # OCR text result
        
        # Skip rows that are non-statistics (like "Average formation line, m", "Ist half", etc.)
        if not any(char.isdigit() for char in text) and len(text.split()) > 1:
            # This is likely a team name (no numbers, multiple words)
            current_team = text
            if current_team not in df_data:
                # Initialize columns for this team if it's not in the data yet
                df_data[current_team] = {col: None for col in columns}
                current_column_index = 0  # Reset column index for each team
        elif '.' in text and current_team:  # If the text is a stat (contains a decimal)
            try:
                # Convert text to a number (stat value)
                number = float(text)
                
                # Ensure we assign values only to the correct column
                if current_column_index < len(columns):
                    df_data[current_team][columns[current_column_index]] = number
                    current_column_index += 1
            except ValueError:
                continue  # Skip if the text can't be converted to a number

    # Remove any teams that have incomplete data (no 'Total', '1st half', or '2nd half' values)
    df_data = {team: stats for team, stats in df_data.items() if all(v is not None for v in stats.values())}
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(df_data, orient='index')
    
    if team_name in df.iloc[0].index:
        # Keep only the first row and black_line_values
        selected_row = df.iloc[0]
        selected_values = black_line_values
    else:
        # Keep the second row and blue_line_values
        selected_row = df.iloc[1]
        selected_values = blue_line_values

    # Create the column_data dictionary with the selected values
    column_data = {
        '1-15': selected_values[0],
        '16-30': selected_values[1],
        '31-45+': selected_values[2],
        '46-60': selected_values[3],
        '61-75': selected_values[4],
        '76-90+': selected_values[5]
    }

    # Add the new columns to the DataFrame
    for col_name, values in column_data.items():
        # Assign the two lists (one for each team) to the new columns
        selected_row[col_name] = values

    # Filter the DataFrame to get only the row for the specified team
    if selected_row.any():
        return selected_row
    else:
        print(f"Team '{team_name}' not found in the table.")
        return None

def avg_formation_from_folder(folder_path, team_name):
    # Initialize an empty list to store the data for the specified team
    team_data = []

    # Loop through all image files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Process image files only
            image_path = os.path.join(folder_path, file_name)
            
            # Extract image data (table and line values)
            info = extract_image_data(image_path)
            
            # Process table data and extract only the specified team's row
            team_row = process_table_data(info['table_data'], info['black_line_values'], info['blue_line_values'], team_name)
            
            if team_row is not None:
                team_data.append(team_row)
    
    # Combine the results into a single DataFrame if any data was found
    if team_data:
        combined_data = pd.concat(team_data, axis=0)
        combined_data = pd.DataFrame(combined_data).reset_index()
        combined_data[0] = combined_data[0].astype(int)
        return combined_data.groupby('index')[0].mean().reset_index()
    else:
        print(f"No data found for team '{team_name}' in the folder.")
        return None

def plot_avg_formation(result_df, team_name):
    # Set style for better aesthetics
    plt.style.use('ggplot')

    # Data from the DataFrame
    result_df = pd.DataFrame(result_df)
    result_df = result_df.set_index('index').T
    time_intervals = ['1-15', '16-30', '31-45+', '46-60', '61-75', '76-90+']
    values = result_df[time_intervals].values[0]
    total = result_df['Total'].values[0]
    first_half = result_df['1st half'].values[0]
    second_half = result_df['2nd half'].values[0]

    # Create figure with two subplots - one for graph, one for table
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                  gridspec_kw={'height_ratios': [3, 1]})

    # Plot line with improved styling
    line = ax1.plot(time_intervals, values, marker='o', color='#1f77b4', 
                    linewidth=2, markersize=8, label='Indiana Hoosiers')

    # Add value labels to points
    for i, value in enumerate(values):
        ax1.annotate(f'{value:.1f}',
                    xy=(i, value),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')

    # Customize grid and appearance
    ax1.grid(True, linestyle='--', alpha=0.7)
    padding = (max(values) - min(values)) * 0.1
    ax1.set_ylim(min(values)-padding, max(values)+padding)
    ax1.set_title(f'{team_name} Average Formation Line, m', pad=20, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Intervals', fontsize=12)
    ax1.set_ylabel('Meters', fontsize=12)

    # Create table data
    table_data = [['Total', f'{total:.1f}'],
                 ['1st Half', f'{first_half:.1f}'],
                 ['2nd Half', f'{second_half:.1f}']]

    # Create and customize table
    table = ax2.table(cellText=table_data,
                     colWidths=[0.5, 0.5],
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Hide axes for table subplot
    ax2.axis('off')

    plt.tight_layout()
    plt.show()
