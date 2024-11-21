import os

def find_team_folder(team_name, folder_path):
    """
    Search for the first folder containing the first part of the team name within the given folder path.

    Parameters:
        team_name (str): The full name of the team (e.g., "Penn State").
        folder_path (str): The path to the folder containing match reports.

    Returns:
        str: The full path to the first matching folder.
        None: If no matching folder is found.
    """
    # Extract the first part of the team name
    search_text = team_name.split()[0].lower()  # Convert to lowercase for case-insensitive matching

    # Check if the folder_path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path '{folder_path}' does not exist.")

    # Traverse all folders in the given path
    for folder_name in os.listdir(folder_path):
        if search_text in folder_name.lower():  # Case-insensitive matching
            # Return the full path to the first matching folder
            return os.path.join(folder_path, folder_name)

    # Return None if no match is found
    return None

def remove_pdfs_with_northwestern(folder_path):
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Iterate through each file in the directory
    for file in files:
        # Check if the file is a PDF and contains 'northwestern' in the name
        if file.endswith('.pdf') and 'northwestern' in file.lower():
            # Construct the full file path
            file_path = os.path.join(folder_path, file)
            
            # Remove the file
            os.remove(file_path)
            print(f"Removed: {file_path}")
