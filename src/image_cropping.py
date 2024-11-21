
def crop_and_save_shooting_side_and_map_from_folder(folder_path, team_name):
    """
    Loops through all PDFs in a folder, crops images based on the specified team name's location,
    and saves the cropped images in specific folders with indexed names.
    
    Args:
    - folder_path (str): Path to the folder containing PDF files.
    - team_name (str): The team name to look for in the PDF file names.
    
    Returns:
    - None
    """

    # Create folders for output if they don't exist
    home_team_folder = os.path.join(folder_path, "home_team")
    opp_team_folder = os.path.join(folder_path, "opp_team")

    home_shooting_side_folder = os.path.join(home_team_folder, "shooting_map")
    opp_shooting_side_folder = os.path.join(opp_team_folder, "shooting_map")

    home_shooting_map_folder = os.path.join(home_team_folder, "shooting_side")
    opp_shooting_map_folder = os.path.join(opp_team_folder, "shooting_side")

    # Create the directories
    os.makedirs(home_shooting_side_folder, exist_ok=True)
    os.makedirs(opp_shooting_side_folder, exist_ok=True)
    os.makedirs(home_shooting_map_folder, exist_ok=True)
    os.makedirs(opp_shooting_map_folder, exist_ok=True)

    # Initialize image index
    image_index = 1

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):  # Process only PDF files
            pdf_path = os.path.join(folder_path, file_name)
            # Open the PDF file
            pdf_document = fitz.open(pdf_path)
            page_count = pdf_document.page_count
            page_mismatch = page_count - 24

            # Extract teams from the file name
            team_part = file_name.split("-")
            team_1 = team_part[0].strip()
            team_2 = team_part[1].strip()

            # Determine cropping logic based on the team name location
            if team_name in team_1:
                # Case 1: Specified team appears first
                crops = [
                    {"page_number": 10+page_mismatch, "crop_rectangle": (10, 350, 285, 565)},  # Shooting side home
                    {"page_number": 4, "crop_rectangle": (445, 600, 577, 707)},  # Shooting map home
                    {"page_number": 11+page_mismatch, "crop_rectangle": (10, 350, 285, 565)},  # Shooting side opp
                    {"page_number": 4, "crop_rectangle": (308, 600, 441, 707)},  # Shooting map opp
                ]
                other_team_name = team_2
            elif team_name in team_2:
                # Case 2: Specified team appears second
                crops = [
                    {"page_number": 11+page_mismatch, "crop_rectangle": (10, 350, 285, 565)},  # Shooting side home
                    {"page_number": 4, "crop_rectangle": (308, 600, 441, 707)},  # Shooting map home
                    {"page_number": 10+page_mismatch, "crop_rectangle": (10, 350, 285, 565)},  # Shooting side opp
                    {"page_number": 4, "crop_rectangle": (445, 600, 577, 707)},  # Shooting map opp
                ]
                other_team_name = team_1
            else:
                print(f"Team '{team_name}' not found in '{file_name}', skipping.")
                continue

            # Define desired DPI for higher resolution
            dpi = 300
            scale = dpi / 72  # Default is 72 DPI
            matrix = fitz.Matrix(scale, scale)  # Scale the rendering

            for i, crop in enumerate(crops, start=1):
                # Load the specified page
                page_number = crop["page_number"]
                page = pdf_document[page_number]

                # Render the page with the specified resolution
                pix = page.get_pixmap(matrix=matrix)

                # Convert the image into a PIL Image object
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Scale the crop rectangle to match the higher DPI image
                crop_rectangle = tuple(int(coord * scale) for coord in crop["crop_rectangle"])
                cropped_image = image.crop(crop_rectangle)

                # Define output folder and filename
                if i == 1:  # First crop is the shooting side for home team
                    output_folder = home_shooting_side_folder
                    output_filename = f"shooting_map_{image_index}_{team_name}_against_{other_team_name}.png"
                elif i == 2:  # Second crop is the shooting map for home team
                    output_folder = home_shooting_map_folder
                    output_filename = f"shooting_side_{image_index}_{team_name}_against_{other_team_name}.png"
                elif i ==3: # Third crop is the shooting side for opp team
                    output_folder = opp_shooting_side_folder
                    output_filename = f"shooting_map_{image_index}_{other_team_name}_against_{team_name}.png"
                elif i == 4:  # Fourth crop is the shooting map for opp team
                    output_folder = opp_shooting_map_folder
                    output_filename = f"shooting_side_{image_index}_{other_team_name}_against_{team_name}.png"

                output_path = os.path.join(output_folder, output_filename)

                # Save the cropped image
                cropped_image.save(output_path)
                print(f"Cropped image saved as {output_path}")

            # Increment the image index
            image_index += 1

    print("All images processed and saved.")
    
import os
import fitz  # PyMuPDF
from PIL import Image

def crop_and_save_avg_formation_from_folder(folder_path, team_name):
    # Create folders for output if they don't exist
    home_team_folder = os.path.join(folder_path, "home_team")
    avg_formation_folder = os.path.join(home_team_folder, "avg_formation")
    os.makedirs(avg_formation_folder, exist_ok=True)

    # Initialize image index
    image_index = 1

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):  # Process only PDF files
            pdf_path = os.path.join(folder_path, file_name)
            
            # Open the PDF file
            pdf_document = fitz.open(pdf_path)

            # Extract teams from the file name
            team_part = file_name.split("-")
            team_1 = team_part[0].strip()
            team_2 = team_part[1].strip()

            # Proceed only if the specified team is in the file name
            if team_name in team_1 or team_name in team_2:
                # Define desired DPI for higher resolution
                dpi = 300
                scale = dpi / 72  # Default is 72 DPI
                matrix = fitz.Matrix(scale, scale)  # Scale the rendering

                # Specify the page number (0-indexed)
                page_number = 3  # Change to your desired page number
                page = pdf_document[page_number]

                # Render the page with the specified resolution
                pix = page.get_pixmap(matrix=matrix)

                # Convert the image into a PIL Image object
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Define the crop rectangle (left, upper, right, lower) in pixels
                crop_rectangle = (5 * scale, 570 * scale, 200 * scale, 755 * scale)
                cropped_image = image.crop(crop_rectangle)

                # Define output filename based on the team names and image index
                output_filename = f"avg_formation_{image_index}_{team_1}_{team_2}.png"
                output_path = os.path.join(avg_formation_folder, output_filename)

                # Save the cropped image
                cropped_image.save(output_path)
                print(f"Cropped image saved as {output_path}")

                # Increment the image index
                image_index += 1
            else:
                print(f"Team '{team_name}' not found in '{file_name}', skipping.")

    print("All images processed and saved.")
