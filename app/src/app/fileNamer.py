import os

def rename_files_in_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Loop through files and rename them
    for index, file_name in enumerate(files, start=1):
        # Create the new file name
        file_extension = os.path.splitext(file_name)[1]
        new_name = f"{index}{file_extension}"
        
        # Get the full paths for old and new file names
        old_file = os.path.join(folder_path, file_name)
        new_file = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {old_file} -> {new_file}")

# Specify the path to your folder
folder_path = "/home/saahas/Downloads/GitHub/Ramblin-Hacks/app/processed_data"
rename_files_in_folder(folder_path)
