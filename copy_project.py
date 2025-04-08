# copy_project_files_recursive.py
import os

# --- Configuration ---
# !!! IMPORTANT: Make sure this path points to your project directory !!!
PROJECT_DIR = "/home/user/AdversarialAttacks/"

# List of base filenames we need the content of
# The script will search for these files within PROJECT_DIR and its subdirectories
# List of base filenames we need the content of
# The script will search for these files within PROJECT_DIR and its subdirectories
FILES_TO_FIND = [
    # Still missing critical files
    "models.py",
    "data.py",
    "tester.py",

    # Dependencies for CommonWeakness.py
    "CommonWeakness.py", # Keep this to ensure we get the right one
    "AdversarialInputBase.py",
    "utils.py", # Will find all instances (in data/, attacks/, attacks/AdversarialInput/, utils/)
    "ImageHandling.py", # Keep this

    # Potential location for data loader / init files
    "__init__.py", # Keep this, might find data/__init__.py, attacks/__init__.py etc.
    "NIPS17.py", # Guess for the NIPS loader

    # Potential dependency from traceback
    "CommonFigures.py",

    # Add back the original core attacks file just in case
    # "attacks.py", # Commented out for now as it found the wrong one last time

    # Base attack class if not in AdversarialInputBase.py
    # "base.py" # Another common name for base classes
]


# --- Script Logic ---
SEPARATOR = "\n" + "="*20 + " START OF FILE: {filepath} " + "="*20 + "\n"
END_SEPARATOR = "\n" + "="*20 + " END OF FILE: {filepath} " + "="*20 + "\n\n"

print("--- Starting recursive file copy script ---")
print(f"Searching for files within: {PROJECT_DIR}")
print(f"Looking for filenames: {', '.join(FILES_TO_FIND)}")
print("-" * 60)
# Add reminder for OpenCV installation
print("Reminder: If you encountered a 'cv2' error, install OpenCV using:")
print("pip install opencv-python")
print("-" * 60)


all_content = ""
found_files_map = {} # Store {filename: full_path}
missing_files = list(FILES_TO_FIND) # Start assuming all are missing

# Recursively walk through the directory
for dirpath, dirnames, filenames in os.walk(PROJECT_DIR):
    # Skip hidden directories like .git, .vscode, etc.
    # Also skip environment directories if they happen to be inside
    dirnames[:] = [d for d in dirnames if not d.startswith('.') and 'env' not in d and '__pycache__' not in d]

    for filename in filenames:
        if filename in FILES_TO_FIND:
            filepath = os.path.join(dirpath, filename)
            # Simple check to avoid adding the script itself if run from PROJECT_DIR
            if os.path.abspath(filepath) == os.path.abspath(__file__):
                continue

            if filename not in found_files_map: # Store the first one found
                 print(f"Found '{filename}' at: {filepath}")
                 found_files_map[filename] = filepath
                 if filename in missing_files:
                     missing_files.remove(filename)
            else:
                 # Optional: Notify if the same filename is found in multiple places
                 # print(f"  Note: Found another '{filename}' at {filepath} (using the first one found)")
                 pass


print("\n--- Reading Found Files ---")
# Read the content of the found files
for filename, filepath in found_files_map.items():
    print(f"Reading content of: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        all_content += SEPARATOR.format(filepath=filepath)
        all_content += content
        all_content += END_SEPARATOR.format(filepath=filepath)
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        all_content += SEPARATOR.format(filepath=filepath)
        all_content += f"# Error reading file: {e}"
        all_content += END_SEPARATOR.format(filepath=filepath)
        # Add back to missing if reading failed
        if filename not in missing_files:
            missing_files.append(f"{filename} (Error reading)")


print("\n--- Summary ---")
found_list = list(found_files_map.keys())
print(f"Found and attempted to read: {', '.join(found_list) if found_list else 'None'}")
if missing_files:
    # Add status for files that were searched for but never found
    for fname in list(missing_files): # Iterate over a copy
        if fname not in found_files_map and '(Error reading)' not in fname:
             missing_files.remove(fname)
             missing_files.append(f"{fname} (Not found in search)")
    print(f"Missing or unreadable: {', '.join(missing_files)}")

print("\n--- Combined File Content ---")
print("Please copy the text below this line (including the START/END markers) and paste it into the chat:")
print("-" * 70)
# Print the combined content to the console
print(all_content)
print("-" * 70)
print("--- End of Combined File Content ---")
