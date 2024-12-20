# %% code for crating script list
import os
import ast
from collections import defaultdict

# Specify the folder path and output markdown file
folder_path = "C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen"
output_md_file = "script_list.md"

# Function to extract function names from a Python script
def extract_functions(file_path):
    functions = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read())
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    except Exception as e:
        functions.append(f"Error reading file: {e}")
    return functions

# Group scripts and their functions by sub-folder
scripts_by_folder = defaultdict(list)

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, folder_path)
            folder_name = os.path.relpath(root, folder_path)  # Sub-folder name relative to the base folder
            functions = extract_functions(file_path)
            scripts_by_folder[folder_name].append((relative_path, functions))

# Write the grouped scripts and functions to a Markdown file
with open(output_md_file, "w", encoding="utf-8") as md_file:
    md_file.write("# List of Python Scripts Grouped by Sub-folder\n\n")
    md_file.write(f"Folder scanned: `{folder_path}`\n\n")
    
    for folder, scripts in sorted(scripts_by_folder.items()):
        md_file.write(f"## Folder: `{folder}`\n\n")
        for script, functions in scripts:
            md_file.write(f"### `{script}`\n\n")
            if functions:
                md_file.write("#### Functions\n")
                for func in functions:
                    md_file.write(f"- `{func}`\n")
            else:
                md_file.write("No functions found in this script.\n")
            md_file.write("\n")

print(f"Markdown file '{output_md_file}' has been created with the scripts grouped by sub-folder.")
