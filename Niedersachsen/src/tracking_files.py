# %%
import re
import os

def used_and_saved_files(script_paths, output_document):
    with open(output_document, 'w') as output_file:
        for script_path in script_paths:
            # Read the script file
            with open(script_path, 'r') as file:
                script_content = file.read()
            
            # Regular expression to find file paths
            file_path_pattern = re.compile(r'["\'](.*?\.shp|.*?\.pkl|.*?\.csv|.*?\.json|.*?\.txt)["\']')
            
            # Regular expression to find function definitions
            function_pattern = re.compile(r'def\s+(\w+)\s*\(.*?\)\s*:')
            
            # Find all function definitions and their positions
            functions = [(match.start(), match.group(1)) for match in function_pattern.finditer(script_content)]
            
            # Find all file paths and their positions
            file_paths = [(match.start(), match.group(1)) for match in file_path_pattern.finditer(script_content)]
            
            # Associate file paths with functions
            file_paths_with_context = []
            for file_pos, file_path in file_paths:
                function_name = None
                for func_pos, func_name in functions:
                    if func_pos < file_pos:
                        function_name = func_name
                    else:
                        break
                file_paths_with_context.append((file_path, function_name))
            
            # Write the collected file paths with context to the document
            output_file.write(f"File paths collected from script: {script_path}\n\n")
            for file_path, function_name in file_paths_with_context:
                context_info = f" in function '{function_name}'" if function_name else " at the top level"
                output_file.write(f"{file_path} used or created{context_info}\n")
            output_file.write("\n")
    
    print(f"Collected file paths with context have been written to {output_document}")

# Example usage
script_paths = ['src/data/dataloading.py', 'src/data/eca.py']
output_document = 'notes/used_and_saved_files.txt'
used_and_saved_files(script_paths, output_document)