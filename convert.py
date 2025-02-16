from tqdm import tqdm
import ast  # To safely convert string list to a Python list
# Input and output file paths
input_file = "prediction2.txt"
output_file = "prediction3.txt"


# Read from the input file and write to the output file
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in tqdm(infile):
        parts = line.strip().split(" ", 1)  # Split only on the first space
        if len(parts) == 2:
            try:
                new_id = int(parts[0])  # Increment ID
                
                # Convert the list from string to a real list and increment each element
                original_list = ast.literal_eval(parts[1])  # Convert string to list
                updated_list = [x + 1 for x in original_list]  # Increment each number
                
                # Convert list to a string format without spaces (e.g., [6,5,4,2,3])
                formatted_list = "[" + ",".join(map(str, updated_list)) + "]"
                
                # Write updated data
                outfile.write(f"{new_id} {formatted_list}\n")
            except (ValueError, SyntaxError):
                print(f"Skipping invalid line: {line.strip()}")

print(f"Processed data saved to {output_file}")

