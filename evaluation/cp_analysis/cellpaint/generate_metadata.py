import os
import json

def generate_metadata(folder_path, output_file):
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Filter to only include image files
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    # Open the output file for writing
    with open(output_file, 'w') as f_out:
        for image_file in image_files:
            # Create the metadata dictionary
            metadata = {
                "file_name": image_file,
                "text": "cell painting"
            }
            # Write the metadata as a JSON line
            f_out.write(json.dumps(metadata) + '\n')

    print(f'Metadata file {output_file} generated with {len(image_files)} entries.')

# Specify the folder containing the images and the output metadata file
folder_path = 'sd_test'
output_file = 'metadata.jsonl'

# Generate the metadata file
generate_metadata(folder_path, output_file)
