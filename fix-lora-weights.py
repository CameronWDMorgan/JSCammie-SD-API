import yaml

# The path to the YAML file
file_path = 'lora_weights.yaml'

# Function to replace '/' with '\' in 'lora' keys
def replace_slashes(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'lora':
                data[key] = value.replace('/', '\\')
            else:
                replace_slashes(value)
    elif isinstance(data, list):
        for item in data:
            replace_slashes(item)

# Read the YAML data from the file
with open(file_path, 'r', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)

# Process the YAML data to replace slashes
replace_slashes(yaml_data)

# Write the modified YAML data back to the file
with open(file_path, 'w', encoding='utf-8') as file:
    yaml.safe_dump(yaml_data, file, default_flow_style=False, allow_unicode=True)

print("Successfully updated the file with replaced slashes.")
