import yaml

def parse_yaml(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
        return config
    
def display_config(config):
    print("config: ", end="")
    for key, value in config.items():
        print(f"{key}={value}, ", end="")
    print()