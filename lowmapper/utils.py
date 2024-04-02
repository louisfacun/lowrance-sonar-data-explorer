import math
import numpy as np
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
    

def x_to_longitude(x):
    """
    POLAR_EARTH_RADIUS = 6356752.3142;

    longitude = Easting / POLAR_EARTH_RADIUS * (180/M_PI);

    https://wiki.openstreetmap.org/wiki/SL2

    """
    return(x/6356752.3142*(180/math.pi))


def y_to_latitude(y):
    return(((2*np.arctan(np.exp(y/6356752.3142)))-(math.pi/2))*(180/math.pi))