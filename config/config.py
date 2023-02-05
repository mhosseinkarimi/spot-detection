import yaml

def load_config():
    with open("/mnt/c/Users/mhuss/OneDrive/Desktop/spot_measurement_material/spot-measurements-codes/config/config.yaml", "r") as cfgfile:
        cfg = yaml.safe_load(cfgfile)
    return cfg
