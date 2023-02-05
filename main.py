import cv2 as cv
import os
import sys
from pathlib import Path 
import pandas as pd
import numpy as np

from src.detection.detectors import ThresholdDetector, WatershedDetector
from config.config import load_config


cfg = load_config()

print(cfg["test_root"])
if not os.path.isdir(cfg["test_root"]):
    print(f"{cfg['test_root']} is not a valid directory")
    sys.exit()

out_path = Path(cfg["output_dir"])
if not os.path.isdir(out_path):
    print(f"{out_path} is not a valid directory path.")
    sys.exit()
if not os.path.exists(out_path):
    os.mkdir(out_path)

image_suffix_list = [".jpg", ".jpeg", ".png"]

thresh_method = cfg["thresh_type"]

if cfg["detector"] == "watershed":
    detector = WatershedDetector(cfg["thresh_type"], tuple(cfg["thresh_steps"]))  
elif cfg["detector"] == "thresholding":
    if "local" in cfg["thresh_type"]:
        detector = ThresholdDetector(cfg["thresh_type"], tuple(cfg["thresh_steps"]))
    else:
        detector = ThresholdDetector(cfg["thresh_type"])
else:
    sys.exit()  


for root, dirs, files in os.walk(cfg["test_root"]):
    saveing_dir_path = out_path / Path(root).relative_to(cfg["test_root"])
    if not os.path.exists(saveing_dir_path):
        os.mkdir(saveing_dir_path)

    stats_df = pd.DataFrame()
    for file in files:
        file = Path(root)/Path(file)

        if file.suffix in image_suffix_list:
            in_img = cv.imread(str(file))
            output = detector.detect(in_img)
            stats = detector.stats(output, cfg["img_height"], cfg["img_width"])
            stats_df = pd.concat([stats_df, pd.DataFrame(stats, index=[file.stem])])

            if cfg["detector"] == "watershed":
                mask = np.ones(output.shape+(3,)) * 255
                for i in range(1, np.max(output)):
                    color = np.random.randint(0, 256, size=(3,), dtype="uint8")
                    mask[output == i] = color
                output = mask
            print(saveing_dir_path/f"{file.stem}.jpg")
            cv.imwrite(str(saveing_dir_path/f"{file.stem}.jpg"), output)

    if len(stats_df) > 0:
        stats_df.to_csv(saveing_dir_path/"stats.csv") 
            

