import cv2
import numpy as np
import os
from pathlib import Path

def standardize_student_masks(student_masks_dir, output_dir, student_mapping, master_mapping):
    """
    Reads masks from a student's directory, remaps the pixel values to match 
    the master mapping, and saves them to an output directory.
    """
    student_dir_path = Path(student_masks_dir)
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Find all png files in the student's mask folder
    mask_files = list(student_dir_path.glob("*.png"))
    
    if not mask_files:
        print(f"No PNG masks found in {student_masks_dir}")
        return

    print(f"Processing {len(mask_files)} masks...")

    for mask_path in mask_files:
        # 1. Load the student's mask in grayscale (loads as 2D integer array)      
        # 2. Create a blank canvas of zeros (Background) with the exact same dimensions   
        # 3. Apply the mapping safely              
        # 4. Save the corrected mask to the new folder
        pass 
        


# ==========================================
# LAB INSTRUCTOR CONFIGURATION
# ==========================================

# 1. Define the Master Mapping (Your strict rules for the final dataset)
MASTER_DICT = {
    "background": 0,
    "track": 1,
    "vegetation": 2
}

# Let's pretend you got vegetation and track backwards.
GROUP_A_DICT = {
    "background": 0,
    "vegetation": 1, 
    "track": 2       
}

# 3. Define the folder paths
# Where are original unzipped masks?
group_a_masks_folder = r"D:\Documents\AIforRobotics\GroupA\masks" 

# Where do you want the fixed masks to go? 
fixed_output_folder = r"D:\Documents\AIforRobotics\MasterDataset\masks"

# Run the standardization!
# standardize_student_masks(group_a_masks_folder, fixed_output_folder, GROUP_A_DICT, MASTER_DICT)
