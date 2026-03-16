"""Combine the images from the training set and the testing set into a single folder"""
import numpy as np
import shutil
import os
from pathlib import Path

def copy_training_images(metadata_path, source_folder, target_folder):
    """
    
    Args:
        metadata_path: image_metadata.npy 
        source_folder: training_images or test_images
        target_folder: target dir path
    """
    
    data = np.load(metadata_path, allow_pickle=True).item()
    
    
    train_concepts = data['train_img_concepts']      # e.g., '00001_aardvark'
    train_concepts_things = data['train_img_concepts_THINGS']  # e.g., '00001_aardvark'
    train_files = data['train_img_files']            # e.g., 'aardvark_01b.jpg'
    
    print(f" {len(train_files)} files in total")
    
    os.makedirs(target_folder, exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    for i in range(len(train_files)):
        source_path = os.path.join(source_folder, train_concepts[i], train_files[i])
        target_dir = os.path.join(target_folder, train_concepts_things[i])
        target_path = os.path.join(target_dir, train_files[i])
        
        try:
            os.makedirs(target_dir, exist_ok=True)
            
            if os.path.exists(source_path):
                shutil.copy(source_path, target_path)
                success_count += 1
                
            
                if success_count % 1000 == 0:
                    print(f"proceeded {success_count}/{len(train_files)} files...")
            else:
                print(f"warning: source file didn`t exist - {source_path}")
                error_count += 1
                
        except Exception as e:
            print(f"error {source_path} -> {target_path}")
            print(f"error information: {e}")
            error_count += 1
    
    print(f"\nfinished!")
    print(f"successfully copy: {success_count} files")
    print(f"error and skip: {error_count} files")


# 使用示例
if __name__ == "__main__":
    METADATA_PATH = "/mnt/dataset2/Datasets/Things-EEG2/image_metadata.npy"          
    SOURCE_FOLDER = "/mnt/dataset2/Datasets/Things-EEG2/training_images"              
    TARGET_FOLDER = "/mnt/dataset2/hcy/preprocessing/images"  
    copy_training_images(METADATA_PATH, SOURCE_FOLDER, TARGET_FOLDER)
