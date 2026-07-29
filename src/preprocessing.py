"""Utilities for dataset preprocessing."""

from pathlib import Path
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def list_dataset_files(dataset_dir: str) -> list[Path]:
    """Return a sorted list of files from the dataset directory."""
    return sorted(Path(dataset_dir).glob("**/*"))


def preprocess_and_save_image(
    src_path: Path, 
    dst_path: Path, 
    target_size: tuple[int, int] = (224, 224)
) -> tuple[bool, str]:
    """
    Loads an image from src_path, converts it to RGB, resizes it to target_size,
    and saves it to dst_path.
    
    Parameters:
        src_path (Path): Path to the source raw image.
        dst_path (Path): Path where the resized image should be saved.
        target_size (tuple): Target dimensions (width, height).
        
    Returns:
        tuple[bool, str]: (success, error_message)
    """
    try:
        # Create destination directory if it doesn't exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load and process the image
        with Image.open(src_path) as img:
            # Convert palette/RGBA images to RGB
            img_rgb = img.convert("RGB")
            # Resize image with high-quality Lanczos/Resampling filter
            resized_img = img_rgb.resize(target_size, Image.Resampling.LANCZOS)
            # Save preprocessed image
            resized_img.save(dst_path, "JPEG", quality=95)
            
        return True, ""
    except Exception as e:
        return False, str(e)


def load_normalized_image(image_path: Path) -> np.ndarray:
    """
    Loads a resized image and normalizes its pixel values to [0.0, 1.0].
    
    Parameters:
        image_path (Path): Path to the image file.
        
    Returns:
        np.ndarray: Normalized image array with shape (height, width, channels)
                    and dtype float32.
    """
    with Image.open(image_path) as img:
        img_arr = np.array(img, dtype=np.float32)
        # Normalize pixel values to range [0.0, 1.0]
        normalized_arr = img_arr / 255.0
        return normalized_arr


def create_stratified_splits(
    metadata_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Creates stratified train, validation, and test splits based on the class_id.
    
    Parameters:
        metadata_df (pd.DataFrame): DataFrame containing 'image_id' and 'class_id'.
        train_ratio (float): Ratio for training set.
        val_ratio (float): Ratio for validation set.
        test_ratio (float): Ratio for test set.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Updated DataFrame with a 'split' column.
    """
    df = metadata_df.copy()
    
    # Calculate ratios relative to the remaining part
    # Split first into Train and (Val + Test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["class_id"],
        random_state=random_state
    )
    
    # Split (Val + Test) into Val and Test
    test_relative_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_relative_ratio,
        stratify=val_test_df["class_id"],
        random_state=random_state
    )
    
    # Assign split labels
    df.loc[df["image_id"].isin(train_df["image_id"]), "split"] = "train"
    df.loc[df["image_id"].isin(val_df["image_id"]), "split"] = "val"
    df.loc[df["image_id"].isin(test_df["image_id"]), "split"] = "test"
    
    return df

