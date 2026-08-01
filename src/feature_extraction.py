"""Feature extraction helpers for bird image classification."""

from pathlib import Path
from typing import Iterable


def load_image_paths(image_dir: str) -> list[Path]:
    """Collect image file paths from a directory."""
    return sorted(Path(image_dir).glob("**/*"))


def describe_feature_set(features: Iterable[float]) -> dict[str, float]:
    """Return a simple summary of a feature iterable."""
    values = list(features)
    return {
        "count": float(len(values)),
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def extract_hog_features(
    image_path_or_img,
    target_size: tuple[int, int] = (128, 128)
) -> "np.ndarray":
    """
    Extract HOG features from an image.
    Converts image to grayscale, resizes to target_size, and computes HOG.
    
    Parameters:
        image_path_or_img: Path to the image or a PIL Image object.
        target_size: Dimensions to resize the image to before extraction.
        
    Returns:
        np.ndarray: 1D feature vector of HOG features.
    """
    import numpy as np
    from PIL import Image
    from skimage.feature import hog

    if isinstance(image_path_or_img, (str, Path)):
        img = Image.open(image_path_or_img)
    else:
        img = image_path_or_img
        
    # Convert to grayscale and resize
    img_gray = img.convert("L")
    img_resized = img_gray.resize(target_size, Image.Resampling.LANCZOS)
    img_arr = np.array(img_resized)
    
    # Extract HOG
    features = hog(
        img_arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    return features.astype(np.float32)


def extract_color_histogram(
    image_path_or_img,
    bins: int = 32
) -> "np.ndarray":
    """
    Extract a normalized RGB color histogram from an image.
    
    Parameters:
        image_path_or_img: Path to the image or a PIL Image object.
        bins: Number of histogram bins for each RGB channel.
        
    Returns:
        np.ndarray: Concatenated RGB histogram feature vector (size: 3 * bins).
    """
    import numpy as np
    from PIL import Image

    if isinstance(image_path_or_img, (str, Path)):
        img = Image.open(image_path_or_img)
    else:
        img = image_path_or_img
        
    # Ensure RGB
    img_rgb = img.convert("RGB")
    img_arr = np.array(img_rgb)
    
    channel_histograms = []
    for channel_index in range(3):
        channel_values = img_arr[:, :, channel_index]
        histogram, _ = np.histogram(
            channel_values,
            bins=bins,
            range=(0, 256)
        )
        histogram = histogram.astype(np.float32)
        histogram_sum = histogram.sum()
        if histogram_sum > 0:
            histogram /= histogram_sum
        channel_histograms.append(histogram)
        
    return np.concatenate(channel_histograms)


def extract_lbp_histogram(
    image_path_or_img,
    number_of_points: int = 8,
    radius: int = 1
) -> "np.ndarray":
    """
    Extract a normalized Local Binary Pattern histogram from an image.
    
    Parameters:
        image_path_or_img: Path to the image or a PIL Image object.
        number_of_points: Number of neighboring points.
        radius: Radius around the center pixel.
        
    Returns:
        np.ndarray: Normalized LBP histogram feature vector (size: number_of_points + 2).
    """
    import numpy as np
    from PIL import Image
    from skimage.feature import local_binary_pattern

    if isinstance(image_path_or_img, (str, Path)):
        img = Image.open(image_path_or_img)
    else:
        img = image_path_or_img
        
    # Convert to grayscale
    img_gray = img.convert("L")
    img_arr = np.array(img_gray, dtype=np.uint8)
    
    lbp_image = local_binary_pattern(
        img_arr,
        P=number_of_points,
        R=radius,
        method="uniform"
    )
    
    number_of_bins = number_of_points + 2
    histogram, _ = np.histogram(
        lbp_image.ravel(),
        bins=np.arange(0, number_of_bins + 1),
        range=(0, number_of_bins)
    )
    histogram = histogram.astype(np.float32)
    histogram_sum = histogram.sum()
    if histogram_sum > 0:
        histogram /= histogram_sum
        
    return histogram


def combine_features(
    hog_feat,
    color_feat,
    lbp_feat=None
) -> "np.ndarray":
    """
    Combine multiple feature vectors.
    
    Parameters:
        hog_feat: HOG features array.
        color_feat: Color histogram features array.
        lbp_feat: (Optional) LBP features array.
        
    Returns:
        np.ndarray: Concatenated feature vector.
    """
    import numpy as np
    features = [hog_feat, color_feat]
    if lbp_feat is not None:
        features.append(lbp_feat)
    return np.concatenate(features, axis=-1)


def build_feature_matrices(
    target_size: tuple[int, int] = (128, 128),
    bins: int = 32,
    number_of_points: int = 8,
    radius: int = 1
):
    """
    Automatically download raw dataset, run preprocessing,
    extract HOG, Color, and LBP features, and save processed splits.
    """
    import os
    import shutil
    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import joblib
    from tqdm.auto import tqdm

    # Locate metadata file
    def locate_metadata():
        for p in [Path("metadata_preprocessed.csv"), Path("processed_data/metadata_preprocessed.csv")]:
            if p.exists():
                return p
        return None

    metadata_file = locate_metadata()
    dataset_preprocessed_dir = Path("dataset_20_species_preprocessed")

    # If dataset prep is needed
    if metadata_file is None or not dataset_preprocessed_dir.exists():
        print("Preprocessed dataset or metadata not found. Running automatic dataset preparation...")
        import kagglehub
        download_path = Path(kagglehub.dataset_download("wenewone/cub2002011"))
        DATASET_ROOT = download_path / "CUB_200_2011"
        IMAGES_FOLDER = DATASET_ROOT / "images"
        
        SUBSET_DIR = Path("./dataset_20_species")
        if SUBSET_DIR.exists():
            shutil.rmtree(SUBSET_DIR)
        SUBSET_DIR.mkdir(parents=True, exist_ok=True)
        
        # Select 20 species based on project keywords
        species_folders = sorted([f for f in IMAGES_FOLDER.iterdir() if f.is_dir()])
        bd_keywords = [
            'Crow', 'Kingfisher', 'Hummingbird', 'Mallard', 'Warbler',
            'Towhee', 'Jay', 'Creeper', 'Waxwing', 'Cuckoo',
            'Thrush', 'Woodpecker', 'Wren', 'Vireo', 'Catbird',
            'Meadowlark', 'Blackbird', 'Gull', 'Tern', 'Pelican'
        ]
        selected_species = []
        for folder in species_folders:
            if any(kw.lower() in folder.name.lower() for kw in bd_keywords):
                if folder not in selected_species:
                    selected_species.append(folder)
            if len(selected_species) == 20:
                break
        
        for species_path in selected_species:
            shutil.copytree(str(species_path), SUBSET_DIR / species_path.name)
            
        classes = pd.read_csv(DATASET_ROOT / "classes.txt", sep=r"\s+", names=["class_id", "class_name"])
        images = pd.read_csv(DATASET_ROOT / "images.txt", sep=r"\s+", names=["image_id", "image_path"])
        image_labels = pd.read_csv(DATASET_ROOT / "image_class_labels.txt", sep=r"\s+", names=["image_id", "class_id"])
        
        selected_species_names = [p.name for p in selected_species]
        dataset_info = images.merge(image_labels, on="image_id").merge(classes, on="class_id")
        dataset_info = dataset_info[dataset_info["class_name"].isin(selected_species_names)].copy().reset_index(drop=True)
        dataset_info["full_image_path"] = dataset_info["image_path"].apply(lambda p: SUBSET_DIR / p)
        
        from src.preprocessing import create_stratified_splits, preprocess_and_save_image
        
        split_metadata = create_stratified_splits(dataset_info, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42)
        
        if dataset_preprocessed_dir.exists():
            shutil.rmtree(dataset_preprocessed_dir)
        dataset_preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessed_paths = []
        for idx, row in split_metadata.iterrows():
            src_file = row["full_image_path"]
            rel_path = Path(row["image_path"])
            dst_file = dataset_preprocessed_dir / rel_path
            preprocess_and_save_image(src_file, dst_file, target_size=(224, 224))
            preprocessed_paths.append(str(dst_file))
            
        split_metadata["preprocessed_image_path"] = preprocessed_paths
        PROCESSED_DIR = Path("./processed_data")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cols = ["image_id", "image_path", "class_id", "class_name", "split", "preprocessed_image_path"]
        split_metadata[cols].to_csv("metadata_preprocessed.csv", index=False)
        split_metadata[cols].to_csv(PROCESSED_DIR / "metadata_preprocessed.csv", index=False)
        metadata_file = Path("metadata_preprocessed.csv")
        print("Dataset preparation complete!")

    metadata_df = pd.read_csv(metadata_file)
    X_hog, X_color, X_lbp, y = [], [], [], []

    print("Extracting features from preprocessed images...")
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting features"):
        img_path = Path(row["preprocessed_image_path"])
        hog_feat = extract_hog_features(img_path, target_size=target_size)
        color_feat = extract_color_histogram(img_path, bins=bins)
        lbp_feat = extract_lbp_histogram(img_path, number_of_points=number_of_points, radius=radius)
        
        X_hog.append(hog_feat)
        X_color.append(color_feat)
        X_lbp.append(lbp_feat)
        y.append(row["class_id"] - 1)

    X_hog = np.array(X_hog, dtype=np.float32)
    X_color = np.array(X_color, dtype=np.float32)
    X_lbp = np.array(X_lbp, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    splits = metadata_df["split"].to_numpy()

    unique_classes = sorted(metadata_df["class_name"].unique())
    label_mapping = {class_name: idx for idx, class_name in enumerate(unique_classes)}

    X_combined_hog_color = combine_features(X_hog, X_color)
    X_combined_hog_color_lbp = combine_features(X_hog, X_color, X_lbp)

    OUTPUT_DIR = Path("./processed_data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X_hog.npy", X_hog)
    np.save(OUTPUT_DIR / "X_color.npy", X_color)
    np.save(OUTPUT_DIR / "X_lbp.npy", X_lbp)
    np.save(OUTPUT_DIR / "X_combined_hog_color.npy", X_combined_hog_color)
    np.save(OUTPUT_DIR / "X_combined_hog_color_lbp.npy", X_combined_hog_color_lbp)
    np.save(OUTPUT_DIR / "y_labels.npy", y)
    np.save(OUTPUT_DIR / "splits.npy", splits)
    joblib.dump(label_mapping, OUTPUT_DIR / "label_mapping.pkl")

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    def save_split_npz(filename, X):
        np.savez_compressed(
            OUTPUT_DIR / filename,
            X_train=X[train_mask],
            y_train=y[train_mask],
            X_val=X[val_mask],
            y_val=y[val_mask],
            X_test=X[test_mask],
            y_test=y[test_mask]
        )

    save_split_npz("hog_features.npz", X_hog)
    save_split_npz("color_features.npz", X_color)
    save_split_npz("lbp_features.npz", X_lbp)
    save_split_npz("combined_hog_color.npz", X_combined_hog_color)
    save_split_npz("combined_hog_color_lbp.npz", X_combined_hog_color_lbp)
    print("Automatic feature matrices generation finished!")


