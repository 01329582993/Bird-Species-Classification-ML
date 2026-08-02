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
    Extract HOG features from an image (grayscale + per-channel RGB HOG).
    Converts image to grayscale AND RGB, resizes to target_size, and computes
    HOG on all four channels, then concatenates them for richer representation.
    
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
        img = Image.open(image_path_or_img).convert("RGB")
    else:
        img = image_path_or_img.convert("RGB")
        
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    img_arr = np.array(img_resized)          # H x W x 3, uint8

    hog_params = dict(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )

    # Grayscale HOG
    gray = np.mean(img_arr, axis=2).astype(np.uint8)
    feats = [hog(gray, **hog_params)]

    # Per-channel HOG (R, G, B)
    for ch in range(3):
        feats.append(hog(img_arr[:, :, ch], **hog_params))

    return np.concatenate(feats).astype(np.float32)


def extract_color_histogram(
    image_path_or_img,
    bins: int = 32
) -> "np.ndarray":
    """
    Extract a normalized multi-colorspace color histogram from an image.
    Computes histograms in RGB, HSV, and L*a*b* color spaces and concatenates
    them for a richer color representation (size: 9 * bins).
    
    Parameters:
        image_path_or_img: Path to the image or a PIL Image object.
        bins: Number of histogram bins for each channel.
        
    Returns:
        np.ndarray: Concatenated multi-colorspace histogram feature vector.
    """
    import numpy as np
    from PIL import Image
    import cv2

    if isinstance(image_path_or_img, (str, Path)):
        img = Image.open(image_path_or_img).convert("RGB")
    else:
        img = image_path_or_img.convert("RGB")

    img_rgb = np.array(img, dtype=np.uint8)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    all_hists = []
    for colorspace_img, ranges in [
        (img_rgb, [(0, 256)] * 3),
        (img_hsv, [(0, 180), (0, 256), (0, 256)]),
        (img_lab, [(0, 256)] * 3),
    ]:
        for ch in range(3):
            hist, _ = np.histogram(
                colorspace_img[:, :, ch],
                bins=bins,
                range=ranges[ch]
            )
            hist = hist.astype(np.float32)
            s = hist.sum()
            if s > 0:
                hist /= s
            all_hists.append(hist)

    return np.concatenate(all_hists)


def extract_lbp_histogram(
    image_path_or_img,
    number_of_points: int = 8,
    radius: int = 1
) -> "np.ndarray":
    """
    Extract a multi-scale LBP histogram from an image.
    Computes LBP at two scales (radius=1 and radius=3) and concatenates
    the histograms for richer texture representation.
    
    Parameters:
        image_path_or_img: Path to the image or a PIL Image object.
        number_of_points: Number of neighboring points (for radius=1 scale).
        radius: Primary radius (a second scale at 3x is always added).
        
    Returns:
        np.ndarray: Concatenated multi-scale LBP histogram.
    """
    import numpy as np
    from PIL import Image
    from skimage.feature import local_binary_pattern

    if isinstance(image_path_or_img, (str, Path)):
        img = Image.open(image_path_or_img)
    else:
        img = image_path_or_img
        
    img_gray = img.convert("L")
    img_arr = np.array(img_gray, dtype=np.uint8)

    def _lbp_hist(arr, P, R):
        lbp = local_binary_pattern(arr, P=P, R=R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, n_bins + 1),
            range=(0, n_bins)
        )
        hist = hist.astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist

    # Scale 1: small radius (fine texture)
    hist1 = _lbp_hist(img_arr, P=number_of_points, R=radius)
    # Scale 2: larger radius (coarser texture)
    hist2 = _lbp_hist(img_arr, P=24, R=3)

    return np.concatenate([hist1, hist2])


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


