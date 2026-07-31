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

