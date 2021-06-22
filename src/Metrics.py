from util.util import *
from skimage.metrics import mean_squared_error, structural_similarity

L = 256


def entropy(image: np.ndarray) -> int:
    hist = np.bincount(np.array(image).flatten(), minlength=MAX_PIXEL_VALUE + 1)
    normalized_hist = hist / np.sum(hist)
    normalized_hist = normalized_hist[np.flatnonzero(normalized_hist)]
    return -np.sum(normalized_hist * np.log2(normalized_hist))


def relative_entropy_error(old_image: np.ndarray, new_image: np.ndarray) -> float:
    return (entropy(new_image) - entropy(old_image)) / (2 * np.log2(L)) + 0.5


def relative_contrast_error(old_image: np.ndarray, new_image: np.ndarray) -> float:
    return (new_image.std() - old_image.std()) / (L - 1) + 0.5


def relative_mean_brightness_error(old_image: np.ndarray, new_image: np.ndarray) -> float:
    return 1 - np.abs(new_image.mean() - old_image.mean()) / (L - 1)


def relative_structural_similarity(old_image: np.ndarray, new_image: np.ndarray) -> float:
    return 1 - np.sqrt(mean_squared_error(old_image, new_image)) / (L - 1)


old_img = read_image(f'res/kodek_dataset/kodim20_org.png')
new_img = read_image(f'out/bp_uni_improved/kodim20_org.png')
print(relative_entropy_error(old_img, new_img))
print(relative_contrast_error(old_img, new_img))
print(relative_mean_brightness_error(old_img, new_img))
print(relative_structural_similarity(old_img, new_img))
print(structural_similarity(old_img, new_img))
