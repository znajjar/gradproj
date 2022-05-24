import os.path
from collections.abc import Iterable
from typing import Union

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

IMAGE_EXTENSIONS = ['png', 'jpeg', 'tiff', 'tif', 'bmp', 'jpg', 'gif']
MAX_PIXEL_VALUE = 255
EPS = 0.00000005

RED = '\033[91m'
ENDC = '\033[0m'


def integer_to_binary(number: int, bits=8):
    return np.array([x == '1' for x in format(number, f'0{bits}b')])[:bits]


def binary_to_string(binary):
    if binary:
        return '1'
    else:
        return '0'


def binary_to_integer(binary, byteorder='big'):
    return int.from_bytes(bits_to_bytes(pad_bits(binary)), byteorder=byteorder, signed=False)


def get_lsb(values):
    lsb = []
    for pixel in values:
        lsb.append(integer_to_binary(pixel)[-1])

    return lsb


def set_lsb(value, lsb):
    if lsb:
        value |= 1
    else:
        value &= ~1

    return value


def bytes_to_bits(buffer):
    return np.where(np.unpackbits(np.frombuffer(buffer, np.uint8)) == 1, True, False)


def bits_to_bytes(buffer, bitorder='big'):
    return np.packbits(buffer, bitorder=bitorder).tobytes()


def pad_bits(bits):
    pad_size = 8 - len(bits) % 8
    return np.append(np.zeros((pad_size,)), bits).astype(bool)


def get_header_and_body(image: np.ndarray, header_size: int) -> (np.ndarray, np.ndarray):
    image = image.ravel().copy()
    return image[:int(header_size)], image[int(header_size):]


def scale_to(image: np.ndarray, r: Union[np.ndarray, Iterable, int, str]) -> np.ndarray:
    try:
        scaled_min, scaled_max = r
    except TypeError:
        scaled_max = r
        scaled_min = 0

    scaled_min = int(scaled_min)
    scaled_max = int(scaled_max)

    image -= np.min(image)
    original_range = np.max(image) + 1
    scaled_range = scaled_max - scaled_min + 1

    image = image.astype(np.float64)
    scale_factor = scaled_range / original_range

    image *= scale_factor

    if scaled_range > original_range:
        image -= EPS
        image = np.ceil(image)
    else:
        image += EPS
        image = np.floor(image)

    image += scaled_min

    return image.astype(np.uint8)


def get_mapped_values(original_max: int, scaled_max: int) -> np.ndarray:
    original_max = int(original_max)
    scaled_max = int(scaled_max)

    og_values = np.arange(original_max + 1)
    scaled_values = scale_to(og_values, scaled_max)
    recovered_values = scale_to(scaled_values, original_max)
    mapped_values = scaled_values[np.where(recovered_values - og_values != 0)]

    if not len(mapped_values):
        mapped_values = np.array([-1])

    return mapped_values


def read_image(path: str) -> np.ndarray:
    return np.uint8(Image.open(path).getchannel(0)).copy()


def save_image(image, path: str):
    Image.fromarray(image).save(path)


def get_peaks(pixels, n=2):
    hist = np.bincount(pixels)
    return np.sort(hist.argsort()[-n:])


def show_hist(image, title=''):
    bins = np.bincount(image.ravel())
    hist = np.zeros((MAX_PIXEL_VALUE + 1,))
    hist[:len(bins)] = bins
    plt.figure()
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.title(title)
    plt.hist(np.arange(MAX_PIXEL_VALUE + 1), MAX_PIXEL_VALUE + 1, weights=hist)
    plt.show()


def assemble_image(header, pixels, shape):
    image = np.append(header, pixels)
    return image.reshape(shape)


def get_minimum_closest_right(hist, pixel_value):
    hist_right = (np.roll(hist, 1) + hist)[pixel_value + 2:]
    candidates = np.flatnonzero(hist_right == hist_right.min()) + pixel_value + 2
    candidates = candidates[np.flatnonzero(hist[candidates - 1] == hist[candidates - 1].min())]
    return candidates[np.abs(candidates - pixel_value).argmin()]


def get_minimum_closest_left(hist, pixel_value):
    hist_left = (np.roll(hist, -1) + hist)[:pixel_value - 2 + 1]
    candidates = np.flatnonzero(hist_left == hist_left.min())
    candidates = candidates[np.flatnonzero(hist[candidates + 1] == hist[candidates + 1].min())]
    return candidates[np.abs(candidates - pixel_value).argmin()]


def get_minimum_closest(hist, pixel_value):
    closest_right = get_minimum_closest_right(hist, pixel_value)
    closest_left = get_minimum_closest_left(hist, pixel_value)
    min_right_value = hist[closest_right] + hist[closest_right - 1]
    min_left_value = hist[closest_left] + hist[closest_left + 1]
    if min_right_value < min_left_value:
        return closest_right
    elif min_right_value > min_left_value:
        return closest_left
    else:
        if abs(closest_right - pixel_value) < abs(closest_left - pixel_value):
            return closest_right
        else:
            return closest_left


def get_shift_direction(P_L, P_H):
    if P_L < P_H:
        return -1
    else:
        return 1


def estimate_compressed_map_size(location_map_size, percentage):
    compressed_map_size = location_map_size.copy()

    low_range = np.logical_and(location_map_size >= 200, location_map_size <= 2000)
    compressed_map_size[low_range] = 2.318468 * percentage[low_range] ** 0.405595 * location_map_size[
        low_range] ** -0.038066 + 226.678704 / location_map_size[low_range]

    high_range = location_map_size > 2000
    compressed_map_size[high_range] = (2.4274572 * percentage[high_range] ** 0.3849364 - 0.1690710) * \
                                      location_map_size[high_range] ** -0.0395816

    return compressed_map_size * location_map_size


def find_closest_candidate(candidates, value):
    candidates = np.atleast_1d(candidates)
    return candidates[np.abs(candidates - value).argmin()]


def get_peaks_from_header(header_pixels: np.ndarray, peak_size: int = 8) -> (np.ndarray, np.ndarray):
    LSB = get_lsb(header_pixels)
    return binary_to_integer(LSB[0:peak_size]), binary_to_integer(LSB[peak_size:2 * peak_size])


def is_image(image_path: str) -> bool:
    return os.path.isfile(image_path) and os.path.splitext(image_path)[1][1:] in IMAGE_EXTENSIONS


def test_algorithm_by_directory(embedder, extractor, directory_path: str, data):
    for filename in os.listdir(directory_path):
        joined_path = os.path.join(directory_path, filename)
        if is_image(joined_path):
            print(f'Filename: {filename}')

            image = read_image(joined_path)
            embedded_image, iterations, pure_embedded_data = embedder(image, data).embed(1000)
            print(f'iterations: {iterations}')
            print(f'rate: {pure_embedded_data / image.size}')
            print(f'Abs change in mean: {abs(embedded_image.mean() - image.mean())}')
            print(f'Change in STD: {embedded_image.std() - image.std()}')
            print(f'SSIM: {structural_similarity(image, embedded_image)}')

            save_image(embedded_image, f'out/{filename}')

            print(f'Correct extraction? {np.sum(np.abs(extractor().extract(embedded_image)[0] - image))}')
            print()


def print_error(text):
    print(f'{RED}{text}{ENDC}')
