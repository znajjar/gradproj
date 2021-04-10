import lzma
from collections.abc import Iterable
from typing import Union

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

COMPRESSED_DATA_LENGTH_BITS = 16
MAX_PIXEL_VALUE = 255
COMPRESSION_LEVEL = 9
HEADER_SIZE = 17
EPS = 0.00000005


def integer_to_binary(number: int, bits=8):
    return [x == '1' for x in format(number, f'0{bits}b')]


def binary_to_string(binary):
    if binary:
        return '1'
    else:
        return '0'


def binary_to_integer(binary):
    return int.from_bytes(bits_to_bytes(binary), byteorder='big', signed=False)


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


def bits_to_bytes(buffer):
    return np.packbits(buffer).tobytes()


def _compress(data_bits):
    data_bytes = bits_to_bytes(data_bits)

    compressor = lzma.LZMACompressor()
    ret = compressor.compress(data_bytes)
    ret += (compressor.flush())
    return ret

    # return zlib.compress(data_bytes, level=COMPRESSION_LEVEL)

    # return bz2.compress(data_bytes)


def decompress(compressed_data_bits):
    data_bytes = bits_to_bytes(compressed_data_bits)

    ret = lzma.LZMADecompressor().decompress(data_bytes)
    return ret

    # return zlib.decompress(data_bytes)

    # return bz2.decompress(data_bytes)


def get_header_and_body(image: np.ndarray, header_size: int = HEADER_SIZE) -> (np.ndarray, np.ndarray):
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
    original_range = np.max(image)
    scaled_range = scaled_max - scaled_min

    image = image.astype(np.float64)
    scale_factor = scaled_range / original_range

    image *= scale_factor

    if scaled_range <= original_range:  # TODO make sure this always works
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


def get_peaks(pixels, n=2):
    hist = np.bincount(pixels)
    return np.sort(hist.argsort()[-n:])


def show_hist(image):
    bins = np.bincount(image.ravel())
    hist = np.zeros((MAX_PIXEL_VALUE + 1,))
    hist[:len(bins)] = bins
    plt.figure()
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.hist(np.arange(MAX_PIXEL_VALUE + 1), MAX_PIXEL_VALUE + 1, weights=hist)
    plt.show()


def assemble_image(header, pixels, shape):
    image = np.append(header, pixels)
    return image.reshape(shape)
