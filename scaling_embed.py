import argparse

import cv2

from compress import Lzma
from compress import Zlib
from data_buffer import DataBuffer
from shared import *
from decimal import *

getcontext().prec = 50


def get_is_rounded(og, processed):
    processed_pixels_temp = processed / (np.max(processed) / (original_max - original_min))
    processed_pixels_temp = np.round(processed_pixels_temp, 7)
    processed_pixels_temp = np.floor(processed_pixels_temp)
    return processed_pixels_temp + original_min - og


def preprocess():
    global is_rounded, original_min, original_max, processed_pixels
    original_min = np.min(processed_pixels)
    original_max = np.max(processed_pixels)
    processed_pixels_og = processed_pixels.copy()
    processed_pixels = processed_pixels.astype(np.longdouble)

    processed_pixels -= original_min
    shifted_max = np.max(processed_pixels)
    scaled_max = MAX_PIXEL_VALUE - 2 * iterations
    scale_factor = scaled_max / shifted_max
    processed_pixels *= scale_factor
    processed_pixels[processed_pixels_og == original_max] = scaled_max
    processed_pixels = np.ceil(processed_pixels)

    is_rounded = get_is_rounded(processed_pixels_og, processed_pixels)
    print('unique values in is_rounded before fixing:', np.unique(is_rounded))

    processed_pixels[is_rounded == -1] += 1


    is_rounded = get_is_rounded(processed_pixels_og, processed_pixels)
    print('unique values in is_rounded after fixing:', np.unique(is_rounded))
    is_rounded = is_rounded.astype(np.bool)

    processed_pixels += iterations
    processed_pixels = processed_pixels.astype(np.uint8)


def fill_buffer():
    global buffer, parity
    is_modified_compressed = compress(bits_to_bytes(is_rounded))
    is_modified_size_bits = integer_to_binary(len(is_modified_compressed), COMPRESSED_DATA_LENGTH_BITS)
    is_modified_bits = bytes_to_bits(is_modified_compressed)
    hidden_data_bits = bytes_to_bits(hidden_data)
    buffer = DataBuffer(np.bool, is_modified_size_bits, is_modified_bits, hidden_data_bits, calc_parity=True)
    parity = buffer.get_parity()


def get_peaks():
    hist, _ = np.histogram(processed_pixels, 256, [0, 256])
    max_value = hist.max(initial=None)
    max_value_indices = np.argwhere(hist == max_value)
    max_peak = max_value_indices[0][0]
    if max_value_indices.size > 1:
        second_max_peak = max_value_indices[-1][0]
    else:
        hist[hist == max_value] = 0
        max_value = hist.max(initial=None)
        max_value_indices = np.argwhere(hist == max_value)
        second_max_peak = max_value_indices[0][0]

    if max_peak > second_max_peak:
        max_peak, second_max_peak = second_max_peak, max_peak

    return max_peak, second_max_peak


def process():
    global iterations
    previous_left_peaks = previous_right_peaks = 0

    def get_previous_binary():
        ret = []
        ret.extend(integer_to_binary(previous_left_peaks))
        ret.extend(integer_to_binary(previous_right_peaks))
        return ret

    buffer.add(integer_to_binary(original_max))
    buffer.add(integer_to_binary(original_min))
    buffer.add(get_lsb(header_pixels))

    while iterations:
        iterations -= 1
        left_peak, right_peak = get_peaks()

        processed_pixels[processed_pixels < left_peak] -= 1
        processed_pixels[processed_pixels > right_peak] += 1

        binary_previous_peaks = get_previous_binary()

        buffer.add(binary_previous_peaks)

        processed_pixels[processed_pixels == left_peak] -= buffer.next(np.count_nonzero(processed_pixels == left_peak))
        processed_pixels[processed_pixels == right_peak] += buffer.next(
            np.count_nonzero(processed_pixels == right_peak))

        previous_left_peaks = left_peak
        previous_right_peaks = right_peak

    header_pixels[0] = set_lsb(header_pixels[0], parity)
    binary_previous_peaks = get_previous_binary()
    binary_index = 0
    for index in range(1, header_pixels.size):
        binary_value = binary_previous_peaks[binary_index]
        binary_index += 1
        header_pixels[index] = set_lsb(header_pixels[index], binary_value)


def assemble_image():
    global processed_image
    pixels = np.append(header_pixels, processed_pixels)
    processed_image = pixels.reshape(cover_image.shape)


def write_image():
    is_successful = cv2.imwrite('out/embedded_with_scaling.png', processed_image)
    if is_successful:
        print('Embedding was successful.')
    else:
        print("Embedding was unsuccessful.")


def main():
    global header_pixels, processed_pixels, binary_data_index, binary_data, peaks
    header_pixels, processed_pixels = get_header_and_body(cover_image, 17)
    binary_data_index = 0
    binary_data = []
    peaks = []

    preprocess()
    fill_buffer()
    process()
    assemble_image()


def embed(image, data_to_be_hidden, iterations_count=32, compression=None) -> (np.ndarray, iter):
    global cover_image, hidden_data, iterations, compress
    cover_image = image
    hidden_data = data_to_be_hidden
    iterations = iterations_count
    if compression:
        compress = compression
    main()
    remaining_data = bits_to_bytes(buffer.next(-1))
    return processed_image, remaining_data


header_pixels = None
processed_pixels = None
binary_data_index = None
binary_data = None
is_rounded = None
peaks = None

path = None
cover_image = None
hidden_data = None
iterations = None
processed_image = None
buffer = None

original_min = None
original_max = None

parity = None

compress = Zlib.compress

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='The path of the cover image.', type=str)
    parser.add_argument('data', help='The data to be hidden in the cover image.', type=str)
    parser.add_argument('iterations', help='Number of iterations.', type=int)
    args = parser.parse_args()

    path = args.source
    cover_image = cv2.imread(path)[:, :, 0]
    hidden_data = args.data
    iterations = args.iterations

    with open(hidden_data, 'rb') as data:
        hidden_data = data.read()

    main()
    write_image()
