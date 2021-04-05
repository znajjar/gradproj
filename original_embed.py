import argparse

import cv2

from compress import Zlib
from data_buffer import BoolDataBuffer
from shared import *


def preprocess():
    global is_modified
    is_modified = np.zeros_like(processed_pixels, dtype=np.bool)
    lower_bound = processed_pixels < iterations
    upper_bound = MAX_PIXEL_VALUE - iterations < processed_pixels
    is_modified |= lower_bound
    is_modified |= upper_bound
    processed_pixels[lower_bound] += iterations
    processed_pixels[upper_bound] -= iterations
    is_modifiable = np.logical_or(processed_pixels < 2 * iterations,
                                  processed_pixels > MAX_PIXEL_VALUE - 2 * iterations)
    is_modified = is_modified[is_modifiable]


def fill_buffer():
    global buffer
    is_modified_compressed = compress(bits_to_bytes(is_modified))
    is_modified_size_bits = integer_to_binary(len(is_modified_compressed), COMPRESSED_DATA_LENGTH_BITS)
    is_modified_bits = bytes_to_bits(is_modified_compressed)
    hidden_data_bits = bytes_to_bits(hidden_data)
    buffer = BoolDataBuffer(is_modified_size_bits, is_modified_bits, hidden_data_bits)


def process():
    global iterations
    previous_left_peaks = previous_right_peaks = 0

    def get_previous_binary():
        ret = []
        ret.extend(integer_to_binary(previous_left_peaks))
        ret.extend(integer_to_binary(previous_right_peaks))
        return ret

    buffer.add(get_lsb(header_pixels))

    while iterations:
        iterations -= 1
        left_peak, right_peak = get_peaks(processed_pixels)

        processed_pixels[processed_pixels < left_peak] -= 1
        processed_pixels[processed_pixels > right_peak] += 1

        binary_previous_peaks = get_previous_binary()

        buffer.add(binary_previous_peaks)

        processed_pixels[processed_pixels == left_peak] -= buffer.next(np.count_nonzero(processed_pixels == left_peak))
        processed_pixels[processed_pixels == right_peak] += buffer.next(
            np.count_nonzero(processed_pixels == right_peak))

        previous_left_peaks = left_peak
        previous_right_peaks = right_peak

    binary_previous_peaks = get_previous_binary()
    binary_index = 0
    for index, value in np.ndenumerate(header_pixels):
        binary_value = binary_previous_peaks[binary_index]
        binary_index += 1
        header_pixels[index] = set_lsb(value, binary_value)


def assemble_image():
    global processed_image
    pixels = np.append(header_pixels, processed_pixels)
    processed_image = pixels.reshape(cover_image.shape)


def write_image():
    is_successful = cv2.imwrite('out/embedded.png', processed_image)
    if is_successful:
        print('Embedding was successful.')
    else:
        print("Embedding was unsuccessful.")


def main():
    global header_pixels, processed_pixels, binary_data_index, binary_data, peaks
    header_pixels = cover_image.ravel()[:16]
    processed_pixels = cover_image.ravel()[16:]
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
    remaining_data = binary_data[binary_data_index:]
    return processed_image, remaining_data


header_pixels = None
processed_pixels = None
binary_data_index = None
binary_data = None
is_modified = None
peaks = None

path = None
cover_image = None
hidden_data = None
iterations = None
processed_image = None
buffer = None

compress = Zlib().compress

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='The path of the cover image.', type=str)
    parser.add_argument('data', help='The data to be hidden in the cover image.', type=str)
    parser.add_argument('iterations', help='Number of iterations.', type=int)
    args = parser.parse_args()

    path = args.source
    cover_image = read_image(path)
    hidden_data = args.data
    iterations = args.iterations

    with open(hidden_data, 'rb') as data:
        hidden_data = data.read()

    main()
    write_image()
