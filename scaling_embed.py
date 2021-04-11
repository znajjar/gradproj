import argparse

import cv2

import shared
from compress import Deflate
from data_buffer import BoolDataBuffer
from measure import Measure
from shared import *


def get_is_rounded(original, processed):
    recovered = scale_to(processed, (np.min(original), np.max(original)))
    return (recovered - original).astype(np.bool)


def preprocess():
    global is_rounded, original_min, original_max, processed_pixels

    original_pixels = processed_pixels.copy()
    original_min = np.min(processed_pixels)
    original_max = np.max(processed_pixels)
    scaled_max = MAX_PIXEL_VALUE - 2 * iterations

    processed_pixels = scale_to(processed_pixels, scaled_max)
    mapped_values = get_mapped_values(original_max - original_min, scaled_max)
    is_rounded = get_is_rounded(original_pixels, processed_pixels)[np.in1d(processed_pixels, mapped_values)]

    processed_pixels += iterations


def fill_buffer():
    global buffer, parity
    is_modified_compressed = compress(bits_to_bytes(is_rounded))
    is_modified_size_bits = integer_to_binary(len(is_modified_compressed), COMPRESSED_DATA_LENGTH_BITS)
    is_modified_bits = bytes_to_bits(is_modified_compressed)
    hidden_data_bits = bytes_to_bits(hidden_data)
    buffer = BoolDataBuffer(is_modified_size_bits, is_modified_bits, hidden_data_bits, calc_parity=True)
    parity = buffer.get_parity()
    buffer.next = Measure(buffer.next)
    buffer.add = Measure(buffer.add)


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

    header_pixels[0] = set_lsb(header_pixels[0], parity)
    binary_previous_peaks = get_previous_binary()
    binary_index = 0
    for index in range(1, header_pixels.size):
        binary_value = binary_previous_peaks[binary_index]
        binary_index += 1
        header_pixels[index] = set_lsb(header_pixels[index], binary_value)


def _assemble_image():
    global processed_image
    processed_image = shared.assemble_image(header_pixels, processed_pixels, cover_image.shape)


def write_image():
    is_successful = cv2.imwrite('out/embedded_with_scaling.png', processed_image)
    if is_successful:
        print('Embedding was successful.')
    else:
        print("Embedding was unsuccessful.")


def main():
    global header_pixels, processed_pixels, binary_data_index, binary_data, peaks, processed_image
    header_pixels, processed_pixels = get_header_and_body(cover_image, 17)
    binary_data_index = 0
    binary_data = []
    peaks = []

    preprocess()
    fill_buffer()
    process()
    processed_image = assemble_image(header_pixels, processed_pixels, cover_image.shape)


def embed(image, data_to_be_hidden, iterations_count=32, compression=None) -> (np.ndarray, iter):
    global cover_image, hidden_data, iterations, compress
    cover_image = image
    hidden_data = data_to_be_hidden
    iterations = iterations_count
    if compression:
        compress = compression.compress
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

compress = Deflate.compress

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

    preprocess = Measure(preprocess)
    fill_buffer = Measure(fill_buffer)
    process = Measure(process)
    get_peaks = Measure(get_peaks)

    main()
    write_image()

    print(f'preprocess time {preprocess.get_total()}')
    print(f'fill buffer time {fill_buffer.get_total()}')
    print(f'process time {process.get_total()}')
    print(f'get peaks time {get_peaks.get_total()}')
    print(f'buffer add time {buffer.add.get_total()}')
    print(f'buffer next time {buffer.next.get_total()}')
