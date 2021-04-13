import argparse

import cv2

from util.compress import deflate
from util.data_buffer import BoolDataBuffer
from util.util import *
import util.util


def get_peaks(peaks):
    return binary_to_integer(peaks[:8]), binary_to_integer(peaks[8:])


def process():
    global iterations, data
    left_peak, right_peak = get_peaks(get_lsb(header_pixels[1:]))
    while left_peak or right_peak:
        iterations += 1

        left_peak_pixels = processed_pixels[
            np.logical_or(processed_pixels == left_peak, processed_pixels == left_peak - 1)]

        right_peak_pixels = processed_pixels[
            np.logical_or(processed_pixels == right_peak, processed_pixels == right_peak + 1)]

        iteration_data = []
        iteration_data.extend(left_peak - left_peak_pixels)
        iteration_data.extend(right_peak_pixels - right_peak)
        buffer.add(iteration_data)

        processed_pixels[processed_pixels == left_peak - 1] = left_peak
        processed_pixels[processed_pixels == right_peak + 1] = right_peak
        processed_pixels[processed_pixels < left_peak] += 1
        processed_pixels[processed_pixels > right_peak] -= 1

        binary_last_peaks = buffer.next(16)
        left_peak, right_peak = get_peaks(binary_last_peaks)


def process_data():
    global hidden_data, data, is_modifiable, is_rounded, original_min, original_max, is_rounded

    for index, value in np.ndenumerate(header_pixels):
        header_pixels[index] = set_lsb(value, buffer.next())

    original_min = binary_to_integer(buffer.next(8))
    original_max = binary_to_integer(buffer.next(8))

    is_modified_size_bits = buffer.next(COMPRESSED_DATA_LENGTH_BITS)
    is_modified_compressed_size = binary_to_integer(is_modified_size_bits)
    is_modified_compressed = buffer.next(is_modified_compressed_size * 8)
    is_modified_minimized_bytes = decompress(bits_to_bytes(is_modified_compressed))
    is_rounded = bytes_to_bits(is_modified_minimized_bytes)[:processed_pixels.size]
    hidden_data = bits_to_bytes(buffer.next(-1))


def recover_image():
    global processed_pixels, recovered_pixels
    processed_pixels -= iterations
    scaled_max = np.max(processed_pixels)

    mapped_values = get_mapped_values(original_max - original_min, scaled_max)
    mapped_values = np.in1d(processed_pixels, mapped_values)

    recovered_pixels = scale_to(processed_pixels, original_max - original_min)
    recovered_pixels[mapped_values] -= is_rounded[:np.count_nonzero(mapped_values)]
    recovered_pixels += original_min


def _assemble_image():
    global cover_image
    cover_image = util.assemble_image(header_pixels, processed_pixels, processed_image.shape)


def write_image():
    print(cv2.imwrite('out/extracted_with_scaling.png', cover_image))


def main():
    global header_pixels, processed_pixels, header_pixels, processed_pixels, \
        iterations, data, hidden_data, buffer, cover_image

    header_pixels, processed_pixels = get_header_and_body(processed_image, 17)

    iterations = 0
    data = []
    hidden_data = ''
    buffer = BoolDataBuffer()
    buffer.set_parity(get_lsb([header_pixels[0]]))

    process()
    process_data()
    recover_image()
    cover_image = assemble_image(header_pixels, recovered_pixels, processed_image.shape)


def extract(image, decompression=None):
    global processed_image, decompress
    processed_image = image
    if decompression:
        decompress = decompression.decompress
    main()
    return cover_image, iterations, hidden_data


header_pixels = None
processed_pixels = None
recovered_pixels = None

processed_image = None
cover_image = None
iterations = None
data = None
hidden_data = None
is_rounded = None
is_modifiable = None
buffer = None

original_min = None
original_max = None

decompress = deflate.decompress

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='The path of the image with the hidden data.', type=str)
    args = parser.parse_args()

    path = args.source
    processed_image = cv2.imread(path)[:, :, 0]
    main()
    print(hidden_data.decode('ascii'))
    write_image()
