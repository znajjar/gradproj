import argparse

import cv2

from compress import Zlib
from data_buffer import BoolDataBuffer
from shared import *


def get_peaks(peaks):
    return binary_to_integer(peaks[:8]), binary_to_integer(peaks[8:])


def process():
    global iterations, data
    left_peak, right_peak = get_peaks(get_lsb(header_pixels))
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
    global hidden_data, data, is_modifiable, is_modified

    is_modifiable = np.logical_or(processed_pixels < 2 * iterations,
                                  processed_pixels > MAX_PIXEL_VALUE - 2 * iterations)
    is_modified = np.zeros_like(processed_pixels, dtype=np.bool)

    for index, value in np.ndenumerate(header_pixels):
        header_pixels[index] = set_lsb(value, buffer.next())

    is_modified_size_bits = buffer.next(COMPRESSED_DATA_LENGTH_BITS)
    is_modified_compressed_size = binary_to_integer(is_modified_size_bits)
    is_modified_compressed = buffer.next(is_modified_compressed_size * 8)
    is_modified_minimized_bytes = decompress(bits_to_bytes(is_modified_compressed))
    is_modified_minimize_with_extra_bits = bytes_to_bits(is_modified_minimized_bytes)
    is_modified[is_modifiable] = is_modified_minimize_with_extra_bits[:np.count_nonzero(is_modifiable)]
    hidden_data = bits_to_bytes(buffer.next(-1))


def recover_image():
    processed_pixels[np.logical_and(is_modified, processed_pixels < 128)] -= iterations
    processed_pixels[np.logical_and(is_modified, processed_pixels >= 128)] += iterations


def assemble_image():
    global cover_image
    pixels = np.append(header_pixels, processed_pixels)
    cover_image = pixels.reshape(processed_image.shape)


def write_image():
    print(cv2.imwrite('out/extracted.png', cover_image))


def main():
    global header_pixels, processed_pixels, header_pixels, processed_pixels, \
        iterations, data, hidden_data, buffer

    header_pixels = processed_image.ravel()[:16]
    processed_pixels = processed_image.ravel()[16:]

    iterations = 0
    data = []
    hidden_data = ''
    buffer = BoolDataBuffer()

    process()
    process_data()
    recover_image()
    assemble_image()


def extract(image, decompression=None):
    global processed_image, decompress
    processed_image = image.copy()
    if decompression:
        decompress = decompression
    main()
    return cover_image, iterations, hidden_data


header_pixels = None
processed_pixels = None

processed_image = None
cover_image = None
iterations = None
data = None
hidden_data = None
is_modified = None
is_modifiable = None
buffer = None

decompress = Zlib().decompress

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='The path of the image with the hidden data.', type=str)
    args = parser.parse_args()

    path = args.source
    processed_image = cv2.imread(path)[:, :, 0]
    main()
    print(hidden_data.decode('ascii'))
    write_image()
