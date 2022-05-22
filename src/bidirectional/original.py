import numpy as np

from bidirectional.configurations import *
from util import *


class OriginalEmbedder:
    _ITERATIONS_LIMIT = 64
    _ITERATIONS_LIMIT_EXCEEDED_ERROR = 'Exceeded the max number of iterations allowed.'

    def __init__(self, cover_image: np.ndarray, hidden_data: Iterable, compression: CompressionAlgorithm = deflate):
        self._cover_image = cover_image
        self._hidden_data = bytes_to_bits(hidden_data)
        self._compress = compression.compress

        self._processed_pixels = None
        self._header_pixels = None
        self._buffer = BoolDataBuffer()

    def embed(self, iterations):
        if iterations > self._ITERATIONS_LIMIT:
            raise ValueError(self._ITERATIONS_LIMIT_EXCEEDED_ERROR)
        self._header_pixels, self._processed_pixels = get_header_and_body(self._cover_image, HEADER_SIZE)
        is_modified = self._preprocess(iterations)
        self._fill_buffer(is_modified)
        self._process(iterations)
        embedded_image = assemble_image(self._header_pixels, self._processed_pixels, self._cover_image.shape)
        return embedded_image, iterations, len(self._hidden_data) - len(self._buffer.next(-1))

    def _preprocess(self, iterations):
        is_modified = np.zeros_like(self._processed_pixels, dtype=np.bool)
        lower_bound = self._processed_pixels < iterations
        upper_bound = MAX_PIXEL_VALUE - iterations < self._processed_pixels
        is_modified |= lower_bound
        is_modified |= upper_bound
        self._processed_pixels[lower_bound] += iterations
        self._processed_pixels[upper_bound] -= iterations
        is_modifiable = np.logical_or(self._processed_pixels < 2 * iterations,
                                      self._processed_pixels > MAX_PIXEL_VALUE - 2 * iterations)
        is_modified = is_modified[is_modifiable]
        return is_modified

    def _fill_buffer(self, is_modified):
        self._buffer.clear()
        is_modified_compressed = self._compress(bits_to_bytes(is_modified))
        is_modified_size_bits = integer_to_binary(len(is_modified_compressed), COMPRESSED_DATA_LENGTH_BITS)
        is_modified_bits = bytes_to_bits(is_modified_compressed)
        self._buffer = BoolDataBuffer(*self._get_overhead(), is_modified_size_bits, is_modified_bits, self._hidden_data)

    def _get_overhead(self):
        return get_lsb(self._header_pixels),

    def _process(self, iterations):
        previous_left_peaks = previous_right_peaks = 0

        def get_previous_binary():
            ret = []
            ret.extend(integer_to_binary(previous_left_peaks))
            ret.extend(integer_to_binary(previous_right_peaks))
            return ret

        while iterations:
            iterations -= 1
            left_peak, right_peak = self._get_peaks()

            self._processed_pixels[self._processed_pixels < left_peak] -= 1
            self._processed_pixels[self._processed_pixels > right_peak] += 1

            binary_previous_peaks = get_previous_binary()

            self._buffer.add(binary_previous_peaks)

            self._processed_pixels[self._processed_pixels == left_peak] -= self._buffer.next(
                np.count_nonzero(self._processed_pixels == left_peak))
            self._processed_pixels[self._processed_pixels == right_peak] += self._buffer.next(
                np.count_nonzero(self._processed_pixels == right_peak))

            previous_left_peaks = left_peak
            previous_right_peaks = right_peak

        self._header_pixels[0] = set_lsb(self._header_pixels[0], self._buffer.get_parity())
        binary_previous_peaks = get_previous_binary()
        binary_index = 0
        for index in range(1, self._header_pixels.size):
            binary_value = binary_previous_peaks[binary_index]
            binary_index += 1
            self._header_pixels[index] = set_lsb(self._header_pixels[index], binary_value)

    def _get_peaks(self):
        hist = np.bincount(self._processed_pixels)
        return np.sort(hist.argsort()[-2:])

    def __iter__(self):
        return self

    def __next__(self):
        if not hasattr(self, '_index'):
            self._index = 0

        try:
            self._index += 1
            return self.embed(self._index)
        except ValueError:
            self.index = 0
            raise StopIteration


class OriginalExtractor:
    def __init__(self, compression=deflate):
        self._decompress = compression.decompress

        self._header_pixels = None
        self._processed_pixels = None
        self._buffer = BoolDataBuffer()

    @staticmethod
    def _get_peaks(peaks):
        return binary_to_integer(peaks[:8]), binary_to_integer(peaks[8:])

    def extract(self, embedded_image):
        embedded_image = embedded_image.copy()
        self._header_pixels, self._processed_pixels = get_header_and_body(embedded_image, HEADER_SIZE)

        iterations = self._process()
        hidden_data, is_modified_packed = self._process_data(iterations)
        is_modified = self._unpack_is_modified(is_modified_packed, iterations)
        self._recover_image(iterations, is_modified)

        cover_image = assemble_image(self._header_pixels, self._processed_pixels, embedded_image.shape)

        return cover_image, iterations, hidden_data

    def _process(self):
        iterations = 0
        parity = get_lsb([self._header_pixels[0]])
        self._buffer.set_parity(parity)
        left_peak, right_peak = self._get_peaks(get_lsb(self._header_pixels[1:]))
        while left_peak or right_peak:
            iterations += 1

            left_peak_pixels = self._processed_pixels[
                np.logical_or(self._processed_pixels == left_peak, self._processed_pixels == left_peak - 1)]

            right_peak_pixels = self._processed_pixels[
                np.logical_or(self._processed_pixels == right_peak, self._processed_pixels == right_peak + 1)]

            iteration_data = []
            iteration_data.extend(left_peak - left_peak_pixels)
            iteration_data.extend(right_peak_pixels - right_peak)
            self._buffer.add(iteration_data)

            self._processed_pixels[self._processed_pixels == left_peak - 1] = left_peak
            self._processed_pixels[self._processed_pixels == right_peak + 1] = right_peak
            self._processed_pixels[self._processed_pixels < left_peak] += 1
            self._processed_pixels[self._processed_pixels > right_peak] -= 1

            binary_last_peaks = self._buffer.next(16)
            left_peak, right_peak = self._get_peaks(binary_last_peaks)

        return iterations

    def _process_data(self, iterations):
        for index, value in np.ndenumerate(self._header_pixels):
            self._header_pixels[index] = set_lsb(value, self._buffer.next())

        is_modified_size_bits = self._buffer.next(COMPRESSED_DATA_LENGTH_BITS)
        is_modified_compressed_size = binary_to_integer(is_modified_size_bits)
        is_modified_compressed = self._buffer.next(is_modified_compressed_size * 8)
        is_modified_minimized_bytes = self._decompress(bits_to_bytes(is_modified_compressed))
        is_modified = bytes_to_bits(is_modified_minimized_bytes)
        hidden_data = bits_to_bytes(self._buffer.next(-1))
        return hidden_data, is_modified

    def _unpack_is_modified(self, is_modified_packed, iterations):
        is_modified = np.zeros_like(self._processed_pixels, dtype=np.bool)

        is_modifiable = np.logical_or(self._processed_pixels < 2 * iterations,
                                      self._processed_pixels > MAX_PIXEL_VALUE - 2 * iterations)

        is_modified[is_modifiable] = is_modified_packed[:np.count_nonzero(is_modifiable)]
        return is_modified

    def _recover_image(self, iterations, is_modified):
        self._processed_pixels[np.logical_and(is_modified, self._processed_pixels < 128)] -= iterations
        self._processed_pixels[np.logical_and(is_modified, self._processed_pixels >= 128)] += iterations


class ValueOrderedOriginalEmbedder(OriginalEmbedder):
    def _preprocess(self, iterations):
        is_modified = np.zeros_like(self._processed_pixels, dtype=np.bool)
        lower_bound = self._processed_pixels < iterations
        upper_bound = MAX_PIXEL_VALUE - iterations < self._processed_pixels
        is_modified |= lower_bound
        is_modified |= upper_bound
        self._processed_pixels[lower_bound] += iterations
        self._processed_pixels[upper_bound] -= iterations

        is_modified_ordered = BoolDataBuffer()

        for value in range(256):
            pixels_with_value = self._processed_pixels == value
            if not MAX_PIXEL_VALUE - 2 * iterations < value < 2 * iterations:
                is_modified_ordered.push(is_modified[pixels_with_value])
        return is_modified_ordered.next(-1)


class ValueOrderedOriginalExtractor(OriginalExtractor):
    def _unpack_is_modified(self, is_modified_packed, iterations):
        return is_modified_packed

    def _recover_image(self, iterations, is_modified_decompressed):
        is_modified_decompressed = BoolDataBuffer(is_modified_decompressed)
        for value in range(256):
            pixels_with_value = self._processed_pixels == value
            pixels_count = np.count_nonzero(pixels_with_value)
            sign = 1 if value >= 128 else -1
            rec_value = value + sign * iterations
            if not MAX_PIXEL_VALUE - 2 * iterations < value < 2 * iterations:
                is_modified_value = np.zeros_like(pixels_with_value)
                is_modified_value[pixels_with_value] = (is_modified_decompressed.next(pixels_count))
                self._processed_pixels[is_modified_value] = rec_value


class NeighboringBinsEmbedder(OriginalEmbedder):
    def _preprocess(self, iterations):
        is_modified = np.zeros_like(self._processed_pixels, dtype=np.bool)

        lower_bound = self._processed_pixels < 2 * iterations
        lower_bound_modified = np.logical_and(lower_bound, self._processed_pixels % 2 == 0)

        upper_bound = self._processed_pixels > MAX_PIXEL_VALUE - 2 * iterations
        upper_bound_modified = np.logical_and(upper_bound, self._processed_pixels % 2 == 1)

        is_modified |= lower_bound_modified
        is_modified |= upper_bound_modified

        self._processed_pixels[lower_bound_modified] += 1
        self._processed_pixels[upper_bound_modified] -= 1

        self._processed_pixels[lower_bound] += (
                ((2 * iterations - 1) - self._processed_pixels[lower_bound]) / 2).astype(np.uint8)
        self._processed_pixels[upper_bound] -= ((self._processed_pixels[upper_bound] - (
                MAX_PIXEL_VALUE - 2 * iterations + 1)) / 2).astype(np.uint8)

        is_modifiable = np.logical_or(self._processed_pixels < 2 * iterations,
                                      self._processed_pixels > MAX_PIXEL_VALUE - 2 * iterations)
        is_modified = is_modified[is_modifiable]
        return is_modified


class NeighboringBinsExtractor(OriginalExtractor):
    def _recover_image(self, iterations, is_modified_decompressed):
        lower_bound = self._processed_pixels < 2 * iterations
        upper_bound = self._processed_pixels > MAX_PIXEL_VALUE - 2 * iterations

        self._processed_pixels[lower_bound] -= (
            ((2 * iterations - 1) - self._processed_pixels[lower_bound])).astype(np.uint8)
        self._processed_pixels[upper_bound] += ((self._processed_pixels[upper_bound] - (
                MAX_PIXEL_VALUE - 2 * iterations + 1))).astype(np.uint8)

        self._processed_pixels[np.logical_and(is_modified_decompressed, self._processed_pixels < 128)] -= 1
        self._processed_pixels[np.logical_and(is_modified_decompressed, self._processed_pixels >= 128)] += 1


class NbVoEmbedder(OriginalEmbedder):
    def _preprocess(self, iterations):
        is_modified = np.zeros_like(self._processed_pixels, bool)

        lower_bound = self._processed_pixels < 2 * iterations
        lower_bound_modified = np.logical_and(lower_bound, self._processed_pixels % 2 == 0)

        upper_bound = self._processed_pixels > MAX_PIXEL_VALUE - 2 * iterations
        upper_bound_modified = np.logical_and(upper_bound, self._processed_pixels % 2 == 1)

        is_modified |= lower_bound_modified
        is_modified |= upper_bound_modified

        self._processed_pixels[lower_bound_modified] += 1
        self._processed_pixels[upper_bound_modified] -= 1

        self._processed_pixels[lower_bound] += (
                ((2 * iterations - 1) - self._processed_pixels[lower_bound]) / 2).astype(np.uint8)
        self._processed_pixels[upper_bound] -= ((self._processed_pixels[upper_bound] - (
                MAX_PIXEL_VALUE - 2 * iterations + 1)) / 2).astype(np.uint8)

        is_modifiable = np.logical_or(self._processed_pixels < 2 * iterations,
                                      self._processed_pixels > MAX_PIXEL_VALUE - 2 * iterations)
        is_modified = is_modified[is_modifiable]

        is_modified_ordered = BoolDataBuffer()

        for value in range(256):
            pixels_with_value = self._processed_pixels == value
            if not MAX_PIXEL_VALUE - 2 * iterations < value < 2 * iterations:
                is_modified_ordered.push(is_modified[pixels_with_value])
        return is_modified_ordered.next(-1)


class NbVoExtractor(OriginalExtractor):
    def _unpack_is_modified(self, is_modified_packed, iterations):
        return is_modified_packed

    def _recover_image(self, iterations, is_modified_decompressed):
        recovered_pixels = self._processed_pixels.copy()
        is_modified_decompressed = BoolDataBuffer(is_modified_decompressed)
        for value in range(256):
            pixels_with_value = self._processed_pixels == value
            pixels_count = np.count_nonzero(pixels_with_value)

            if not MAX_PIXEL_VALUE - 2 * iterations < value < 2 * iterations:
                is_modified_value = np.zeros_like(pixels_with_value)
                is_modified_value[pixels_with_value] = (is_modified_decompressed.next(pixels_count))

                if value >= 128:
                    recovered_pixels[pixels_with_value] += value - (MAX_PIXEL_VALUE - 2 * iterations + 1)
                    recovered_pixels[is_modified_value] += 1
                else:
                    recovered_pixels[pixels_with_value] -= 2 * iterations - 1 - value
                    recovered_pixels[is_modified_value] -= 1
        self._processed_pixels = recovered_pixels


if __name__ == '__main__':
    import cv2

    image = read_image('res/dataset-50/2.gif')
    data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)
    embedder = NbVoEmbedder(image.copy(), data)
    extractor = NbVoExtractor()

    embedded, hidden_data_size, _ = embedder.embed(64)
    cv2.imwrite('out/embedded.png', embedded)

    extracted, iteration_count, extracted_data = extractor.extract(embedded)
    cv2.imwrite('out/extracted.png', extracted)

    print(f'difference: {np.sum(np.abs(image - extracted))} \n'
          f'hidden data size: {8 * len(extracted_data)}')

    # for it in embedder:
    #     print(it)
