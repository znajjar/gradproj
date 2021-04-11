from configurations import *
from util.compress import CompressionAlgorithm, deflate
from util.data_buffer import BoolDataBuffer
from util.util import *


class OriginalEmbedder:
    def __init__(self, cover_image: np.ndarray, hidden_data: Iterable, compression: CompressionAlgorithm = deflate):
        self._cover_image = cover_image
        self._hidden_data = bytes_to_bits(hidden_data)
        self._compress = compression.compress

        self._processed_pixels = None
        self._header_pixels = None
        self._buffer = BoolDataBuffer()

    def embed(self, iterations):
        if iterations > 64:
            raise TypeError
        self._header_pixels, self._processed_pixels = get_header_and_body(self._cover_image)
        is_modified = self._preprocess(iterations)
        self._fill_buffer(is_modified)
        self._process(iterations)
        embedded_image = assemble_image(self._header_pixels, self._processed_pixels, self._cover_image.shape)
        return embedded_image, len(self._hidden_data) - len(self._buffer.next(-1))

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
        self._buffer = BoolDataBuffer(is_modified_size_bits, is_modified_bits, self._hidden_data)

    def _process(self, iterations):
        previous_left_peaks = previous_right_peaks = 0

        def get_previous_binary():
            ret = []
            ret.extend(integer_to_binary(previous_left_peaks))
            ret.extend(integer_to_binary(previous_right_peaks))
            return ret

        self._buffer.add(get_lsb(self._header_pixels))

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
        if not hasattr(self, 'index'):
            self.index = 0

        if self.index < 64:
            self.index += 1
            print(self.index)
            return self.embed(self.index)

        self.index = 0
        raise StopIteration
