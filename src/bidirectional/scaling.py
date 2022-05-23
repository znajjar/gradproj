from bidirectional.original import *
from util.util import *


class ScalingEmbedder(OriginalEmbedder):
    _ITERATIONS_LIMIT = 10000

    def _preprocess(self, iterations):
        original_pixels = self._processed_pixels.copy()
        self._original_min = np.min(self._processed_pixels)
        self._original_max = np.max(self._processed_pixels)
        scaled_max = MAX_PIXEL_VALUE - 2 * iterations

        self._processed_pixels = scale_to(self._processed_pixels, scaled_max)
        mapped_values = get_mapped_values(self._original_max - self._original_min, scaled_max)
        is_rounded = self.get_is_rounded(original_pixels, self._processed_pixels)

        if len(np.unique(is_rounded)) > 2:
            raise ValueError(super()._ITERATIONS_LIMIT_EXCEEDED_ERROR)

        is_rounded = is_rounded.astype(bool)

        is_rounded = is_rounded[np.in1d(self._processed_pixels, mapped_values)]

        self._processed_pixels += iterations

        return is_rounded

    def _get_overhead(self):
        return (
            get_lsb(self._header_pixels),
            integer_to_binary(self._original_min),
            integer_to_binary(self._original_max),
        )

    @staticmethod
    def get_is_rounded(original, processed):
        recovered = scale_to(processed, (np.min(original), np.max(original)))
        return original - recovered


class ScalingExtractor(OriginalExtractor):
    def _process_data(self, iterations):
        for index, value in np.ndenumerate(self._header_pixels):
            self._header_pixels[index] = set_lsb(value, self._buffer.next())

        self._original_min = binary_to_integer(self._buffer.next(8))
        self._original_max = binary_to_integer(self._buffer.next(8))

        is_modified_size_bits = self._buffer.next(COMPRESSED_DATA_LENGTH_BITS)
        is_modified_compressed_size = binary_to_integer(is_modified_size_bits)
        is_modified_compressed = self._buffer.next(is_modified_compressed_size * 8)
        is_modified_minimized_bytes = self._decompress(bits_to_bytes(is_modified_compressed))
        is_rounded = bytes_to_bits(is_modified_minimized_bytes)
        hidden_data = bits_to_bytes(self._buffer.next(-1))
        return hidden_data, is_rounded

    def _unpack_is_modified(self, is_modified_packed, iterations):
        return is_modified_packed[:self._processed_pixels.size]

    def _recover_image(self, iterations, is_rounded):
        self._processed_pixels -= iterations
        scaled_max = np.max(self._processed_pixels)

        mapped_values = get_mapped_values(self._original_max - self._original_min, scaled_max)
        mapped_values = np.in1d(self._processed_pixels, mapped_values)

        recovered_pixels = scale_to(self._processed_pixels, self._original_max - self._original_min)
        recovered_pixels[mapped_values] += is_rounded[:np.count_nonzero(mapped_values)]
        recovered_pixels += self._original_min
        self._processed_pixels = recovered_pixels


class ValueOrderScalingEmbedder(ScalingEmbedder):
    def _preprocess(self, iterations):
        original_pixels = self._processed_pixels.copy()
        self._original_min = np.min(self._processed_pixels)
        self._original_max = np.max(self._processed_pixels)
        scaled_max = MAX_PIXEL_VALUE - 2 * iterations

        self._processed_pixels = scale_to(self._processed_pixels, scaled_max)
        mapped_values = get_mapped_values(self._original_max - self._original_min, scaled_max)
        is_rounded = self.get_is_rounded(original_pixels, self._processed_pixels)

        if len(np.unique(is_rounded)) > 2:
            raise ValueError(super()._ITERATIONS_LIMIT_EXCEEDED_ERROR)

        is_rounded = is_rounded.astype(np.bool)

        ordered_is_rounded = BoolDataBuffer()

        for value in range(256):
            pixels_with_value = self._processed_pixels == value
            if value in mapped_values:
                ordered_is_rounded.push(is_rounded[pixels_with_value])

        self._processed_pixels += iterations

        return ordered_is_rounded.next(-1)


class ValueOrderedScalingExtractor(ScalingExtractor):
    def _recover_image(self, iterations, is_rounded):
        self._processed_pixels -= iterations
        scaled_max = np.max(self._processed_pixels)

        mapped_values = get_mapped_values(self._original_max - self._original_min, scaled_max)
        recovered_pixels = scale_to(self._processed_pixels, self._original_max - self._original_min)

        is_rounded = BoolDataBuffer(is_rounded)
        for value in range(256):
            pixels_with_value = self._processed_pixels == value
            if value in mapped_values:
                recovered_pixels[pixels_with_value] -= is_rounded.next(np.count_nonzero(pixels_with_value))

        recovered_pixels += self._original_min
        self._processed_pixels = recovered_pixels


class VariableBitsScalingEmbedder(ScalingEmbedder):
    def __init__(self, cover_image: np.ndarray,
                 hidden_data: Iterable,
                 compression: CompressionAlgorithm = deflate,
                 bit_limit=2):
        super().__init__(cover_image, hidden_data, compression)
        self._bit_limit = bit_limit

    def _preprocess(self, iterations):
        original_pixels = self._processed_pixels.copy()
        self._original_min = np.min(self._processed_pixels)
        self._original_max = np.max(self._processed_pixels)
        scaled_max = MAX_PIXEL_VALUE - 2 * iterations

        self._processed_pixels = scale_to(self._processed_pixels, scaled_max)
        # mapped_values = get_mapped_values(self._original_max - self._original_min, scaled_max)
        map_sizes = get_values_freqs(self._original_max - self._original_min, scaled_max)
        if np.max(map_sizes) > self._bit_limit:
            raise ValueError

        is_rounded = self.get_is_rounded(original_pixels, self._processed_pixels)
        is_rounded_bits = BoolDataBuffer()

        # map_sizes_bits = integers_to_binary(is_rounded, 4)

        for value in range(256):
            pixels_with_value = self._processed_pixels == value
            is_rounded_bits.push(integers_to_bits(is_rounded[pixels_with_value], map_sizes[value]))

        self._processed_pixels += iterations

        return is_rounded_bits.next(-1)


class VariableBitsScalingExtractor(ScalingExtractor):
    def _process_data(self, iterations):
        for index, value in np.ndenumerate(self._header_pixels):
            self._header_pixels[index] = set_lsb(value, self._buffer.next())

        self._original_min = binary_to_integer(self._buffer.next(8))
        self._original_max = binary_to_integer(self._buffer.next(8))

        is_modified_size_bits = self._buffer.next(COMPRESSED_DATA_LENGTH_BITS)
        is_modified_compressed_size = binary_to_integer(is_modified_size_bits)
        is_modified_compressed = self._buffer.next(is_modified_compressed_size * 8)
        is_modified_minimized_bytes = self._decompress(bits_to_bytes(is_modified_compressed))
        is_rounded = bytes_to_bits(is_modified_minimized_bytes)
        hidden_data = bits_to_bytes(self._buffer.next(-1))
        return hidden_data, is_rounded

    def _unpack_is_modified(self, is_modified_packed, iterations):
        return is_modified_packed

    def _recover_image(self, iterations, is_rounded):
        self._processed_pixels -= iterations
        scaled_max = np.max(self._processed_pixels)
        shifted_max = self._original_max - self._original_min

        map_sizes = get_values_freqs(shifted_max, scaled_max)
        recovered_pixels = scale_to(self._processed_pixels, shifted_max)

        is_rounded = BoolDataBuffer(is_rounded)

        for value in range(256):
            pixels_with_value = self._processed_pixels == value
            bits_length = map_sizes[value]
            if not bits_length:
                continue
            bits_count = np.count_nonzero(pixels_with_value) * bits_length
            bits = is_rounded.next(bits_count)
            is_rounded_value = bits_to_integers(bits, bits_length)
            recovered_pixels[pixels_with_value] += is_rounded_value

        self._processed_pixels = recovered_pixels + self._original_min


def get_values_freqs(original_max: int, scaled_max: int):
    og_values = np.arange(original_max + 1)
    scaled_values = scale_to(og_values, scaled_max)

    # recovered_values = scale_to(scaled_values, original_max)

    values_freq = np.bincount(scaled_values, minlength=256)
    values_freq = np.where(values_freq == 0, 1, values_freq)  # so we get 0 for 0s
    return np.ceil(np.log2(values_freq)).astype(int)


def integers_to_bits(r, m=8):
    return ((r[:, None] & (1 << np.arange(m))) > 0).ravel().astype(bool)


def bits_to_integers(r: np.ndarray, m=8):
    return np.packbits(r.reshape((r.size // m, m)), bitorder='little', axis=1).ravel()


def resize_values(pixels, sizes_map):
    for value, size in enumerate(sizes_map):
        value_bits = integers_to_bits(value, size)
        pixels[value] = value_bits


if __name__ == '__main__':
    import cv2

    image = read_image('res/mo3tamad/5.3.01.tiff')
    data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)
    embedder = VariableBitsScalingEmbedder(image.copy(), data, bit_limit=2)
    extractor = VariableBitsScalingExtractor()

    embedded_image, iterations, embedded_data_size = embedder.embed(96)
    recovered_image, recovery_iterations, extracted_data = extractor.extract(embedded_image)

    print(f'difference: {np.sum(np.abs(image - recovered_image))} \n'
          f'hidden data size: {8 * len(extracted_data)}')
    cv2.imwrite('out/embedded_scaling.png', embedded_image)
    cv2.imwrite('out/extracted_scaling.png', recovered_image)

