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

        is_rounded = is_rounded.astype(np.bool)

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
        return recovered - original


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
        is_rounded = bytes_to_bits(is_modified_minimized_bytes)[:self._processed_pixels.size]
        hidden_data = bits_to_bytes(self._buffer.next(-1))
        return hidden_data, is_rounded

    def _recover_image(self, iterations, is_rounded):
        self._processed_pixels -= iterations
        scaled_max = np.max(self._processed_pixels)

        mapped_values = get_mapped_values(self._original_max - self._original_min, scaled_max)
        mapped_values = np.in1d(self._processed_pixels, mapped_values)

        recovered_pixels = scale_to(self._processed_pixels, self._original_max - self._original_min)
        recovered_pixels[mapped_values] -= is_rounded[:np.count_nonzero(mapped_values)]
        recovered_pixels += self._original_min
        self._processed_pixels = recovered_pixels


class VariableBitsScalingEmbedder(ScalingEmbedder):
    def _preprocess(self, iterations):
        original_pixels = self._processed_pixels.copy()
        self._original_min = np.min(self._processed_pixels)
        self._original_max = np.max(self._processed_pixels)
        scaled_max = MAX_PIXEL_VALUE - 2 * iterations

        self._processed_pixels = scale_to(self._processed_pixels, scaled_max)
        # mapped_values = get_mapped_values(self._original_max - self._original_min, scaled_max)
        map_sizes = get_values_freqs(self._original_max - self._original_min, scaled_max)
        # values_freq = np.where(values_freq == 0, 1, values_freq)  # so we get 0 for 0s
        # map_sizes = np.ceil(np.log2(values_freq)).astype(int)
        is_rounded = self.get_is_rounded(original_pixels, self._processed_pixels)
        is_rounded_bits = BoolDataBuffer()

        print(np.unique(is_rounded))
        # map_sizes_bits = integers_to_binary(is_rounded, 4)

        for index, value in enumerate(is_rounded):
            diff = integer_to_binary(value, map_sizes[self._processed_pixels[index]])
            is_rounded_bits.push(diff)

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

    def _recover_image(self, iterations, is_rounded):
        self._processed_pixels -= iterations
        scaled_max = np.max(self._processed_pixels)
        shifted_max = self._original_max - self._original_min

        map_sizes = get_values_freqs(shifted_max, scaled_max)
        recovered_pixels = scale_to(self._processed_pixels, shifted_max)

        is_rounded = BoolDataBuffer(is_rounded)

        for index, value in enumerate(self._processed_pixels):
            bits_length = map_sizes[value]
            diff = binary_to_integer(is_rounded.next(bits_length))
            recovered_pixels[index] -= diff

        # recovered_pixels[mapped_values] -= is_rounded[:np.count_nonzero(mapped_values)]
        # recovered_pixels += self._original_min
        self._processed_pixels = recovered_pixels + self._original_min


def get_values_freqs(original_max: int, scaled_max: int):
    og_values = np.arange(original_max + 1)
    scaled_values = scale_to(og_values, scaled_max)

    # recovered_values = scale_to(scaled_values, original_max)

    values_freq = np.bincount(scaled_values, minlength=255)
    values_freq = np.where(values_freq == 0, 1, values_freq)  # so we get 0 for 0s
    return np.ceil(np.log2(values_freq)).astype(int)


def integers_to_binary(r, m=8):
    return ((r[:, None] & (1 << np.arange(m))) > 0).astype(int)


def resize_values(pixels, sizes_map):
    for value, size in enumerate(sizes_map):
        # values = np.where(pixels == value)
        # pixels[values]
        value_bits = integers_to_binary(value, size)
        pixels[value] = value_bits


if __name__ == '__main__':
    import cv2

    im = read_image('res/f-16.png')
    embedder = VariableBitsScalingEmbedder(im, bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0), deflate)
    extractor = VariableBitsScalingExtractor()

    embedded_image, iterations, embedded_data_size = embedder.embed(100)
    recovered_image, recovery_iterations, extracted_data = extractor.extract(embedded_image)

    print(embedded_data_size)
    cv2.imwrite('out/vb_embedded_scaling.png', embedded_image)
    # vb._header_pixels, vb._processed_pixels = get_header_and_body(vb._cover_image)
    # a = vb._preprocess(100)

    # sys.exit(-1)
    # import cv2
    #
    # image = read_image('res/dataset-50/43.gif')
    # data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)
    # embedder = ScalingEmbedder(image, data)
    #
    # embedded_image, hidden_data_size, _ = embedder.embed(64)
    #
    # cv2.imwrite('out/embedded_scaling.png', embedded_image)
    #
    # for it in embedder:
    #     print(it)
