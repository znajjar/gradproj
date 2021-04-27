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


if __name__ == '__main__':
    import cv2

    image = read_image('res/dataset-50/43.gif')
    data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)
    embedder = ScalingEmbedder(image, data)

    embedded_image, hidden_data_size, _ = embedder.embed(64)

    cv2.imwrite('out/embedded_scaling.png', embedded_image)

    for it in embedder:
        print(it)
