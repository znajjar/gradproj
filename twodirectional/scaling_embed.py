from original_embed import OriginalEmbedder
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


if __name__ == '__main__':
    import cv2

    image = read_image('res/f-16.png')
    data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)
    embedder = ScalingEmbedder(image, data)

    embedded_image, hidden_data_size = embedder.embed(64)

    cv2.imwrite('out/embedded_scaling.png', embedded_image)

    for it in embedder:
        print(it)
