from unidirection.configurations import *
from unidirection.uni_original import UnidirectionEmbedder, UnidirectionExtractor
from util import *


class BPUnidirectionEmbedder(UnidirectionEmbedder):
    def __init__(self, cover_image: np.ndarray, hidden_data: Iterable, compression: CompressionAlgorithm = deflate):
        super().__init__(cover_image, hidden_data, compression)
        self._original_brightness = np.mean(cover_image)

    def _get_peaks(self):
        hist = self._get_hist()
        current_brightness = np.mean(self._body_pixels)
        if self._original_brightness - current_brightness > BRIGHTNESS_THRESHOLD:
            P_H = hist[:MAX_PIXEL_VALUE - 1].argmax()
        elif self._original_brightness - current_brightness < -BRIGHTNESS_THRESHOLD:
            P_H = hist[2:].argmax() + 2
        else:
            P_H = hist.argmax()

        if self._original_brightness - current_brightness > BRIGHTNESS_THRESHOLD or P_H < 2:
            P_L = get_minimum_closest_right(hist, P_H)
        elif self._original_brightness - current_brightness < -BRIGHTNESS_THRESHOLD or P_H > 253:
            P_L = get_minimum_closest_left(hist, P_H)
        else:
            P_L = get_minimum_closest(hist, P_H)

        return P_L, P_H


class BPUnidirectionExtractor(UnidirectionExtractor):
    pass
