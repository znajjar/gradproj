from util import *
from .configurations import *
from .uni_original import UnidirectionEmbedder, UnidirectionExtractor


class BPUnidirectionEmbedder(UnidirectionEmbedder):
    def __init__(self, cover_image: np.ndarray, hidden_data: Iterable, compression: CompressionAlgorithm = deflate):
        super().__init__(cover_image, hidden_data, compression)
        self._original_brightness = np.mean(cover_image)

    def _get_peaks(self):
        hist = self._get_hist()
        current_brightness = np.mean(self._body_pixels)
        if self._original_brightness - current_brightness > BRIGHTNESS_THRESHOLD:
            P_H = hist[:L - 2].argmax()
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


class ImprovedBPUnidirectionEmbedder(BPUnidirectionEmbedder):

    def _get_peaks(self):
        best_P_L, best_P_H = super()._get_peaks()
        self._hist = self._get_hist()

        if self._get_extra_space(best_P_L, best_P_H) >= 0:
            return best_P_L, best_P_H

        self._closest_left = self._get_closest_left()
        self._closest_right = self._get_closest_right()
        current_brightness = np.mean(self._body_pixels)
        if self._original_brightness - current_brightness > BRIGHTNESS_THRESHOLD:
            return self._get_best_shift_right(best_P_L, best_P_H)
        elif self._original_brightness - current_brightness < -BRIGHTNESS_THRESHOLD:
            return self._get_best_shift_left(best_P_L, best_P_H)

    def _get_closest_left(self):
        closest_left = np.empty(MAX_PIXEL_VALUE + 1, dtype=np.uint8)
        min_so_far = 0
        min_sum = self._hist[0] + self._hist[1]
        for right_peak in range(2, MAX_PIXEL_VALUE + 1):
            closest_left[right_peak] = min_so_far
            new_sum = self._hist[right_peak - 1] + self._hist[right_peak]
            if min_sum >= new_sum:
                min_sum = new_sum
                min_so_far = right_peak - 1

        return closest_left

    def _get_closest_right(self):
        closest_right = np.empty(MAX_PIXEL_VALUE + 1, dtype=np.uint8)
        min_so_far = MAX_PIXEL_VALUE
        min_sum = self._hist[MAX_PIXEL_VALUE] + self._hist[MAX_PIXEL_VALUE - 1]
        for left_peak in range(MAX_PIXEL_VALUE - 2, -1, -1):
            closest_right[left_peak] = min_so_far
            new_sum = self._hist[left_peak + 1] + self._hist[left_peak]
            if min_sum >= new_sum:
                min_sum = new_sum
                min_so_far = left_peak + 1

        return closest_right

    def _get_top_candidates(self):
        return self._hist.argsort()[::-1]

    def _get_extra_space(self, P_L, P_H):
        # location_map = self._get_location_map(P_L, P_H)
        # compressed_map = bytes_to_bits(de.gzip_compress(location_map, 1))
        # overhead_size = 17 + min(COMPRESSED_DATA_LENGTH_BITS + len(compressed_map), len(location_map))
        # return self._hist[P_H] - overhead_size
        return self._hist[P_H] - self._hist[P_L] - self._hist[P_L - get_shift_direction(P_L, P_H)]

    def _get_best_shift_right(self, best_P_L, best_P_H):
        candidates = self._get_top_candidates()
        candidates = candidates[candidates < MAX_PIXEL_VALUE - 1]
        for P_H in candidates:
            P_L = self._closest_right[P_H]
            extra_space = self._get_extra_space(P_L, P_H)
            if extra_space >= 0:
                return P_L, P_H

        return best_P_L, best_P_H

    def _get_best_shift_left(self, best_P_L, best_P_H):
        candidates = self._get_top_candidates()
        candidates = candidates[candidates >= 2]
        for P_H in candidates:
            P_L = self._closest_left[P_H]
            extra_space = self._get_extra_space(P_L, P_H)
            if extra_space >= 0:
                return P_L, P_H

        return best_P_L, best_P_H

    # def _get_new_brightness(self, P_L, P_H):
    #     d = get_shift_direction(P_L, P_H)
    #     ones_in_data = np.sum(self._buffer[:self._get_capacity(P_H)])
    #     return (self._weighted_cum_hist[MAX_PIXEL_VALUE] + d * abs(self._cum_hist[P_L - d] - self._cum_hist[P_H]) +
    #             d * ones_in_data) / self._cum_hist[MAX_PIXEL_VALUE]


class BPUnidirectionExtractor(UnidirectionExtractor):
    pass
