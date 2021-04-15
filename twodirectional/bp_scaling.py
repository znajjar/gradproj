from twodirectional.scaling import ScalingEmbedder, ScalingExtractor
from twodirectional.configurations import BRIGHTNESS_THRESHOLD
from util.compress import *
from util.util import *


class BPScalingEmbedder(ScalingEmbedder):
    def __init__(self, cover_image: np.ndarray, hidden_data: Iterable, compression: CompressionAlgorithm = deflate):
        ScalingEmbedder.__init__(self, cover_image, hidden_data, compression)
        self._original_brightness = np.mean(cover_image)
        self._iteration = 0

    def embed(self, iterations):
        self._iteration = 0
        return super(BPScalingEmbedder, self).embed(iterations)

    def _get_peaks(self):
        self._hist = np.bincount(self._processed_pixels, minlength=MAX_PIXEL_VALUE + 1)
        self._cum_hist = np.cumsum(self._hist)
        self._weighted_cum_hist = np.cumsum(np.arange(0, MAX_PIXEL_VALUE + 1) * self._hist)

        candidates_left = self._get_left_candidates()
        best_left_peak, best_right_peak = self._get_best_peaks(candidates_left)

        self._iteration += 1
        return best_left_peak, best_right_peak

    def _get_left_candidates(self):
        return np.sort(self._hist.argsort()[-2 - self._iteration:])

    def _get_best_peaks(self, candidates_left: np.ndarray):
        best_left_peak = 0
        best_right_peak = 0
        smallest_distance = MAX_PIXEL_VALUE
        largest_capacity = 0
        for left_peak in candidates_left:
            candidates_right = candidates_left[candidates_left > left_peak]
            for right_peak in candidates_right:
                new_distance = self._get_brightness_distance(left_peak, right_peak)
                new_capacity = self._get_capacity(left_peak, right_peak)
                if new_distance < BRIGHTNESS_THRESHOLD and largest_capacity < new_capacity or \
                        BRIGHTNESS_THRESHOLD < new_distance < smallest_distance:
                    best_left_peak = left_peak
                    best_right_peak = right_peak
                    smallest_distance = new_distance
                    largest_capacity = new_capacity

        # print(best_left_peak, best_right_peak, smallest_distance)
        return best_left_peak, best_right_peak

    def _get_brightness_distance(self, left_peak: int, right_peak: int):
        ones_in_left, ones_in_right = self._get_ones_in_peaks(left_peak, right_peak)
        new_brightness = self._get_new_brightness(left_peak, right_peak, ones_in_left, ones_in_right)
        return abs(new_brightness - self._original_brightness)

    def _get_ones_in_peaks(self, left_peak: int, right_peak: int):
        max_hide_left, max_hide_right = self._hist[left_peak], self._hist[right_peak]
        left_data, right_data = self._buffer[:max_hide_left], self._buffer[max_hide_left:max_hide_left + max_hide_right]
        ones_in_left, ones_in_right = np.sum(left_data), np.sum(right_data)
        return ones_in_left, ones_in_right

    def _get_new_brightness(self, left_peak: int, right_peak: int, ones_in_left: int, ones_in_right: int):
        return (self._weighted_cum_hist[MAX_PIXEL_VALUE] - self._cum_hist[left_peak - 1] +
                (self._cum_hist[MAX_PIXEL_VALUE] - self._cum_hist[right_peak]) - ones_in_left + ones_in_right) / \
               self._cum_hist[MAX_PIXEL_VALUE]

    def _get_capacity(self, left_peak: int, right_peak: int):
        return self._hist[left_peak] + self._hist[right_peak]


class BPScalingExtractor(ScalingExtractor):
    pass


if __name__ == '__main__':
    image = read_image('res/dataset-50/18.gif')
    np.random.seed(2115)
    data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)
    embedder = BPScalingEmbedder(image, data)

    embedded_image, iterations, hidden_data_size = embedder.embed(63)

    print(hidden_data_size)
    print(np.mean(image))
    print(np.mean(embedded_image))
    print(np.std(image))
    print(np.std(embedded_image))
    show_hist(embedded_image)

    Image.fromarray(embedded_image).save('out/bp_scaling.png')
