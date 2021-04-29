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


class ImprovedBPUnidirectionEmbedder(BPUnidirectionEmbedder):

    def _get_peaks(self):
        self._hist = self._get_hist()
        current_brightness = np.mean(self._body_pixels)
        if self._original_brightness - current_brightness > BRIGHTNESS_THRESHOLD:
            return self._get_best_shift_right()
        elif self._original_brightness - current_brightness < -BRIGHTNESS_THRESHOLD:
            return self._get_best_shift_left()
        else:
            return self._get_best_shift()

    def _get_best_shift_right(self):
        closest_right = self._get_closest_right()
        max_diff_right = self._get_extra_space(closest_right, np.arange(0, MAX_PIXEL_VALUE + 1), RIGHT_DIRECTION)
        best_P_H = max_diff_right[:MAX_PIXEL_VALUE - 1].argmax()
        best_P_L = closest_right[best_P_H]

        return best_P_L, best_P_H

    def _get_closest_right(self):
        closest_right = np.zeros(MAX_PIXEL_VALUE + 1, dtype=np.uint8)
        min_so_far = MAX_PIXEL_VALUE
        min_sum = self._hist[MAX_PIXEL_VALUE] + self._hist[MAX_PIXEL_VALUE - 1]
        for left_peak in range(MAX_PIXEL_VALUE - 2, -1, -1):
            closest_right[left_peak] = min_so_far
            new_sum = self._hist[left_peak + 1] + self._hist[left_peak]
            if min_sum >= new_sum:
                min_sum = new_sum
                min_so_far = left_peak + 1

        return closest_right

    def _get_best_shift_left(self):
        closest_left = self._get_closest_left()
        max_diff_left = self._get_extra_space(closest_left, np.arange(0, MAX_PIXEL_VALUE + 1), LEFT_DIRECTION)
        best_P_H = max_diff_left[2:].argmax() + 2
        best_P_L = closest_left[best_P_H]

        return best_P_L, best_P_H

    def _get_closest_left(self):
        closest_left = np.zeros(MAX_PIXEL_VALUE + 1, dtype=np.uint8)
        min_so_far = 0
        min_sum = self._hist[0] + self._hist[1]
        for right_peak in range(2, MAX_PIXEL_VALUE + 1):
            closest_left[right_peak] = min_so_far
            new_sum = self._hist[right_peak - 1] + self._hist[right_peak]
            if min_sum >= new_sum:
                min_sum = new_sum
                min_so_far = right_peak - 1

        return closest_left

    def _get_best_shift(self):
        P_L_left, P_H_left = self._get_best_shift_left()
        P_L_right, P_H_right = self._get_best_shift_right()

        if self._get_extra_space(P_L_left, P_H_left, LEFT_DIRECTION) > \
                self._get_extra_space(P_L_right, P_H_right, RIGHT_DIRECTION):
            return P_L_left, P_H_left
        else:
            return P_L_right, P_H_right

    def _get_extra_space(self, P_L, P_H, direction) -> np.ndarray:
        location_map_size = np.asarray(self._hist[P_L] + self._hist[P_L - direction], dtype=float)
        percentage = np.minimum(self._hist[P_L], self._hist[P_L - direction]) / location_map_size

        compressed_map_size = location_map_size.copy()

        low_range = np.logical_and(location_map_size > 0, location_map_size <= 1500)
        compressed_map_size[low_range] = 2.4274572 * percentage[low_range] ** 0.3849364 * location_map_size[
            low_range] ** -0.0395816 + 180.07035 / location_map_size[low_range]

        high_range = location_map_size > 1500
        compressed_map_size[high_range] = (2.4274572 * percentage[high_range] ** 0.3849364 - 0.1690710) * \
                                          location_map_size[high_range] ** -0.0395816

        compressed_map_size = compressed_map_size * location_map_size

        return self._hist[P_H] - np.minimum(location_map_size, compressed_map_size)


class BPUnidirectionExtractor(UnidirectionExtractor):
    pass


if __name__ == '__main__':
    from skimage.metrics import structural_similarity

    for i in range(1, 50):
        print(f'Filename: {i}.gif')
        image = read_image(f'res/dataset-50/{i}.gif')
        np.random.seed(2115)
        data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)
        embedder = ImprovedBPUnidirectionEmbedder(image, data)

        embedded_image, iterations, pure_embedded_data = embedder.embed(1000)
        print(f'iterations: {iterations}')
        print(f'rate: {pure_embedded_data / image.size}')
        print(f'mean difference: {abs(embedded_image.mean() - image.mean())}')
        print(f'STD: {embedded_image.std()}')
        print(f'SSIM: {structural_similarity(image, embedded_image)}')

        extractor = BPUnidirectionExtractor()
        print(f'Correct extraction? {np.sum(np.abs(extractor.extract(embedded_image)[0] - image))}')
        print()
