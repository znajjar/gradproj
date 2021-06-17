from unidirection import BPUnidirectionEmbedder, BPUnidirectionExtractor
from unidirection.configurations import *
from util import *


class ImprovedBPUnidirectionEmbedder(BPUnidirectionEmbedder):

    def __init__(self, cover_image: np.ndarray, hidden_data: Iterable, compression: CompressionAlgorithm = deflate):
        super().__init__(cover_image, hidden_data, compression)
        self._P_L = None
        self._P_H = None
        self._offset = None
        self._minimum_closest_P_L = None

    def _shift_in_between(self, P_L, P_H):
        self._move_bin(P_L)                                                # Not needed
        super()._shift_in_between(P_L, P_H)

    def _move_bin(self, P_L):
        self._body_pixels[self._body_pixels == P_L] = self._minimum_closest_P_L[P_L]
        self._hist[self._minimum_closest_P_L[P_L]] += self._hist[P_L]  # if self._minimum_closest_P_L[P_L] == P_H
        self._hist[P_L] = 0

    def _get_location_map(self, P_L, P_H):
        combined_bins = np.logical_or(self._body_pixels == self._minimum_closest_P_L[P_L], self._body_pixels == P_L)
        location_map = self._body_pixels[combined_bins]
        return np.equal(location_map, P_L)

    def _get_overhead(self, P_L, P_H, location_map):
        overhead_data = super()._get_overhead(P_L, P_H, location_map)
        return self._insert_offset_bits(overhead_data)

    def _insert_offset_bits(self, overhead_data):
        offset_bits = self._get_peak_offset()
        insert_index = 2*PEAK_BITS
        return np.concatenate([overhead_data[:insert_index], offset_bits, overhead_data[insert_index:]])

    def _get_peak_offset(self):
        sign = self._minimum_closest_P_L[self._P_L] < self._P_L
        offset = np.abs(int(self._P_L) - self._minimum_closest_P_L[self._P_L])
        return np.concatenate([[sign], integer_to_binary(offset - 1, PLACEMENT_BITS)], axis=None).astype(bool)

    def _get_peaks(self):
        self._hist = self._get_hist()
        self._minimum_closest_P_L = self._get_minimum_closest_by_N(2**PLACEMENT_BITS)
        current_brightness = np.mean(self._body_pixels)
        if self._original_brightness - current_brightness > BRIGHTNESS_THRESHOLD:
            self._P_L, self._P_H = self._get_peaks_difference_right()
        elif self._original_brightness - current_brightness < -BRIGHTNESS_THRESHOLD:
            self._P_L, self._P_H = self._get_peaks_difference_left()
        else:
            self._P_L, self._P_H = self._get_best_difference_peaks()

        return self._P_L, self._P_H

    def _get_minimum_closest_by_N(self, N):
        minimum_closest = np.zeros(MAX_PIXEL_VALUE + 1, dtype=np.uint8)
        for pixel_value in range(0, MAX_PIXEL_VALUE + 1):
            minimum_closest[pixel_value] = self._get_minimum_closest_pixel_value(pixel_value, N)

        return minimum_closest

    def _get_minimum_closest_pixel_value(self, pixel_value, N):
        old_value = self._hist[pixel_value]
        self._hist[pixel_value] = MAX_FREQUENCY
        candidates = self._hist[max(0, pixel_value - N):min(MAX_PIXEL_VALUE, pixel_value + N) + 1]
        candidates = np.flatnonzero(candidates == candidates.min()) + max(0, pixel_value - N)
        self._hist[pixel_value] = old_value
        return find_closest_candidate(candidates, pixel_value)

    def _get_peaks_difference_right(self):
        minimum_closest_right = self._get_minimum_closest_right()
        max_diff_right = self._get_embedding_capacity(minimum_closest_right, np.arange(0, MAX_PIXEL_VALUE + 1))
        best_P_H = max_diff_right[:MAX_PIXEL_VALUE].argmax()
        best_P_L = minimum_closest_right[best_P_H]

        return best_P_L, best_P_H

    def _get_minimum_closest_right(self):
        minimum_closest_right = np.zeros(MAX_PIXEL_VALUE + 1, dtype=np.uint8)
        min_so_far = MAX_PIXEL_VALUE
        min_sum = self._hist[MAX_PIXEL_VALUE] + self._hist[self._minimum_closest_P_L[MAX_PIXEL_VALUE]]
        for left_peak in range(MAX_PIXEL_VALUE - 1, -1, -1):
            minimum_closest_right[left_peak] = min_so_far
            new_sum = self._hist[left_peak] + self._hist[self._minimum_closest_P_L[left_peak]]
            if min_sum >= new_sum:
                min_sum = new_sum
                min_so_far = left_peak

        return minimum_closest_right

    def _get_peaks_difference_left(self):
        minimum_closest_left = self._get_minimum_closest_left()
        max_diff_left = self._get_embedding_capacity(minimum_closest_left, np.arange(0, MAX_PIXEL_VALUE + 1))
        best_P_H = max_diff_left[1:].argmax() + 1
        best_P_L = minimum_closest_left[best_P_H]

        return best_P_L, best_P_H

    def _get_minimum_closest_left(self):
        minimum_closest_left = np.zeros(MAX_PIXEL_VALUE + 1, dtype=np.uint8)
        min_so_far = 0
        min_sum = self._hist[0] + self._hist[self._minimum_closest_P_L[0]]
        for right_peak in range(1, MAX_PIXEL_VALUE + 1):
            minimum_closest_left[right_peak] = min_so_far
            new_sum = self._hist[right_peak] + self._hist[self._minimum_closest_P_L[right_peak]]
            if min_sum >= new_sum:
                min_sum = new_sum
                min_so_far = right_peak

        return minimum_closest_left

    def _get_best_difference_peaks(self):
        P_L_left, P_H_left = self._get_peaks_difference_left()
        P_L_right, P_H_right = self._get_peaks_difference_right()

        if self._get_embedding_capacity(P_L_left, P_H_left) > self._get_embedding_capacity(P_L_right, P_H_right):
            return P_L_left, P_H_left
        else:
            return P_L_right, P_H_right

    def _get_embedding_capacity(self, P_L, P_H):
        P_L_frequency = self._hist[P_L]
        placement_peak_frequency = self._hist[self._minimum_closest_P_L[P_L]]
        location_map_size = np.asarray(P_L_frequency + placement_peak_frequency, dtype=float)
        percentage = np.minimum(P_L_frequency, placement_peak_frequency) / location_map_size
        compressed_map_size = estimate_compressed_map_size(location_map_size, percentage)

        return self._hist[P_H] - np.minimum(location_map_size, compressed_map_size + COMPRESSED_DATA_LENGTH_BITS)


class ImprovedBPUnidirectionExtractor(BPUnidirectionExtractor):

    def _shift_in_between(self, P_L, P_H):
        if P_L < P_H:
            in_between = np.logical_and(self._body_pixels >= P_L, self._body_pixels < P_H)
        else:
            in_between = np.logical_and(self._body_pixels > P_H, self._body_pixels <= P_L)
        self._body_pixels[in_between] = self._body_pixels[in_between] - self._direction

    def _fix_P_L_bin(self, P_L):
        location_map = self._get_location_map(P_L)
        combined_bin = self._body_pixels == (P_L + self._offset)
        location_map = location_map[:np.sum(combined_bin)]
        self._body_pixels[combined_bin] = self._body_pixels[combined_bin] - self._offset * location_map

    def _get_location_map(self, P_L):
        self._offset = self._get_offset()
        return super()._get_location_map(P_L + self._offset)

    def _get_offset(self):
        sign = self._buffer.next(SIGN_BIT)
        offset = binary_to_integer(self._buffer.next(PLACEMENT_BITS)) + 1

        if sign:
            return -offset
        else:
            return offset


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

        extractor = ImprovedBPUnidirectionExtractor()
        print(f'Correct extraction? {np.sum(np.abs(extractor.extract(embedded_image)[0] - image))}')
        print()
