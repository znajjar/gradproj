from unidirection import BPUnidirectionEmbedder, UnidirectionExtractor
from unidirection.configurations import *
from util import *


class ImprovedBPUnidirectionEmbedder(BPUnidirectionEmbedder):

    def _get_peaks(self):
        self._hist = self._get_hist()
        current_brightness = np.mean(self._body_pixels)
        if self._original_brightness - current_brightness > BRIGHTNESS_THRESHOLD:
            return self._get_best_overall_right()
        elif self._original_brightness - current_brightness < -BRIGHTNESS_THRESHOLD:
            return self._get_best_overall_left()
        else:
            return self._get_best_overall()

    def _get_best_overall_right(self):
        P_L_zero, P_H_zero = self._get_peaks_zero_right()
        P_L_right, P_H_right = self._get_peaks_difference_right()
        if P_L_zero != P_H_zero and self._is_zero_better(P_H_zero, P_L_right, P_H_right):
            return P_L_zero, P_H_zero
        else:
            return P_L_right, P_H_right

    def _is_zero_better(self, P_H_zero, P_L, P_H):
        return self._hist[P_H_zero] - COMPRESSED_DATA_LENGTH_BITS > \
               self._get_embedding_capacity(P_L, P_H, get_shift_direction(P_L, P_H))

    def _get_best_overall_left(self):
        P_L_zero, P_H_zero = self._get_peaks_zero_left()
        P_L_left, P_H_left = self._get_peaks_difference_left()
        if P_L_zero != P_H_zero and self._is_zero_better(P_H_zero, P_L_left, P_H_left):
            return P_L_zero, P_H_zero
        else:
            return P_L_left, P_H_left

    def _get_best_overall(self):
        P_L_zero, P_H_zero = self._get_peaks_zero()
        P_L_best, P_H_best = self._get_best_difference_peaks()
        if P_L_zero != P_H_zero and self._is_zero_better(P_H_zero, P_L_best, P_H_best):
            return P_L_zero, P_H_zero
        else:
            return P_L_best, P_H_best

    def _get_peaks_difference_right(self):
        closest_right = self._get_closest_right()
        max_diff_right = self._get_embedding_capacity(closest_right, np.arange(0, MAX_PIXEL_VALUE + 1), RIGHT_DIRECTION)
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

    def _get_peaks_zero_right(self):
        zeros = np.nonzero(self._hist == 0)[0]
        if len(zeros) == 0 or zeros[-1] == 0:
            return 0, 0
        else:
            P_H = self._hist[:zeros[-1]].argmax()
            right_of_P_H = zeros[zeros > P_H]
            P_L = right_of_P_H[(right_of_P_H - P_H).argmin()]
            return P_L, P_H

    def _get_peaks_difference_left(self):
        closest_left = self._get_closest_left()
        max_diff_left = self._get_embedding_capacity(closest_left, np.arange(0, MAX_PIXEL_VALUE + 1), LEFT_DIRECTION)
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

    def _get_peaks_zero_left(self):
        zeros = np.nonzero(self._hist == 0)[0]
        if len(zeros) == 0 or zeros[0] == MAX_PIXEL_VALUE:
            return 0, 0
        else:
            P_H = self._hist[zeros[0] + 1:].argmax() + zeros[0] + 1
            left_of_P_H = zeros[zeros < P_H]
            P_L = left_of_P_H[(P_H - left_of_P_H).argmin()]
            return P_L, P_H

    def _get_best_difference_peaks(self):
        P_L_left, P_H_left = self._get_peaks_difference_left()
        P_L_right, P_H_right = self._get_peaks_difference_right()

        if self._get_embedding_capacity(P_L_left, P_H_left, LEFT_DIRECTION) > \
                self._get_embedding_capacity(P_L_right, P_H_right, RIGHT_DIRECTION):
            return P_L_left, P_H_left
        else:
            return P_L_right, P_H_right

    def _get_peaks_zero(self):
        zeros = np.nonzero(self._hist == 0)[0]
        if len(zeros) == 0:
            return 0, 0
        else:
            P_H = self._hist.argmax()
            P_L = zeros[(np.abs(zeros - P_H)).argmin()]
            return P_L, P_H

    def _get_embedding_capacity(self, P_L, P_H, direction):
        location_map_size = np.asarray(self._hist[P_L] + self._hist[P_L - direction], dtype=float)
        percentage = np.minimum(self._hist[P_L], self._hist[P_L - direction]) / location_map_size
        compressed_map_size = estimite_compressed_map_size(location_map_size, percentage)

        return self._hist[P_H] - np.minimum(location_map_size, compressed_map_size + COMPRESSED_DATA_LENGTH_BITS)

    def _get_buffer_data(self, P_L, P_H):
        if self._hist[P_L] == 0 and self._is_zero_better_than_difference(P_L, P_H):
            overhead_data = self._get_overhead_zero_peak()
            return overhead_data, self._get_capacity(P_H) - len(overhead_data)
        else:
            return super()._get_buffer_data(P_L, P_H)

    def _is_zero_better_than_difference(self, P_L, P_H):
        direction = get_shift_direction(P_L, P_H)
        return P_L - direction < 0 or P_L - direction > MAX_PIXEL_VALUE or self._hist[
            P_L - direction] > COMPRESSED_DATA_LENGTH_BITS

    def _get_overhead_zero_peak(self):
        return np.concatenate([
            integer_to_binary(self._old_P_L, PEAK_BITS),
            integer_to_binary(self._old_P_H, PEAK_BITS),
            integer_to_binary(1, FLAG_BIT),
            integer_to_binary(0, COMPRESSED_DATA_LENGTH_BITS)], axis=None).astype(bool)


class BPUnidirectionExtractor(UnidirectionExtractor):

    def _get_location_map(self, P_L):
        is_map_compressed = self._buffer.next(FLAG_BIT)[0]
        if is_map_compressed:
            return self._get_compressed_map()
        else:
            return self._buffer.next(np.sum(self._body_pixels == P_L))

    def _get_compressed_map(self):
        map_size = binary_to_integer(self._buffer.next(COMPRESSED_DATA_LENGTH_BITS))
        if map_size == 0:
            return np.ndarray(shape=(0, 0), dtype=bool)
        else:
            return bytes_to_bits(self._decompress(bits_to_bytes(self._buffer.next(map_size))))


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
