from unidirection import ImprovedBPUnidirectionEmbedder, ImprovedBPUnidirectionExtractor
from unidirection.configurations import *
from util import *


class BPZeroUnidirectionEmbedder(ImprovedBPUnidirectionEmbedder):

    def _shift_in_between(self, P_L, P_H):
        if self._zero_peak:
            in_between = np.logical_and(self._body_pixels > min((P_L, P_H)), self._body_pixels < max((P_L, P_H)))
            self._body_pixels[in_between] = self._body_pixels[in_between] + get_shift_direction(P_L, P_H)
        else:
            super()._shift_in_between(P_L, P_H)

    def _get_buffer_data(self, P_L, P_H):
        if self._zero_peak:
            overhead_data = self._get_overhead_zero_peak()
            return overhead_data, self._hist[P_H] - len(overhead_data)
        else:
            return super()._get_buffer_data(P_L, P_H)

    def _get_peaks(self):
        self._hist = self._get_hist()
        self._minimum_closest_P_L = self._get_minimum_closest_by_N(2 ** PLACEMENT_BITS)
        current_brightness = np.mean(self._body_pixels)
        if self._original_brightness - current_brightness > BRIGHTNESS_THRESHOLD:
            self._P_L, self._P_H = self._get_best_overall_right()
        elif self._original_brightness - current_brightness < -BRIGHTNESS_THRESHOLD:
            self._P_L, self._P_H = self._get_best_overall_left()
        else:
            self._P_L, self._P_H = self._get_best_overall()

        return self._P_L, self._P_H

    def _get_best_overall_right(self):
        P_L_zero, P_H_zero = self._get_peaks_zero_right()
        P_L_right, P_H_right = self._get_peaks_difference_right()
        self._zero_peak = P_L_zero != P_H_zero and self._is_zero_better(P_H_zero, P_L_right, P_H_right)
        if self._zero_peak:
            return P_L_zero, P_H_zero
        else:
            return P_L_right, P_H_right

    def _get_peaks_zero_right(self):
        zeros = np.nonzero(self._hist == 0)[0]
        if len(zeros) == 0 or zeros[-1] == 0:
            return 0, 0
        else:
            P_H = self._hist[:zeros[-1]].argmax()
            right_of_P_H = zeros[zeros > P_H]
            P_L = right_of_P_H[(right_of_P_H - P_H).argmin()]
            return P_L, P_H

    def _is_zero_better(self, P_H_zero, P_L, P_H):
        return self._hist[P_H_zero] - COMPRESSED_DATA_LENGTH_BITS > \
               self._get_embedding_capacity(P_L, P_H) - SIGN_BIT - PLACEMENT_BITS

    def _get_best_overall_left(self):
        P_L_zero, P_H_zero = self._get_peaks_zero_left()
        P_L_left, P_H_left = self._get_peaks_difference_left()
        self._zero_peak = P_L_zero != P_H_zero and self._is_zero_better(P_H_zero, P_L_left, P_H_left)
        if self._zero_peak:
            return P_L_zero, P_H_zero
        else:
            return P_L_left, P_H_left

    def _get_peaks_zero_left(self):
        zeros = np.nonzero(self._hist == 0)[0]
        if len(zeros) == 0 or zeros[0] == MAX_PIXEL_VALUE:
            return 0, 0
        else:
            P_H = self._hist[zeros[0] + 1:].argmax() + zeros[0] + 1
            left_of_P_H = zeros[zeros < P_H]
            P_L = left_of_P_H[(P_H - left_of_P_H).argmin()]
            return P_L, P_H

    def _get_best_overall(self):
        P_L_zero, P_H_zero = self._get_peaks_zero()
        P_L_best, P_H_best = self._get_best_difference_peaks()
        self._zero_peak = P_L_zero != P_H_zero and self._is_zero_better(P_H_zero, P_L_best, P_H_best)
        if self._zero_peak:
            return P_L_zero, P_H_zero
        else:
            return P_L_best, P_H_best

    def _get_peaks_zero(self):
        zeros = np.nonzero(self._hist == 0)[0]
        if len(zeros) == 0:
            return 0, 0
        else:
            P_H = self._hist.argmax()
            P_L = zeros[(np.abs(zeros - P_H)).argmin()]
            return P_L, P_H

    def _get_overhead_zero_peak(self):
        return np.concatenate([
            integer_to_binary(self._old_P_L, PEAK_BITS),
            integer_to_binary(self._old_P_H, PEAK_BITS),
            integer_to_binary(1, FLAG_BIT),
            integer_to_binary(0, COMPRESSED_DATA_LENGTH_BITS)], axis=None).astype(bool)

    def _insert_offset_bits(self, overhead_data):
        offset_bits = self._get_peak_offset()
        is_compressed = overhead_data[2 * PEAK_BITS]
        insert_index = 2 * PEAK_BITS + FLAG_BIT + is_compressed * COMPRESSED_DATA_LENGTH_BITS
        return np.concatenate([overhead_data[:insert_index], offset_bits, overhead_data[insert_index:]])


class BPZeroUnidirectionExtractor(ImprovedBPUnidirectionExtractor):

    def _fix_P_L_bin(self, P_L):
        location_map = self._get_location_map(P_L)
        if location_map.size > 0:
            combined_bin = self._body_pixels == (P_L + self._offset)
            location_map = location_map[:np.sum(combined_bin)]
            self._body_pixels[combined_bin] = self._body_pixels[combined_bin] - self._offset * location_map

    def _get_location_map(self, P_L):
        is_map_compressed = self._buffer.next(FLAG_BIT)[0]
        if is_map_compressed:
            return self._get_compressed_map()
        else:
            self._offset = self._get_offset()
            return self._buffer.next(np.sum(self._body_pixels == P_L + self._offset))

    def _get_compressed_map(self):
        map_size = binary_to_integer(self._buffer.next(COMPRESSED_DATA_LENGTH_BITS)) * BITS_PER_BYTE
        if map_size == 0:
            return np.ndarray(shape=(0, 0), dtype=bool)
        else:
            self._offset = self._get_offset()
            return bytes_to_bits(self._decompress(bits_to_bytes(self._buffer.next(map_size))))


if __name__ == '__main__':
    from skimage.metrics import structural_similarity
    import cv2

    for i in range(1, 25):
        filename = f'kodim{str(i).zfill(2)}_org'
        print(f'Filename: {filename}.png')
        image = read_image(f'res/kodek_dataset/{filename}.png')
        np.random.seed(2115)
        data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)
        embedder = BPZeroUnidirectionEmbedder(image, data)

        embedded_image, iterations, pure_embedded_data = embedder.embed(1000)
        print(f'iterations: {iterations}')
        print(f'rate: {pure_embedded_data / image.size}')
        print(f'Abs change in mean: {abs(embedded_image.mean() - image.mean())}')
        print(f'Change in STD: {embedded_image.std() - image.std()}')
        print(f'SSIM: {structural_similarity(image, embedded_image)}')

        cv2.imwrite(f'out/bp_uni_improved_zero/{filename}.png', embedded_image)

        extractor = BPZeroUnidirectionExtractor()
        print(f'Correct extraction? {np.sum(np.abs(extractor.extract(embedded_image)[0] - image))}')
        print()
