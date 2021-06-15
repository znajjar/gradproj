from unidirection.configurations import *
from unidirection.uni_original import UnidirectionEmbedder, UnidirectionExtractor
from util import *


class BPUnidirectionEmbedder(UnidirectionEmbedder):
    def __init__(self, cover_image: np.ndarray, hidden_data: Iterable, compression: CompressionAlgorithm = deflate):
        super().__init__(cover_image, hidden_data, compression)
        self._original_brightness = np.mean(cover_image)

    def _get_peaks(self):
        self._hist = self._get_hist()
        current_brightness = np.mean(self._body_pixels)
        if self._original_brightness - current_brightness > BRIGHTNESS_THRESHOLD:
            P_H = self._hist[:MAX_PIXEL_VALUE - 1].argmax()
        elif self._original_brightness - current_brightness < -BRIGHTNESS_THRESHOLD:
            P_H = self._hist[2:].argmax() + 2
        else:
            P_H = self._hist.argmax()

        if self._original_brightness - current_brightness > BRIGHTNESS_THRESHOLD or P_H < 2:
            P_L = get_minimum_closest_right(self._hist, P_H)
        elif self._original_brightness - current_brightness < -BRIGHTNESS_THRESHOLD or P_H > 253:
            P_L = get_minimum_closest_left(self._hist, P_H)
        else:
            P_L = get_minimum_closest(self._hist, P_H)

        return P_L, P_H


class BPUnidirectionExtractor(UnidirectionExtractor):
    pass


if __name__ == '__main__':
    from skimage.metrics import structural_similarity

    image = read_image('res/lena_gray_512.png')
    np.random.seed(2115)
    data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)
    embedder = BPUnidirectionEmbedder(image, data)

    embedded_image, iterations, hidden_data_size = embedder.embed()

    print(f'Mean difference: {abs(np.mean(embedded_image) - np.mean(image))}')
    print(f'SSIM: {structural_similarity(image, embedded_image)}')
    print(f'Old STD: {np.std(image)}')
    print(f'New STD: {np.std(embedded_image)}')
    print(f'Rate: {hidden_data_size / image.size}')

    show_hist(embedded_image)

    Image.fromarray(embedded_image).save('out/bp_vb_scaling.png')
