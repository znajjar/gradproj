from bidirectional.configurations import BRIGHTNESS_THRESHOLD
from util import *
from bidirectional.scaling import ScalingEmbedder, ScalingExtractor


class BPScalingEmbedder(ScalingEmbedder):
    def __init__(self, cover_image: np.ndarray, hidden_data: Iterable, compression: CompressionAlgorithm = deflate):
        ScalingEmbedder.__init__(self, cover_image, hidden_data, compression)
        self._original_brightness = np.mean(cover_image)

    def embed(self, iterations):
        return super(BPScalingEmbedder, self).embed(iterations)

    def _get_peaks(self):
        hist = np.bincount(self._processed_pixels)
        current_brightness = np.mean(self._processed_pixels)
        cutoff_index = int(np.ceil(current_brightness))

        if self._original_brightness - current_brightness > BRIGHTNESS_THRESHOLD:
            return np.sort(hist[:cutoff_index].argsort()[-2:])
        elif self._original_brightness - current_brightness < -BRIGHTNESS_THRESHOLD:
            return np.sort(hist[cutoff_index:].argsort()[-2:]) + cutoff_index
        else:
            return np.sort(hist.argsort()[-2:])


class BPScalingExtractor(ScalingExtractor):
    pass


if __name__ == '__main__':
    image = read_image('res/dataset-50/36.gif')
    np.random.seed(2115)
    data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)
    embedder = BPScalingEmbedder(image, data)

    embedded_image, iterations, hidden_data_size = embedder.embed(64)

    print(hidden_data_size)
    print(np.mean(image))
    print(np.mean(embedded_image))
    print(np.std(image))
    print(np.std(embedded_image))
    show_hist(embedded_image)

    Image.fromarray(embedded_image).save('out/bp_scaling.png')
