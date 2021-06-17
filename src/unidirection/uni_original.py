from util import *
from unidirection.configurations import *


class UnidirectionEmbedder:
    def __init__(self, cover_image: np.ndarray, hidden_data: Iterable, compression: CompressionAlgorithm = deflate):
        self._cover_image = cover_image
        self._hidden_data = bytes_to_bits(hidden_data)
        self._compress = compression.compress

        self._header_pixels = None
        self._body_pixels = None
        self._buffer = None

        self._hist = None
        self._old_P_L = None
        self._old_P_H = None
        self._index = None

    def embed(self, iterations):
        self._initialize()
        return self._process(iterations)

    def _process(self, iterations=1):
        pure_embedded_data = 0

        P_L, P_H = self._get_peaks()
        buffer_data, extra_space = self._get_buffer_data(P_L, P_H)
        if not self._index:
            extra_space -= HEADER_SIZE

        while extra_space >= 0 and self._index < iterations:
            self._fill_buffer(buffer_data)
            pure_embedded_data += extra_space
            self._shift_histogram(P_L, P_H)

            self._old_P_L = P_L
            self._old_P_H = P_H
            self._index += 1
            P_L, P_H = self._get_peaks()
            buffer_data, extra_space = self._get_buffer_data(P_L, P_H)

        self._embed_in_LSB()

        embedded_image = assemble_image(self._header_pixels, self._body_pixels, self._cover_image.shape)

        return embedded_image, self._index, pure_embedded_data

    def _initialize(self):
        self._header_pixels, self._body_pixels = get_header_and_body(self._cover_image, HEADER_SIZE)
        self._buffer = BoolDataBuffer(self._get_header_LSBs(), self._hidden_data)

        self._old_P_L = 0
        self._old_P_H = 0
        self._index = 0

    def _get_header_LSBs(self):
        return np.array(get_lsb(self._header_pixels), dtype=bool)

    def _get_buffer_data(self, P_L, P_H):
        location_map = self._get_location_map(P_L, P_H)
        overhead_data = self._get_overhead(self._old_P_L, self._old_P_H, location_map)
        return overhead_data, self._hist[P_H] - len(overhead_data)

    def _fill_buffer(self, buffer_data):
        self._buffer.add(buffer_data)

    def _get_peaks(self):
        self._hist = self._get_hist()
        P_H = self._hist.argmax()
        if P_H < 2:
            P_L = get_minimum_closest_right(self._hist, P_H)
        elif P_H > 253:
            P_L = get_minimum_closest_left(self._hist, P_H)
        else:
            P_L = get_minimum_closest(self._hist, P_H)

        return P_L, P_H

    def _get_location_map(self, P_L, P_H) -> np.ndarray:
        d = get_shift_direction(P_L, P_H)
        location_map = self._body_pixels[np.logical_or(self._body_pixels == P_L - d, self._body_pixels == P_L)]
        return np.equal(location_map, P_L - d)

    def _get_hist(self):
        return np.bincount(np.array(self._body_pixels).flatten(), minlength=MAX_PIXEL_VALUE + 1)

    def _get_overhead(self, P_L, P_H, location_map: np.ndarray):
        compressed_map = bytes_to_bits(self._compress(bits_to_bytes(location_map)))
        flag = len(location_map) > len(compressed_map) + COMPRESSED_DATA_LENGTH_BITS

        if flag:
            return np.concatenate([
                integer_to_binary(P_L, PEAK_BITS),
                integer_to_binary(P_H, PEAK_BITS),
                integer_to_binary(flag, FLAG_BIT),
                integer_to_binary(compressed_map.size//BITS_PER_BYTE, COMPRESSED_DATA_LENGTH_BITS),
                compressed_map], axis=None).astype(bool)
        else:
            return np.concatenate([
                integer_to_binary(P_L, PEAK_BITS),
                integer_to_binary(P_H, PEAK_BITS),
                integer_to_binary(flag, FLAG_BIT),
                location_map], axis=None).astype(bool)

    def _shift_histogram(self, P_L, P_H):
        self._shift_in_between(P_L, P_H)

        embedding_pixels = self._body_pixels == P_H
        embedded_data = self._buffer.next(self._hist[P_H])
        d = get_shift_direction(P_L, P_H)
        self._body_pixels[embedding_pixels] = self._body_pixels[embedding_pixels] + d * embedded_data

    def _shift_in_between(self, P_L, P_H):
        in_between = np.logical_and(self._body_pixels > min((P_L, P_H)), self._body_pixels < max((P_L, P_H)))
        self._body_pixels[in_between] = self._body_pixels[in_between] + get_shift_direction(P_L, P_H)

    def _embed_in_LSB(self):
        LSBs = np.concatenate([integer_to_binary(self._old_P_L, PEAK_BITS),
                               integer_to_binary(self._old_P_H, PEAK_BITS)])
        for i in range(0, HEADER_SIZE):
            self._header_pixels[i] = set_lsb(self._header_pixels[i], LSBs[i])

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if not self._index:
            self._initialize()

        try:
            prev_iterations = self._index
            embedded_image, iterations, pure_embedded_data = self._process(prev_iterations + 1)
            if iterations == prev_iterations:
                raise StopIteration
            return embedded_image, self._index, pure_embedded_data
        except ValueError:
            self.index = 0
            raise StopIteration


class UnidirectionExtractor:
    def __init__(self, compression=deflate):
        self._decompress = compression.decompress

        self._header_pixels = None
        self._body_pixels = None
        self._buffer = BoolDataBuffer()

        self._direction = None

    def extract(self, embedded_image):
        self._header_pixels, self._body_pixels = get_header_and_body(embedded_image, HEADER_SIZE)
        P_L, P_H = get_peaks_from_header(self._header_pixels, PEAK_BITS)
        iterations = 0
        hidden_data = []

        while P_L != 0 or P_H != 0:
            self._direction = get_shift_direction(P_L, P_H)
            self._fill_payload(P_H)
            new_P_L, new_P_H = self._get_next_peaks()
            self._shift_in_between(P_L, P_H)
            self._fix_P_L_bin(P_L)

            if new_P_L == 0 and new_P_H == 0:
                self._fix_LSB(self._buffer.next(HEADER_SIZE))

            hidden_data.extend(self._buffer.next(-1))
            P_L = new_P_L
            P_H = new_P_H
            iterations += 1

        cover_image = assemble_image(self._header_pixels, self._body_pixels, embedded_image.shape)
        hidden_data.reverse()

        return cover_image, iterations, bits_to_bytes(hidden_data)

    def _fill_payload(self, P_H):
        embedded_data = np.logical_or(self._body_pixels == P_H, self._body_pixels == P_H + self._direction)
        self._buffer.add(self._body_pixels[embedded_data] != P_H)

    def _get_next_peaks(self):
        return binary_to_integer(self._buffer.next(PEAK_BITS)), binary_to_integer(self._buffer.next(PEAK_BITS))

    def _shift_in_between(self, P_L, P_H):
        in_between = np.logical_and(self._body_pixels > min((P_H, P_L)), self._body_pixels < max((P_H, P_L)))
        self._body_pixels[in_between] = self._body_pixels[in_between] - self._direction

    def _fix_P_L_bin(self, P_L):
        location_map = self._get_location_map(P_L)
        combined_bin = self._body_pixels == P_L
        if location_map.size == 0:
            self._body_pixels[combined_bin] = self._body_pixels[combined_bin] - self._direction
        else:
            self._body_pixels[combined_bin] = self._body_pixels[combined_bin] - \
                                              self._direction * location_map[:np.sum(combined_bin)]

    def _get_location_map(self, P_L):
        is_map_compressed = self._buffer.next(FLAG_BIT)[0]
        if is_map_compressed:
            map_size = binary_to_integer(self._buffer.next(COMPRESSED_DATA_LENGTH_BITS)) * BITS_PER_BYTE
            return bytes_to_bits(self._decompress(bits_to_bytes(self._buffer.next(map_size))))
        else:
            return self._buffer.next(np.sum(self._body_pixels == P_L))

    def _fix_LSB(self, LSBs):
        for i in range(0, HEADER_SIZE):
            self._header_pixels[i] = set_lsb(self._header_pixels[i], LSBs[i])
