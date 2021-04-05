import PIL.Image as Image

from unidirection.configurations import *


def imsave(filename, img):
    Image.fromarray(img).save(filename)


def get_hist(img):
    return np.bincount(np.array(img).flatten(), minlength=L)


def get_peaks_from_hist(hist):
    P_H = hist.argmax()
    if P_H < 2:
        P_L = get_minimum_closest_right(hist, P_H)
    elif P_H > 253:
        P_L = get_minimum_closest_left(hist, P_H)
    else:
        P_L = get_minimum_closest(hist, P_H)

    return P_L, P_H


def get_minimum_closest_right(hist, P_H):
    hist_right = (np.roll(hist, 1) + hist)[P_H + 2:]
    candidates = np.flatnonzero(hist_right == hist_right.min()) + P_H + 2
    candidates = candidates[np.flatnonzero(hist[candidates - 1] == hist[candidates - 1].min())]
    return candidates[np.abs(candidates - P_H).argmin()]


def get_minimum_closest_left(hist, P_H):
    hist_left = (np.roll(hist, -1) + hist)[:P_H - 2 + 1]
    candidates = np.flatnonzero(hist_left == hist_left.min())
    candidates = candidates[np.flatnonzero(hist[candidates + 1] == hist[candidates + 1].min())]
    return candidates[np.abs(candidates - P_H).argmin()]

def get_minimum_closest(hist, P_H):
    closest_right = get_minimum_closest_right(hist, P_H)
    closest_left = get_minimum_closest_left(hist, P_H)
    min_right_value = hist[closest_right] + hist[closest_right - 1]
    min_left_value = hist[closest_left] + hist[closest_left + 1]
    if min_right_value < min_left_value:
        return closest_right
    elif min_right_value > min_left_value:
        return closest_left
    else:
        if abs(closest_right - P_H) < abs(closest_left - P_H):
            return closest_right
        else:
            return closest_left


def get_location_map(P_L):
    location_map = img[np.logical_or(img == P_L - d, img == P_L)]
    return location_map == P_L - d


def fast_shift_histogram(P_L, P_H):
    global hidden_data_index
    shift_in_between(P_L, P_H)

    embedding_pixels = img == P_H
    embedded_data = np.zeros(np.sum(embedding_pixels))
    embedded_data[:overhead_data.size] = overhead_data
    hidden_data_new_index = min(hidden_data_index + capacity - overhead_data.size, hidden_data.size)
    embedded_data[overhead_data.size: overhead_data.size + hidden_data_new_index - hidden_data_index] = \
        hidden_data[hidden_data_index:hidden_data_new_index]
    hidden_data_index = hidden_data_new_index
    img[embedding_pixels] = img[embedding_pixels] + d * embedded_data


def shift_in_between(P_L, P_H):
    in_between = np.logical_and(img > min((P_H, P_L)), img < max((P_H, P_L)))
    img[in_between] = img[in_between] + d


def embed_in_LSB(P_L, P_H):
    LSBs = np.concatenate([integer_to_binary(P_L, PEAK_BITS), integer_to_binary(P_H, PEAK_BITS)])
    for i in range(0, LSB_BITS):
        header_pixels[i] = set_lsb(header_pixels[i], LSBs[i])


def get_overhead(P_L, P_H, location_map, footer):
    compressed_map = bytes_to_bits(compress(location_map))
    flag = location_map.size < compressed_map.size

    if flag:
        return np.concatenate([
            integer_to_binary(P_L, PEAK_BITS),
            integer_to_binary(P_H, PEAK_BITS),
            integer_to_binary(flag, FLAG_BIT),
            integer_to_binary(compressed_map.size, COMPRESSED_DATA_LENGTH_BITS),
            compressed_map,
            footer], axis=None).astype(bool)
    else:
        return np.concatenate([
            integer_to_binary(P_L, PEAK_BITS),
            integer_to_binary(P_H, PEAK_BITS),
            integer_to_binary(flag, FLAG_BIT),
            location_map,
            footer], axis=None).astype(bool)


def process():
    global img, old_P_L, old_P_H, overhead_data, pure_embedded_data, d, iteration, capacity
    while True:
        hist = get_hist(img)
        P_L, P_H = get_peaks_from_hist(hist)  # what if there are multiple options?
        capacity = np.sum(img == P_H)

        if P_L < P_H:
            d = -1
        else:
            d = 1

        location_map = get_location_map(P_L)
        overhead_data = get_overhead(old_P_L, old_P_H, location_map, overhead_data)
        if overhead_data.size > capacity or iteration == iterations_limit:
            break

        pure_embedded_data += capacity - overhead_data.size

        fast_shift_histogram(P_L, P_H)
        overhead_data = []
        old_P_L = P_L
        old_P_H = P_H
        iteration += 1

    embed_in_LSB(old_P_L, old_P_H)

    img = np.append(img, header_pixels).reshape((M, N))


def main():
    global overhead_data, img, header_pixels, M, N, hidden_data_index, d, old_P_L, old_P_H, \
        iteration, pure_embedded_data
    hidden_data_index = 0
    d = 1
    old_P_L = 0
    old_P_H = 0
    iteration = 0
    pure_embedded_data = 0
    M, N = img.shape
    img = img.flatten()
    header_pixels = img[-16:]
    img = img[:-16]
    overhead_data = np.array(get_lsb(header_pixels), dtype=bool)
    process()


def embed(cover_image, data, limit=None):
    global img, hidden_data, iterations_limit
    if limit:
        iterations_limit = limit
    img = cover_image
    hidden_data = bytes_to_bits(data)
    main()
    return img, hidden_data[hidden_data_index:]


hidden_data_index = None
d = None
old_P_L = None
old_P_H = None
iteration = None
pure_embedded_data = None
header_pixels = None
img = None
M = None
N = None
overhead_data = None
capacity = None
iterations_limit = 1000000

if __name__ == '__main__':
    # hidden_data = bytes_to_bits(open('res/data.txt', 'rb').read())
    # hidden_data = np.array([])
    img = np.uint8(Image.open(IMAGE_PATH))
    np.random.seed(2115)
    hidden_data = np.random.randint(0, 2, size=2000 * 2000) > 0
    main()
    print("Took", iteration, "iterations")
    print("Pure embedding data:", pure_embedded_data, "bits")
    print("Embedding Capacity:", pure_embedded_data / M / N)
    print("STD: ", np.std(img))
    imsave("out/embedded_uni.png", img)
