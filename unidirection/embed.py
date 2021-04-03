import numpy as np
import PIL.Image as Image

from configurations import *

img = np.uint8(Image.open(IMAGE_PATH))
M, N = img.shape
img = img.flatten()
header_pixels = img[-16:]
img = img[:-16]

hidden_data_index = 0
d = 1

def imsave(filename, img):
    Image.fromarray(img).save(filename)

def get_hist(img):
    return np.bincount(np.array(img).flatten(), minlength=L)

def get_peaks_from_hist(hist):
    P_L = hist.argmin()
    P_H = hist.argmax()

    if P_H - P_L == 1:
        new_hist = np.concatenate([hist[0:P_L], [hist[P_H] + 1, hist[P_H] + 1, hist[P_H] + 1], hist[P_H + 2:]])
        P_L = new_hist.argmin()
    elif P_L - P_H == 1:
        new_hist = np.concatenate([hist[0:P_H], [hist[P_H] + 1, hist[P_H] + 1, hist[P_H] + 1], hist[P_L + 2:]])
        P_L = new_hist.argmin()
    elif P_L == P_H:
        new_hist = np.concatenate([hist[0:P_H], [hist[P_H] + 1, hist[P_H] + 1], hist[P_H + 2:]])
        P_L = new_hist.argmin()

    return P_L, P_H
    
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
    embedded_data[overhead_data.size : overhead_data.size + hidden_data_new_index - hidden_data_index] = hidden_data[hidden_data_index:hidden_data_new_index]
    hidden_data_index = hidden_data_new_index
    img[embedding_pixels] = img[embedding_pixels] + d*embedded_data

def shift_in_between(P_L, P_H):
    in_between = np.logical_and(img > min((P_H, P_L)), img < max((P_H, P_L)))
    img[in_between] = img[in_between] + d

def embed_in_LSB(P_L, P_H):
    LSBs = np.concatenate([integer_to_binary(P_L, PEAK_BITS), integer_to_binary(P_H, PEAK_BITS)])
    for i in range(0, LSB_BITS):
        header_pixels[i] = set_lsb(header_pixels[i], LSBs[i])

def get_overhead(P_L, P_H, flag, location_map, arr):
    overhead = np.concatenate([
    integer_to_binary(P_L, PEAK_BITS),
    integer_to_binary(P_H, PEAK_BITS),
    integer_to_binary(flag, FLAG_BIT)])

    if flag:
        overhead = np.concatenate([overhead, integer_to_binary(location_map.size, COMPRESSED_DATA_LENGTH_BITS)])

    return np.concatenate([overhead, location_map, arr], axis=None).astype(np.bool)


# hidden_data = bytes_to_bits(open('hidden/hiddenInput', 'rb').read())
# hidden_data = np.array([])
np.random.seed(2115)
hidden_data = np.random.randint(0, 2, size=2000*2000) > 0
overhead_data = np.array(get_lsb(header_pixels), dtype=np.bool)
old_P_L = 0
old_P_H = 0
iteration = 0
pure_embedded_data = 0

while True:
    hist = get_hist(img)
    P_L, P_H = get_peaks_from_hist(hist) # what if there are multiple options?
    capacity = np.sum(img == P_H)

    if P_L < P_H:
        d = -1
    else:
        d = 1

    location_map = get_location_map(P_L)
    compressed_map = bytes_to_bits(compress(location_map))
    if location_map.size < compressed_map.size:
        overhead_data = get_overhead(old_P_L, old_P_H, 0, location_map, overhead_data)
    else:
        overhead_data = get_overhead(old_P_L, old_P_H, 1, compressed_map, overhead_data)

    if overhead_data.size > capacity:
        break
    
    pure_embedded_data += capacity - overhead_data.size

    fast_shift_histogram(P_L, P_H)
    overhead_data = []
    old_P_L = P_L
    old_P_H = P_H
    iteration += 1

embed_in_LSB(old_P_L, old_P_H)

img = np.append(img, header_pixels).reshape((M, N))
print("Took", iteration, "iterations")
print("Pure embedding data:", pure_embedded_data, "bits")
print("Embedding Capacity:", pure_embedded_data/M/N)
print("STD: ", np.std(img))
imsave("out/embedded.png", img)
