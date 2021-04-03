import numpy as np
import PIL.Image as Image

from configurations import *
from data_buffer import DataBuffer


img = np.uint8(Image.open("out/embedded.png"))
M, N = img.shape
img = img.flatten()
header_pixel = img[-16:]
img = img[:-16]
hidden_data = []
d = 1

def imsave(filename, img):
    Image.fromarray(img).save(filename)

def get_payload(P_H):
    embedded_data = np.logical_or(img == P_H, img == P_H + d)
    return DataBuffer(bool, img[embedded_data] != P_H)

def shift_in_between(P_L, P_H):
    in_between = np.logical_and(img > min((P_H, P_L)), img < max((P_H, P_L)))
    img[in_between] = img[in_between] - d

def fix_P_L_bin(P_L, location_map):
    combined_bin = img == P_L
    if location_map.size == 0:
        img[combined_bin] = img[combined_bin] - d
    else:
        img[combined_bin] = img[combined_bin] - d*location_map[:np.sum(combined_bin)]

def fix_LSB(LSBs):
    for i in range(0, LSB_BITS):
        header_pixel[i] = set_lsb(header_pixel[i], LSBs[i])


LSB = get_lsb(header_pixel)
P_L = binary_to_integer(LSB[0:PEAK_BITS])
P_H = binary_to_integer(LSB[PEAK_BITS:2*PEAK_BITS])
hidden_data = np.array([], dtype=np.bool)

iteration = 0

while P_L != 0 or P_H != 0:
    if P_L < P_H:
        d = -1
    else:
        d = 1

    payload = get_payload(P_H)
    new_P_L = binary_to_integer(payload.next(PEAK_BITS))
    new_P_H = binary_to_integer(payload.next(PEAK_BITS))
    is_map_compressed = payload.next(FLAG_BIT)[0]
    if is_map_compressed:
        map_size = binary_to_integer(payload.next(COMPRESSED_DATA_LENGTH_BITS))
        location_map = bytes_to_bits(decompress(payload.next(map_size)))
    else:
        location_map = payload.next(np.sum(img == P_L))

    shift_in_between(P_L, P_H)
    fix_P_L_bin(P_L, location_map)

    if new_P_L == 0 and new_P_H == 0: 
        fix_LSB(payload.next(LSB_BITS))

    hidden_data = np.append(payload.next(-1), hidden_data)
    P_L = new_P_L
    P_H = new_P_H
    iteration += 1

img = np.append(img, header_pixel).reshape((M, N))
imsave("out/recovered.png", img)
open("hidden/hiddenOutput", 'wb').write(bits_to_bytes(hidden_data))
print("Hidden data size:", hidden_data.size)

orignal_img = np.uint8(Image.open(IMAGE_PATH))
print("Sum of absolute difference:", np.sum(np.abs(orignal_img - img)))
