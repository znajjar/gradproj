import zlib

import numpy as np

L = 256
COMPRESSED_DATA_LENGTH_BITS = 16
LSB_BITS = 16
PEAK_BITS = 8
FLAG_BIT = 1
COMPRESSION_LEVEL = 9
IMAGE_PATH = "img/1.gif"


def integer_to_binary(number: int, bits=8):
    return [x == '1' for x in format(number, f'0{bits}b')]

def binary_to_integer(binary):
    return int.from_bytes(bits_to_bytes(binary), byteorder='big')

def get_lsb(values):
    lsb = []
    for pixel in values:
        lsb.append(integer_to_binary(pixel)[-1])

    return lsb

def set_lsb(value, lsb):
    if lsb:
        value |= 1
    else:
        value &= ~1

    return value

def bytes_to_bits(buffer):
    return np.where(np.unpackbits(np.frombuffer(buffer, np.uint8)) == 1, True, False)

def bits_to_bytes(buffer):
    return np.packbits(buffer).tobytes()

def compress(data_bits):
    data_bytes = bits_to_bytes(data_bits)
    return zlib.compress(data_bytes, level=COMPRESSION_LEVEL)

def decompress(compressed_data_bits):
    data_bytes = bits_to_bytes(compressed_data_bits)
    return zlib.decompress(data_bytes)
