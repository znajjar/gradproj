import lzma

import numpy as np

COMPRESSED_DATA_LENGTH_BITS = 16
MAX_PIXEL_VALUE = 255
COMPRESSION_LEVEL = 9
HEADER_SIZE = 17
EPS = 0.00000005


def integer_to_binary(number: int, bits=8):
    return [x == '1' for x in format(number, f'0{bits}b')]


def binary_to_string(binary):
    if binary:
        return '1'
    else:
        return '0'


def binary_to_integer(binary):
    return int.from_bytes(bits_to_bytes(binary), byteorder='big', signed=False)


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


def _compress(data_bits):
    data_bytes = bits_to_bytes(data_bits)

    compressor = lzma.LZMACompressor()
    ret = compressor.compress(data_bytes)
    ret += (compressor.flush())
    return ret

    # return zlib.compress(data_bytes, level=COMPRESSION_LEVEL)

    # return bz2.compress(data_bytes)


def decompress(compressed_data_bits):
    data_bytes = bits_to_bytes(compressed_data_bits)

    ret = lzma.LZMADecompressor().decompress(data_bytes)
    return ret

    # return zlib.decompress(data_bytes)

    # return bz2.decompress(data_bytes)


def get_header_and_body(image: np.ndarray, header_size=HEADER_SIZE) -> (np.ndarray, np.ndarray):
    image = image.ravel()
    return image[:header_size], image[header_size:]
