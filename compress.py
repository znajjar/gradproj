import bz2
import lzma
import zlib


class Lzma:
    @staticmethod
    def compress(data_bytes):
        return lzma.compress(data_bytes)

    @staticmethod
    def decompress(data_bytes):
        return lzma.decompress(data_bytes)

    def __str__(self):
        return 'LZMA'


class Zlib:
    @staticmethod
    def compress(data_bytes):
        return zlib.compress(data_bytes, 9)

    @staticmethod
    def decompress(data_bytes):
        return zlib.decompress(data_bytes)

    def __str__(self):
        return 'zlib'


class Bz2:
    def __init__(self):
        self.compress = bz2.compress
        self.decompress = bz2.decompress

    def __str__(self):
        return 'bz2'


class NoCompress:
    @staticmethod
    def compress(data_bytes):
        return data_bytes

    @staticmethod
    def decompress(data_bytes):
        return data_bytes

    def __str__(self):
        return 'no compression'
