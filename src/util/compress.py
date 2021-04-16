import zlib as zl

import deflate as de

__all__ = [
    'deflate',
    'zlib',
    'no_compress',
    'CompressionAlgorithm'
]


class CompressionAlgorithm:
    def __init__(self, compress, decompress, label):
        self.compress = compress
        self.decompress = decompress
        self.label = label

    def __str__(self):
        return self.label


deflate = CompressionAlgorithm(lambda data_bytes: de.gzip_compress(data_bytes, 12),
                               de.gzip_decompress,
                               'deflate')

zlib = CompressionAlgorithm(lambda data_bytes: zl.compress(data_bytes, 9),
                            zl.decompress,
                            'zlib')

no_compress = CompressionAlgorithm(lambda data_bytes: data_bytes,
                                   lambda data_bytes: data_bytes,
                                   'no compression')
