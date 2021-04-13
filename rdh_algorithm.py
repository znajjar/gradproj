from collections import namedtuple

from twodirectional import original

RdhAlgorithm = namedtuple('RdhAlgorithm', ('embedder', 'extractor', 'label'))

original_algorithm = RdhAlgorithm(original.OriginalEmbedder, original.OriginalExtractor, 'original')
