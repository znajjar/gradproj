from collections import namedtuple

from twodirectional import original
from twodirectional import scaling

RdhAlgorithm = namedtuple('RdhAlgorithm', ('embedder', 'extractor', 'label'))

original_algorithm = RdhAlgorithm(original.OriginalEmbedder, original.OriginalExtractor, 'original')
scaling_algorithm = RdhAlgorithm(scaling.ScalingEmbedder, scaling.ScalingExtractor, 'scaling')
