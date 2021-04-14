from collections import namedtuple

from twodirectional import bp_scaling
from twodirectional import original
from twodirectional import scaling

RdhAlgorithm = namedtuple('RdhAlgorithm', ('embedder', 'extractor', 'label'))

original_algorithm = RdhAlgorithm(original.OriginalEmbedder, original.OriginalExtractor, 'original')
scaling_algorithm = RdhAlgorithm(scaling.ScalingEmbedder, scaling.ScalingExtractor, 'scaling')
bp_scaling_algorithm = RdhAlgorithm(bp_scaling.BPScalingEmbedder, scaling.ScalingExtractor, 'bp_scaling')
