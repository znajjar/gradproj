from collections import namedtuple

from twodirectional import bp_scaling
from twodirectional import original
from twodirectional import scaling
from unidirection import uni_original
from unidirection import bp_uni

RdhAlgorithm = namedtuple('RdhAlgorithm', ('embedder', 'extractor', 'label'))

original_algorithm = RdhAlgorithm(original.OriginalEmbedder, original.OriginalExtractor, 'original')
scaling_algorithm = RdhAlgorithm(scaling.ScalingEmbedder, scaling.ScalingExtractor, 'scaling')
bp_scaling_algorithm = RdhAlgorithm(bp_scaling.BPScalingEmbedder, bp_scaling.BPScalingExtractor, 'bp_scaling')
uni_algorithm = RdhAlgorithm(uni_original.UnidirectionEmbedder, uni_original.UnidirectionExtractor, 'unidirection')
bp_uni_algorithm = RdhAlgorithm(bp_uni.BPUnidirectionEmbedder, bp_uni.BPUnidirectionExtractor, 'bp_unidirection')
