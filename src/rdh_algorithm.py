from typing import NamedTuple, Any

from src.bidirectional import bp_scaling, scaling, original
from src.unidirection import bp_uni
from src.unidirection import bp_uni_improved
from src.unidirection import uni_original


# RdhAlgorithm = namedtuple('RdhAlgorithm', ('embedder', 'extractor', 'label'))
class RdhAlgorithm(NamedTuple):
    embedder: Any
    extractor: Any
    label: str


original_algorithm = RdhAlgorithm(original.OriginalEmbedder, original.OriginalExtractor, 'original')
scaling_algorithm = RdhAlgorithm(scaling.ScalingEmbedder, scaling.ScalingExtractor, 'scaling')
bp_scaling_algorithm = RdhAlgorithm(bp_scaling.BPScalingEmbedder, bp_scaling.BPScalingExtractor, 'bp_scaling')
uni_algorithm = RdhAlgorithm(uni_original.UnidirectionEmbedder, uni_original.UnidirectionExtractor, 'unidirection')
bp_uni_algorithm = RdhAlgorithm(bp_uni.BPUnidirectionEmbedder, bp_uni.BPUnidirectionExtractor, 'bp_unidirection')

bp_uni_algorithm_improved = RdhAlgorithm(bp_uni_improved.ImprovedBPUnidirectionEmbedder,
                                         bp_uni_improved.ImprovedBPUnidirectionExtractor,
                                         'bp_unidirection_improved')

vb_scaling_algorithm = RdhAlgorithm(scaling.VariableBitsScalingEmbedder,
                                    scaling.VariableBitsScalingExtractor,
                                    'vb_scaling')

bp_vb_scaling_algorithm = RdhAlgorithm(bp_scaling.BPVariableBitsScalingEmbedder,
                                       bp_scaling.BPVariableBitsScalingExtractor,
                                       'bp_vb_scaling')
