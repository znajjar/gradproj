from typing import NamedTuple, Any

from src.bidirectional import bp_scaling, scaling, original
from src.unidirection import bp_uni
from src.unidirection import bp_uni_improved
from src.unidirection import bp_uni_improved_zero
from src.unidirection import uni_original


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

bp_uni_algorithm_improved_zero = RdhAlgorithm(bp_uni_improved_zero.BPZeroUnidirectionEmbedder,
                                              bp_uni_improved_zero.BPZeroUnidirectionExtractor,
                                              'bp_unidirection_improved_zero')

vb_scaling_algorithm = RdhAlgorithm(scaling.VariableBitsScalingEmbedder,
                                    scaling.VariableBitsScalingExtractor,
                                    'vb_scaling')

bp_vb_scaling_algorithm = RdhAlgorithm(bp_scaling.BPVariableBitsScalingEmbedder,
                                       bp_scaling.BPVariableBitsScalingExtractor,
                                       'bp_vb_scaling')

vo_scaling_algorithm = RdhAlgorithm(scaling.ValueOrderScalingEmbedder,
                                    scaling.ValueOrderedScalingExtractor,
                                    'vo_scaling')

bp_vo_scaling_algorithm = RdhAlgorithm(bp_scaling.BPValueOrderScalingEmbedder,
                                    bp_scaling.BPValueOrderedScalingExtractor,
                                    'bp_vo_scaling')

nb_original_algorithm = RdhAlgorithm(original.NeighboringBinsEmbedder,
                                     original.NeighboringBinsExtractor,
                                     'nb_original')

bp_nb_original_algorithm = RdhAlgorithm(original.BPNeighboringBinsEmbedder,
                                        original.BPNeighboringBinsExtractor,
                                        'bp_nb_original')

nb_vo_original_algorithm = RdhAlgorithm(original.NbVoEmbedder,
                                        original.NbVoExtractor,
                                        'nb_vo_original')

bp_nb_vo_original_algorithm = RdhAlgorithm(original.BPNbVoEmbedder,
                                           original.BPNbVoExtractor,
                                           'bp_nb_vo_original')
