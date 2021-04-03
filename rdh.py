import original_embed
import original_extract
import scaling_embed
import scaling_extract
from unidirection import embed, extract


class RDH:
    def __init__(self, label, embed, extract=None):
        self.embed = embed
        self.extract = extract
        self.label = label

    def __str__(self) -> str:
        return self.label


original_algorithm = RDH('original', original_embed.embed, original_extract.extract)
scaling_algorithm = RDH('scaling', scaling_embed.embed, scaling_extract.extract)
unidirectional_algorithm = RDH('unidirectional', embed.embed, extract.extract)
