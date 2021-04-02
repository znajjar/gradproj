class RDH:
    def __init__(self, label, embed, extract=None):
        self.embed = embed
        self.extract = extract
        self.label = label

    def __str__(self) -> str:
        return self.label
