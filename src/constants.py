ARTIFACT_LABELS = [
    "low-level-edge_shape",
    "low-level-texture",
    "low-level-color",
    "low-level-symbol",
    "high-level-semantics",
    "cognitive-level-commonsense",
    "cognitive-level-physics",
]

ARTIFACT_DISPLAY_NAMES = {
    "low-level-edge_shape": "Edge / shape inconsistency",
    "low-level-texture": "Texture inconsistency",
    "low-level-color": "Colour inconsistency",
    "low-level-symbol": "Symbol / text inconsistency",
    "high-level-semantics": "Semantic inconsistency",
    "cognitive-level-commonsense": "Commonsense violation",
    "cognitive-level-physics": "Physics violation",
}

ARTIFACT_TO_INDEX = {name: i for i, name in enumerate(ARTIFACT_LABELS)}
INDEX_TO_ARTIFACT = {i: name for name, i in ARTIFACT_TO_INDEX.items()}

TINY_GENERATOR_TO_ID = {
    "Real": 0,
    "ADM": 1,
    "BigGAN": 2,
    "GLIDE": 3,
    "Midjourney": 4,
    "SD14": 5,
    "SD15": 6,
    "VQDM": 7,
    "Wukong": 8,
}

TINY_ID_TO_GENERATOR = {v: k for k, v in TINY_GENERATOR_TO_ID.items()}
