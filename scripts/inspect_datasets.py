from collections import Counter

from src.constants import ARTIFACT_LABELS
from src.datasets.clearml_data import DEFAULT_TINY_DATASET_ID, DEFAULT_XAIGD_DATASET_ID
from src.datasets.tiny_genimage import load_tiny_genimage, generator_mapping
from src.datasets.xaigd import load_xaigd_labeled, labels_to_multihot


print("\n=== Tiny-GenImage from ClearML ===")
tiny, tiny_root = load_tiny_genimage(dataset_id=DEFAULT_TINY_DATASET_ID)
print("Root:", tiny_root)
print(tiny)
print("Generator mapping:", generator_mapping(tiny["train"]))
print("Train labels:", Counter(map(int, tiny["train"]["label"])))
print("Validation labels:", Counter(map(int, tiny["validation"]["label"])))
print("Train generators:", Counter(map(int, tiny["train"]["generator"])))
print("Validation generators:", Counter(map(int, tiny["validation"]["generator"])))

sample = tiny["train"][0]
print("Sample image type:", type(sample["image"]))
print("Sample label/generator:", sample["label"], sample["generator"])

print("\n=== X-AIGD labeled from ClearML ===")
xaigd, xaigd_root = load_xaigd_labeled(dataset_id=DEFAULT_XAIGD_DATASET_ID)
print("Root:", xaigd_root)
print(xaigd)

counts = Counter()
for labels in xaigd["labeled_train"]["labels"]:
    y = labels_to_multihot(labels)
    for i, value in enumerate(y):
        if value:
            counts[ARTIFACT_LABELS[i]] += 1
print("Artifact image counts:", counts)

sample = xaigd["labeled_train"][0]
print("Sample image type:", type(sample["image"]))
print("Sample generator/uid:", sample["generator"], sample["uid"])
print("Number of annotations:", len(sample.get("labels", [])))
