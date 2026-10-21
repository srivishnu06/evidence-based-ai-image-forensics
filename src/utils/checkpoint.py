from pathlib import Path
import torch


def save_checkpoint(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        **metadata,
    }
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)
