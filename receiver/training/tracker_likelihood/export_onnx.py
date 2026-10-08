from __future__ import annotations

import argparse
from pathlib import Path

import torch

from receiver_tools.dataset import repo_root_from_tools

from tracker_likelihood.model import PATCH_CHANNELS, PATCH_HEIGHT, PATCH_WIDTH, create_model


def parse_args() -> argparse.Namespace:
    root = repo_root_from_tools()
    parser = argparse.ArgumentParser(description="Export tracker likelihood CNN to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "receiver" / "models" / "tracker_likelihood" / "tracker_likelihood_precise_v005.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "receiver" / "models" / "tracker_likelihood" / "tracker_likelihood_precise_v005.onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    input_shape = checkpoint.get("input_shape", [PATCH_CHANNELS, PATCH_HEIGHT, PATCH_WIDTH])
    _, patch_height, patch_width = (int(value) for value in input_shape)
    model = create_model(patch_width, patch_height)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy = torch.zeros(1, PATCH_CHANNELS, patch_height, patch_width, dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["patch"],
        output_names=["likelihood_logits"],
        dynamic_axes={"patch": {0: "batch"}, "likelihood_logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
