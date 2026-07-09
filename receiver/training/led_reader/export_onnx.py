from __future__ import annotations

import argparse
from pathlib import Path

import torch

from receiver_tools.dataset import repo_root_from_tools

from led_reader.model import CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH, create_model


def parse_args() -> argparse.Namespace:
    root = repo_root_from_tools()
    parser = argparse.ArgumentParser(description="Export LED crop scorer to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "receiver" / "models" / "led_reader" / "led_reader_crop_v003_gate.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "receiver" / "models" / "led_reader" / "led_reader_crop_v003_gate.onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = create_model()
    state_dict = checkpoint["state_dict"]
    if "gate_log_slope" in state_dict and "gate_slope" not in state_dict:
        state_dict = dict(state_dict)
        state_dict["gate_slope"] = torch.nn.functional.softplus(state_dict.pop("gate_log_slope")) + 0.25
    model.load_state_dict(state_dict)
    model.eval()
    dummy = torch.zeros(1, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH, dtype=torch.float32)
    dummy_likelihood = torch.ones(1, dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy, dummy_likelihood),
        args.out,
        input_names=["led_crop", "detector_likelihood"],
        output_names=["led_logit"],
        dynamic_axes={
            "led_crop": {0: "batch"},
            "detector_likelihood": {0: "batch"},
            "led_logit": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
