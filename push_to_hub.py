import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load_file
from midi_model import config_name_list, MIDIModelConfig, MIDIModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default="", help="load ckpt"
    )
    parser.add_argument(
        "--config", type=str, default="auto",
        help="model config name, file or automatically find config.json"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="convert precision",
    )
    parser.add_argument(
        "--repo-id", type=str, default="midi-model-test",
        help="repo id"
    )
    parser.add_argument(
        "--private", action="store_true", default=False, help="private repo"
    )

    opt = parser.parse_args()
    print(opt)

    if opt.config in config_name_list:
        config = MIDIModelConfig.from_name(opt.config)
    elif opt.config == "auto":
        config_path = Path(opt.ckpt).parent / "config.json"
        if config_path.exists():
            config = MIDIModelConfig.from_json_file(config_path)
        else:
            raise ValueError("can not find config.json, please specify config")
    else:
        config = MIDIModelConfig.from_json_file(opt.config)

    model = MIDIModel(config=config)
    if opt.ckpt.endswith(".safetensors"):
        state_dict = safe_load_file(opt.ckpt)
    else:
        ckpt = torch.load(opt.ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    precision_dict = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    model.to(dtype=precision_dict[opt.precision]).eval()
    model.push_to_hub(repo_id=opt.repo_id, private=opt.private)
