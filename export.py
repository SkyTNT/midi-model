import argparse
from itertools import chain

import torch
import torch.nn as nn
from transformers import LlamaConfig, DynamicCache

from midi_model import MIDIModel, config_name_list, MIDIModelConfig


class MIDIModelBase(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = model.net

    def forward(self, x, past_kv):
        cache = DynamicCache.from_legacy_cache(past_kv)
        x = self.net.embed_tokens(x)
        x = x.sum(dim=-2)
        x = self.net.forward(inputs_embeds=x,
                             past_key_values=cache,
                             use_cache=True)
        return x.last_hidden_state, cache.to_legacy_cache()


class MIDIModelToken(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net_token = model.net_token
        self.lm_head = model.lm_head

    def forward(self, hidden_state, x, past_kv):
        cache = DynamicCache.from_legacy_cache(past_kv)
        x = self.net_token.embed_tokens(x)
        x = torch.cat([hidden_state, x], dim=1)
        hidden_state = x
        hidden_state = self.net_token.forward(inputs_embeds=hidden_state,
                                              past_key_values=cache,
                                              use_cache=True).last_hidden_state
        return self.lm_head(hidden_state), cache.to_legacy_cache()


def export_onnx(model, model_inputs, input_names, output_names, dynamic_axes, meta_data, path):
    import onnx
    from onnxsim import simplify
    torch.onnx.export(model,  # model being run
                      model_inputs,  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=14,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=input_names,  # the model's input names
                      output_names=output_names,  # the model's output names
                      verbose=True,
                      dynamic_axes=dynamic_axes
                      )
    onnx_model = onnx.load(path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    for k, v in meta_data.items():
        meta = model_simp.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_simp, path)
    print('finished exporting onnx')


def get_past_kv(config: LlamaConfig, batch_size=1, past_seq_len=16, torch_dtype= torch.float32, device="cpu"):
    head_size = config.hidden_size // config.num_attention_heads
    past_kv = [
        (
            torch.rand(batch_size, config.num_attention_heads,
                       past_seq_len, head_size, dtype=torch_dtype, device=device),
            torch.rand(batch_size, config.num_attention_heads,
                       past_seq_len, head_size, dtype=torch_dtype, device=device),
        )
        for _ in range(config.num_hidden_layers)
    ]
    input_names = list(
        chain.from_iterable(
            (f"past_key_values.{i}.key", f"past_key_values.{i}.value") for i in
            range(config.num_hidden_layers)
        )
    )
    output_names = list(
        chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(config.num_hidden_layers))
    )
    return past_kv, input_names, output_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default="model.ckpt", help="load ckpt"
    )
    parser.add_argument(
        "--config", type=str, default="tv2o-medium", choices=config_name_list, help="model config"
    )
    parser.add_argument(
        "--lora", type=str, default="", help="load lora"
    )
    parser.add_argument(
        "--model-base-out", type=str, default="model_base.onnx", help="model base output path"
    )
    parser.add_argument(
        "--model-token-out", type=str, default="model_token.onnx", help="model token output path"
    )
    opt = parser.parse_args()
    config = MIDIModelConfig.from_name(opt.config)
    tokenizer = config.tokenizer
    model = MIDIModel(config).to(device="cpu")
    ckpt = torch.load(opt.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    if opt.lora != "":
        model.load_merge_lora(opt.lora)
    model.eval()
    model_base = MIDIModelBase(model).eval()
    model_token = MIDIModelToken(model).eval()
    meta_data = {"config_name": opt.config, "config": config}
    past_kv_shape = {0: "batch", 2: "past_seq"}
    present_kv_shape = {0: "batch", 2: "present_seq"}
    with torch.no_grad():
        dynamic_axes = {
            "x": {0: "batch", 1: "mid_seq", 2: "token_seq"},
            "hidden": {0: "batch", 1: "mid_seq"}
        }
        x = torch.randint(tokenizer.vocab_size, (1, 16, tokenizer.max_token_seq), dtype=torch.int64, device="cpu")
        past_kv, input_names, output_names= get_past_kv(config.net_config, past_seq_len=16,
                                                        torch_dtype=torch.float32)
        for name in input_names:
            dynamic_axes[name] = past_kv_shape
        for name in output_names:
            dynamic_axes[name] = present_kv_shape
        input_names = [ "x"] + input_names
        output_names = ["hidden"] + output_names
        export_onnx(model_base, (x, past_kv),
                    input_names, output_names, dynamic_axes, meta_data, opt.model_base_out)

        dynamic_axes = {
            "x": {0: "batch", 1: "token_seq"},
            "hidden": {0: "batch", 1: "states"},
            "y": {0: "batch", 1: "token_seq1"}
        }
        hidden = torch.randn(1, 1, config.n_embd, device="cpu")
        x = torch.randint(tokenizer.vocab_size, (1, tokenizer.max_token_seq //2), dtype=torch.int64, device="cpu")
        past_kv, input_names, output_names = get_past_kv(config.net_token_config,
                                                         past_seq_len=(tokenizer.max_token_seq // 2),
                                                         torch_dtype=torch.float32)
        for name in input_names:
            dynamic_axes[name] = past_kv_shape
        for name in output_names:
            dynamic_axes[name] = present_kv_shape
        input_names = ["hidden", "x"] + input_names
        output_names = ["y"] + output_names
        export_onnx(model_token, (hidden, x, past_kv),
                    input_names, output_names, dynamic_axes, meta_data, opt.model_token_out)
