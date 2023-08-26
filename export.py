import torch
import argparse
import torch.nn as nn
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer


class MIDIModelBase(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = model.net


MIDIModelBase.forward = MIDIModel.forward


class MIDIModelToken(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net_token = model.net_token
        self.lm_head = model.lm_head


MIDIModelToken.forward = MIDIModel.forward_token


def export_onnx(model, model_inputs, input_names, output_names, dynamic_axes, path):
    import onnx
    from onnxsim import simplify
    torch.onnx.export(model,  # model being run
                      model_inputs,  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=input_names,  # the model's input names
                      output_names=output_names,  # the model's output names
                      verbose=True,
                      dynamic_axes=dynamic_axes
                      )
    onnx_model = onnx.load(path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, path)
    print('finished exporting onnx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default="model.ckpt", help="load ckpt"
    )
    parser.add_argument(
        "--model-base-out", type=str, default="model_base.onnx", help="model base output path"
    )
    parser.add_argument(
        "--model-token-out", type=str, default="model_token.onnx", help="model token output path"
    )
    opt = parser.parse_args()
    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device="cpu")
    ckpt = torch.load(opt.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model_base = MIDIModelBase(model).eval()
    model_token = MIDIModelToken(model).eval()
    with torch.no_grad():
        x = torch.randint(tokenizer.vocab_size, (1, 16, tokenizer.max_token_seq), dtype=torch.int64, device="cpu")
        export_onnx(model_base, x, ["x"], ["hidden"], {"x": {0: "batch", 1: "mid_seq", 2: "token_seq"},
                                                       "hidden": {0: "batch", 1: "mid_seq", 2: "emb"}},
                    opt.model_base_out)

        hidden = torch.randn(1, 1024, device="cpu")
        x = torch.randint(tokenizer.vocab_size, (1, tokenizer.max_token_seq), dtype=torch.int64, device="cpu")
        export_onnx(model_token, (hidden, x), ["hidden", "x"], ["y"], {"x": {0: "batch", 1: "token_seq"},
                                                                       "hidden": {0: "batch", 1: "emb"},
                                                                       "y": {0: "batch", 1: "token_seq1", 2: "voc"}},
                    opt.model_token_out)
