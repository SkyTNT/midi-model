import argparse
import glob

import PIL
import gradio as gr
import numpy as np
import torch

import torch.nn.functional as F
import tqdm

import MIDI
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
from midi_synthesizer import synthesis
from huggingface_hub import hf_hub_download

@torch.inference_mode()
def generate(prompt=None, max_len=512, temp=1.0, top_p=0.98, top_k=20,
             disable_patch_change=False, disable_control_change=False, disable_channels=None, amp=True):
    if disable_channels is not None:
        disable_channels = [tokenizer.parameter_ids["channel"][c] for c in disable_channels]
    else:
        disable_channels = []
    max_token_seq = tokenizer.max_token_seq
    if prompt is None:
        input_tensor = torch.full((1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=model.device)
        input_tensor[0, 0] = tokenizer.bos_id  # bos
    else:
        prompt = prompt[:, :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = np.pad(prompt, ((0, 0), (0, max_token_seq - prompt.shape[-1])),
                            mode="constant", constant_values=tokenizer.pad_id)
        input_tensor = torch.from_numpy(prompt).to(dtype=torch.long, device=model.device)
    input_tensor = input_tensor.unsqueeze(0)
    cur_len = input_tensor.shape[1]
    bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
    with bar, torch.cuda.amp.autocast(enabled=amp):
        while cur_len < max_len:
            end = False
            hidden = model.forward(input_tensor)[0, -1].unsqueeze(0)
            next_token_seq = None
            event_name = ""
            for i in range(max_token_seq):
                mask = torch.zeros(tokenizer.vocab_size, dtype=torch.int64, device=model.device)
                if i == 0:
                    mask_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                    if disable_patch_change:
                        mask_ids.remove(tokenizer.event_ids["patch_change"])
                    if disable_control_change:
                        mask_ids.remove(tokenizer.event_ids["control_change"])
                    mask[mask_ids] = 1
                else:
                    param_name = tokenizer.events[event_name][i - 1]
                    mask_ids = tokenizer.parameter_ids[param_name]
                    if param_name == "channel":
                        mask_ids = [i for i in mask_ids if i not in disable_channels]
                    mask[mask_ids] = 1
                logits = model.forward_token(hidden, next_token_seq)[:, -1:]
                scores = torch.softmax(logits / temp, dim=-1) * mask
                sample = model.sample_top_p_k(scores, top_p, top_k)
                if i == 0:
                    next_token_seq = sample
                    eid = sample.item()
                    if eid == tokenizer.eos_id:
                        end = True
                        break
                    event_name = tokenizer.id_events[eid]
                else:
                    next_token_seq = torch.cat([next_token_seq, sample], dim=1)
                    if len(tokenizer.events[event_name]) == i:
                        break
            if next_token_seq.shape[1] < max_token_seq:
                next_token_seq = F.pad(next_token_seq, (0, max_token_seq - next_token_seq.shape[1]),
                                       "constant", value=tokenizer.pad_id)
            next_token_seq = next_token_seq.unsqueeze(1)
            input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
            cur_len += 1
            bar.update(1)
            yield next_token_seq.reshape(-1).cpu().numpy()
            if end:
                break


def run(tab, instruments, drum_kit, mid, midi_events, gen_events, temp, top_p, top_k, allow_cc, amp):
    mid_seq = []
    max_len = int(gen_events)
    img_len = 1024
    img = np.full((128 * 2, img_len, 3), 255, dtype=np.uint8)
    state = {"t1": 0, "t": 0, "cur_pos": 0}
    rand = np.random.RandomState(0)
    colors = {(i, j): rand.randint(0, 200, 3) for i in range(128) for j in range(16)}

    def draw_event(tokens):
        if tokens[0] in tokenizer.id_events:
            name = tokenizer.id_events[tokens[0]]
            if len(tokens) <= len(tokenizer.events[name]):
                return
            params = tokens[1:]
            params = [params[i] - tokenizer.parameter_ids[p][0] for i, p in enumerate(tokenizer.events[name])]
            if not all([0 <= params[i] < tokenizer.event_parameters[p] for i, p in enumerate(tokenizer.events[name])]):
                return
            event = [name] + params
            state["t1"] += event[1]
            t = state["t1"] * 16 + event[2]
            state["t"] = t
            if name == "note":
                tr, d, c, p = event[3:7]
                shift = t + d - (state["cur_pos"] + img_len)
                if shift > 0:
                    img[:, :-shift] = img[:, shift:]
                    img[:, -shift:] = 255
                    state["cur_pos"] += shift
                t = t - state["cur_pos"]
                img[p * 2:(p + 1) * 2, t: t + d] = colors[(tr, c)]

    def get_img():
        t = state["t"] - state["cur_pos"]
        img_new = img.copy()
        img_new[:, t: t + 2] = 0
        return PIL.Image.fromarray(np.flip(img_new, 0))

    disable_patch_change = False
    disable_channels = None
    if tab == 0:
        i = 0
        mid = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
        patches = {}
        for instr in instruments:
            patches[i] = patch2number[instr]
            i = (i + 1) if i != 9 else 10
        if drum_kit != "None":
            patches[9] = drum_kits2number[drum_kit]
        for i, (c, p) in enumerate(patches.items()):
            mid.append(tokenizer.event2tokens(["patch_change", 0, 0, i, c, p]))
        mid_seq = mid
        mid = np.asarray(mid, dtype=np.int64)
        if len(instruments) > 0 or drum_kit != "None":
            disable_patch_change = True
            disable_channels = [i for i in range(16) if i not in patches]
    elif mid is not None:
        mid = tokenizer.tokenize(MIDI.midi2score(mid))
        mid = np.asarray(mid, dtype=np.int64)
        mid = mid[:int(midi_events)]
        max_len += len(mid)
        for token_seq in mid:
            mid_seq.append(token_seq)
            draw_event(token_seq)
    generator = generate(mid, max_len=max_len, temp=temp, top_p=top_p, top_k=top_k,
                         disable_patch_change=disable_patch_change, disable_control_change=not allow_cc,
                         disable_channels=disable_channels, amp=amp)
    for token_seq in generator:
        mid_seq.append(token_seq)
        draw_event(token_seq)
        yield mid_seq, get_img(), None, None
    mid = tokenizer.detokenize(mid_seq)
    with open(f"output.mid", 'wb') as f:
        f.write(MIDI.score2midi(mid))
    audio = synthesis(MIDI.score2opus(mid), soundfont_path)
    yield mid_seq, get_img(), "output.mid", (44100, audio)


def cancel_run(mid_seq):
    mid = tokenizer.detokenize(mid_seq)
    with open(f"output.mid", 'wb') as f:
        f.write(MIDI.score2midi(mid))
    audio = synthesis(MIDI.score2opus(mid), soundfont_path)
    return "output.mid", (44100, audio)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return "success"


def get_model_path():
    model_paths = sorted(glob.glob("**/*.ckpt", recursive=True))
    return model_path_input.update(choices=model_paths)


number2drum_kits = {-1: "None", 0: "Standard", 8: "Room", 16: "Power", 24: "Electric", 25: "TR-808", 32: "Jazz",
                    40: "Blush", 48: "Orchestra"}
patch2number = {v: k for k, v in MIDI.Number2patch.items()}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="gradio server port")
    parser.add_argument("--device", type=str, default="cuda", help="device to run model")
    soundfont_path = hf_hub_download(repo_id="skytnt/midi-model", filename="soundfont.sf2")
    opt = parser.parse_args()
    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device=opt.device)

    app = gr.Blocks()
    with app:
        with gr.Accordion(label="Model option", open=False):
            load_model_path_btn = gr.Button("Get Models")
            model_path_input = gr.Dropdown(label="model")
            load_model_path_btn.click(get_model_path, [], model_path_input)
            load_model_btn = gr.Button("Load")
            model_msg = gr.Textbox()
            load_model_btn.click(
                load_model, model_path_input, model_msg
            )
        tab_select = gr.Variable(value=0)
        with gr.Tabs():
            with gr.TabItem("instrument prompt") as tab1:
                input_instruments = gr.Dropdown(label="instruments (auto if empty)", choices=list(patch2number.keys()),
                                                multiselect=True, max_choices=15, type="value")
                input_drum_kit = gr.Dropdown(label="drum kit", choices=list(drum_kits2number.keys()), type="value",
                                             value="None")
                example1 = gr.Examples([
                    [[], "None"],
                    [["Acoustic Grand"], "None"],
                    [["Acoustic Grand", "Violin", "Viola", "Cello", "Contrabass"], "Orchestra"],
                    [["Flute", "Cello", "Bassoon", "Tuba"], "None"],
                    [["Violin", "Viola", "Cello", "Contrabass", "Trumpet", "French Horn", "Brass Section",
                      "Flute", "Piccolo", "Tuba", "Trombone", "Timpani"], "Orchestra"],
                    [["Acoustic Guitar(nylon)", "Acoustic Guitar(steel)", "Electric Guitar(jazz)",
                      "Electric Guitar(clean)", "Electric Guitar(muted)", "Overdriven Guitar", "Distortion Guitar",
                      "Electric Bass(finger)"], "Standard"]
                ], [input_instruments, input_drum_kit])
            with gr.TabItem("midi prompt") as tab2:
                input_midi = gr.File(label="input midi", file_types=[".midi", ".mid"], type="binary")
                input_midi_events = gr.Slider(label="use first n midi events as prompt", minimum=1, maximum=512,
                                              step=1,
                                              value=128)

        tab1.select(lambda: 0, None, tab_select, queue=False)
        tab2.select(lambda: 1, None, tab_select, queue=False)
        input_gen_events = gr.Slider(label="generate n midi events", minimum=1, maximum=4096, step=1, value=512)
        with gr.Accordion("options", open=False):
            input_temp = gr.Slider(label="temperature", minimum=0.1, maximum=1.2, step=0.01, value=1)
            input_top_p = gr.Slider(label="top p", minimum=0.1, maximum=1, step=0.01, value=0.98)
            input_top_k = gr.Slider(label="top k", minimum=1, maximum=20, step=1, value=12)
            input_allow_cc = gr.Checkbox(label="allow midi cc event", value=True)
            input_amp = gr.Checkbox(label="enable amp", value=True)
            example3 = gr.Examples([[1, 0.98, 12], [1.2, 0.95, 8]], [input_temp, input_top_p, input_top_k])
        run_btn = gr.Button("generate", variant="primary")
        stop_btn = gr.Button("stop and output")
        output_midi_seq = gr.Variable()
        output_midi_img = gr.Image(label="output image")
        output_midi = gr.File(label="output midi", file_types=[".mid"])
        output_audio = gr.Audio(label="output audio", format="mp3")
        run_event = run_btn.click(run, [tab_select, input_instruments, input_drum_kit, input_midi, input_midi_events,
                                        input_gen_events, input_temp, input_top_p, input_top_k,
                                        input_allow_cc, input_amp],
                                  [output_midi_seq, output_midi_img, output_midi, output_audio])
        stop_btn.click(cancel_run, output_midi_seq, [output_midi, output_audio], cancels=run_event, queue=False)
    app.queue(1).launch(server_port=opt.port)
