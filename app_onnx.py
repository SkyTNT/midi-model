import argparse
import glob
import json
import os.path
import time
from sys import exit

import gradio as gr
import numpy as np
import onnxruntime as rt
import requests
import tqdm

import MIDI
from midi_synthesizer import synthesis
from midi_tokenizer import MIDITokenizer

MAX_SEED = np.iinfo(np.int32).max


def softmax(x, axis):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def sample_top_p_k(probs, p, k, generator=None):
    if generator is None:
        generator = np.random
    probs_idx = np.argsort(-probs, axis=-1)
    probs_sort = np.take_along_axis(probs, probs_idx, -1)
    probs_sum = np.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    mask = np.zeros(probs_sort.shape[-1])
    mask[:k] = 1
    probs_sort = probs_sort * mask
    probs_sort /= np.sum(probs_sort, axis=-1, keepdims=True)
    shape = probs_sort.shape
    probs_sort_flat = probs_sort.reshape(-1, shape[-1])
    probs_idx_flat = probs_idx.reshape(-1, shape[-1])
    next_token = np.stack([generator.choice(idxs, p=pvals) for pvals, idxs in zip(probs_sort_flat, probs_idx_flat)])
    next_token = next_token.reshape(*shape[:-1])
    return next_token


def generate(model, prompt=None, max_len=512, temp=1.0, top_p=0.98, top_k=20,
             disable_patch_change=False, disable_control_change=False, disable_channels=None, generator=None):
    if disable_channels is not None:
        disable_channels = [tokenizer.parameter_ids["channel"][c] for c in disable_channels]
    else:
        disable_channels = []
    if generator is None:
        generator = np.random
    max_token_seq = tokenizer.max_token_seq
    if prompt is None:
        input_tensor = np.full((1, max_token_seq), tokenizer.pad_id, dtype=np.int64)
        input_tensor[0, 0] = tokenizer.bos_id  # bos
    else:
        prompt = prompt[:, :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = np.pad(prompt, ((0, 0), (0, max_token_seq - prompt.shape[-1])),
                            mode="constant", constant_values=tokenizer.pad_id)
        input_tensor = prompt
    input_tensor = input_tensor[None, :, :]
    cur_len = input_tensor.shape[1]
    bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
    with bar:
        while cur_len < max_len:
            end = False
            hidden = model[0].run(None, {'x': input_tensor})[0][:, -1]
            next_token_seq = np.empty((1, 0), dtype=np.int64)
            event_name = ""
            for i in range(max_token_seq):
                mask = np.zeros(tokenizer.vocab_size, dtype=np.int64)
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
                logits = model[1].run(None, {'x': next_token_seq, "hidden": hidden})[0][:, -1:]
                scores = softmax(logits / temp, -1) * mask
                sample = sample_top_p_k(scores, top_p, top_k, generator)
                if i == 0:
                    next_token_seq = sample
                    eid = sample.item()
                    if eid == tokenizer.eos_id:
                        end = True
                        break
                    event_name = tokenizer.id_events[eid]
                else:
                    next_token_seq = np.concatenate([next_token_seq, sample], axis=1)
                    if len(tokenizer.events[event_name]) == i:
                        break
            if next_token_seq.shape[1] < max_token_seq:
                next_token_seq = np.pad(next_token_seq, ((0, 0), (0, max_token_seq - next_token_seq.shape[-1])),
                                        mode="constant", constant_values=tokenizer.pad_id)
            next_token_seq = next_token_seq[None, :, :]
            input_tensor = np.concatenate([input_tensor, next_token_seq], axis=1)
            cur_len += 1
            bar.update(1)
            yield next_token_seq.reshape(-1)
            if end:
                break


def create_msg(name, data):
    return {"name": name, "data": data}


def send_msgs(msgs):
    return json.dumps(msgs)


def run(tab, instruments, drum_kit, bpm, mid, midi_events, seed, seed_rand,
        gen_events, temp, top_p, top_k, allow_cc):
    mid_seq = []
    bpm = int(bpm)
    gen_events = int(gen_events)
    max_len = gen_events
    if seed_rand:
        seed = np.random.randint(0, MAX_SEED)
    generator = np.random.RandomState(seed)
    disable_patch_change = False
    disable_channels = None
    if tab == 0:
        i = 0
        mid = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
        if bpm != 0:
            mid.append(tokenizer.event2tokens(["set_tempo",0,0,0, bpm]))
        patches = {}
        if instruments is None:
            instruments = []
        for instr in instruments:
            patches[i] = patch2number[instr]
            i = (i + 1) if i != 8 else 10
        if drum_kit != "None":
            patches[9] = drum_kits2number[drum_kit]
        for i, (c, p) in enumerate(patches.items()):
            mid.append(tokenizer.event2tokens(["patch_change", 0, 0, i, c, p]))
        mid_seq = mid
        mid = np.asarray(mid, dtype=np.int64)
        if len(instruments) > 0:
            disable_patch_change = True
            disable_channels = [i for i in range(16) if i not in patches]
    elif mid is not None:
        mid = tokenizer.tokenize(MIDI.midi2score(mid))
        mid = np.asarray(mid, dtype=np.int64)
        mid = mid[:int(midi_events)]
        for token_seq in mid:
            mid_seq.append(token_seq.tolist())
    max_len += len(mid)
    events = [tokenizer.tokens2event(tokens) for tokens in mid_seq]
    init_msgs = [create_msg("visualizer_clear", None), create_msg("visualizer_append", events)]
    t = time.time() + 0.2
    yield mid_seq, None, None, seed, send_msgs(init_msgs)
    model = (model_base, model_token)
    midi_generator = generate(model, mid, max_len=max_len, temp=temp, top_p=top_p, top_k=top_k,
                         disable_patch_change=disable_patch_change, disable_control_change=not allow_cc,
                         disable_channels=disable_channels, generator=generator)
    events = []
    for i, token_seq in enumerate(midi_generator):
        token_seq = token_seq.tolist()
        mid_seq.append(token_seq)
        events.append(tokenizer.tokens2event(token_seq))
        ct = time.time()
        if ct - t > 0.2:
            yield mid_seq, None, None, seed, send_msgs([create_msg("visualizer_append", events), create_msg("progress", [i + 1, gen_events])])
            t = ct
            events = []

    mid = tokenizer.detokenize(mid_seq)
    with open(f"output.mid", 'wb') as f:
        f.write(MIDI.score2midi(mid))
    audio = synthesis(MIDI.score2opus(mid), soundfont_path)
    events = [tokenizer.tokens2event(tokens) for tokens in mid_seq]
    yield mid_seq, "output.mid", (44100, audio), seed, send_msgs([create_msg("visualizer_end", events)])


def cancel_run(mid_seq):
    if mid_seq is None:
        return None, None, []
    mid = tokenizer.detokenize(mid_seq)
    with open(f"output.mid", 'wb') as f:
        f.write(MIDI.score2midi(mid))
    audio = synthesis(MIDI.score2opus(mid), soundfont_path)
    events = [tokenizer.tokens2event(tokens) for tokens in mid_seq]
    return "output.mid", (44100, audio), send_msgs([create_msg("visualizer_end", events)])


def download(url, output_file):
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    with tqdm.tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
                   desc=f"Downloading {output_file}") as pbar:
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_if_not_exit(url, output_file):
    if os.path.exists(output_file):
        return
    try:
        download(url, output_file)
    except Exception as e:
        print(f"Failed to download {output_file} from {url}")
        raise e


def load_javascript(dir="javascript"):
    scripts_list = glob.glob(f"{dir}/*.js")
    javascript = ""
    for path in scripts_list:
        with open(path, "r", encoding="utf8") as jsfile:
            javascript += f"\n<!-- {path} --><script>{jsfile.read()}</script>"
    template_response_ori = gr.routes.templates.TemplateResponse

    def template_response(*args, **kwargs):
        res = template_response_ori(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response


number2drum_kits = {-1: "None", 0: "Standard", 8: "Room", 16: "Power", 24: "Electric", 25: "TR-808", 32: "Jazz",
                    40: "Blush", 48: "Orchestra"}
patch2number = {v: k for k, v in MIDI.Number2patch.items()}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--port", type=int, default=-1, help="gradio server port")
    parser.add_argument("--max-gen", type=int, default=4096, help="max")
    parser.add_argument("--soundfont-path", type=str, default="soundfont.sf2", help="soundfont")
    parser.add_argument("--model-base-path", type=str, default="model_base.onnx", help="model path")
    parser.add_argument("--model-token-path", type=str, default="model_token.onnx", help="model path")
    parser.add_argument("--soundfont-url", type=str,
                        default="https://huggingface.co/skytnt/midi-model/resolve/main/soundfont.sf2",
                        help="download soundfont to soundfont-path if file not exist")
    parser.add_argument("--model-base-url", type=str,
                        default="https://huggingface.co/skytnt/midi-model/resolve/main/onnx/model_base.onnx",
                        help="download model-base to model-base-path if file not exist")
    parser.add_argument("--model-token-url", type=str,
                        default="https://huggingface.co/skytnt/midi-model/resolve/main/onnx/model_token.onnx",
                        help="download model-token to model-token-path if file not exist")
    opt = parser.parse_args()

    try:
        download_if_not_exit(opt.soundfont_url, opt.soundfont_path)
        download_if_not_exit(opt.model_base_url, opt.model_base_path)
        download_if_not_exit(opt.model_token_url, opt.model_token_path)
    except Exception as e:
        print(e)
        input("Failed to download files.\nPress any key to continue...")
        exit(-1)
    soundfont_path = opt.soundfont_path
    tokenizer = MIDITokenizer()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        model_base = rt.InferenceSession(opt.model_base_path, providers=providers)
        model_token = rt.InferenceSession(opt.model_token_path, providers=providers)
    except Exception as e:
        print(e)
        input("Failed to load models, maybe you need to delete them and re-download it.\nPress any key to continue...")
        exit(-1)

    load_javascript()
    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Midi Composer</h1>")
        gr.Markdown("![Visitors](https://api.visitorbadge.io/api/visitors?path=skytnt.midi-composer&style=flat)\n\n"
                    "Midi event transformer for music generation\n\n"
                    "Demo for [SkyTNT/midi-model](https://github.com/SkyTNT/midi-model)\n\n"
                    "[Open In Colab]"
                    "(https://colab.research.google.com/github/SkyTNT/midi-model/blob/main/demo.ipynb)"
                    " for faster running and longer generation\n\n"
                    "**Update v1.2**: Optimise the tokenizer and dataset"
                    )
        js_msg = gr.Textbox(elem_id="msg_receiver", visible=False)
        js_msg.change(None, [js_msg], [], js="""
        (msg_json) =>{
            let msgs = JSON.parse(msg_json);
            executeCallbacks(msgReceiveCallbacks, msgs);
            return [];
        }
        """)
        tab_select = gr.State(value=0)
        with gr.Tabs():
            with gr.TabItem("instrument prompt") as tab1:
                input_instruments = gr.Dropdown(label="🪗instruments (auto if empty)", choices=list(patch2number.keys()),
                                                multiselect=True, max_choices=15, type="value")
                input_drum_kit = gr.Dropdown(label="🥁drum kit", choices=list(drum_kits2number.keys()), type="value",
                                             value="None")
                input_bpm = gr.Slider(label="BPM (beats per minute, auto if 0)", minimum=0, maximum=255,
                                              step=1,
                                              value=0)
                example1 = gr.Examples([
                    [[], "None"],
                    [["Acoustic Grand"], "None"],
                    [['Acoustic Grand', 'SynthStrings 2', 'SynthStrings 1', 'Pizzicato Strings',
                      'Pad 2 (warm)', 'Tremolo Strings', 'String Ensemble 1'], "Orchestra"],
                    [['Trumpet', 'Oboe', 'Trombone', 'String Ensemble 1', 'Clarinet',
                      'French Horn', 'Pad 4 (choir)', 'Bassoon', 'Flute'], "None"],
                    [['Flute', 'French Horn', 'Clarinet', 'String Ensemble 2', 'English Horn', 'Bassoon',
                      'Oboe', 'Pizzicato Strings'], "Orchestra"],
                    [['Electric Piano 2', 'Lead 5 (charang)', 'Electric Bass(pick)', 'Lead 2 (sawtooth)',
                      'Pad 1 (new age)', 'Orchestra Hit', 'Cello', 'Electric Guitar(clean)'], "Standard"],
                    [["Electric Guitar(clean)", "Electric Guitar(muted)", "Overdriven Guitar", "Distortion Guitar",
                      "Electric Bass(finger)"], "Standard"]
                ], [input_instruments, input_drum_kit])
            with gr.TabItem("midi prompt") as tab2:
                input_midi = gr.File(label="input midi", file_types=[".midi", ".mid"], type="binary")
                input_midi_events = gr.Slider(label="use first n midi events as prompt", minimum=1, maximum=512,
                                              step=1,
                                              value=128)

        tab1.select(lambda: 0, None, tab_select, queue=False)
        tab2.select(lambda: 1, None, tab_select, queue=False)
        input_seed = gr.Slider(label="seed", minimum=0, maximum=2 ** 31 - 1,
                               step=1, value=0)
        input_seed_rand = gr.Checkbox(label="random seed", value=True)
        input_gen_events = gr.Slider(label="generate max n midi events", minimum=1, maximum=opt.max_gen,
                                     step=1, value=opt.max_gen // 2)
        with gr.Accordion("options", open=False):
            input_temp = gr.Slider(label="temperature", minimum=0.1, maximum=1.2, step=0.01, value=1)
            input_top_p = gr.Slider(label="top p", minimum=0.1, maximum=1, step=0.01, value=0.98)
            input_top_k = gr.Slider(label="top k", minimum=1, maximum=128, step=1, value=20)
            input_allow_cc = gr.Checkbox(label="allow midi cc event", value=True)
            example3 = gr.Examples([[1, 0.98, 20], [1, 0.98, 12]], [input_temp, input_top_p, input_top_k])
        run_btn = gr.Button("generate", variant="primary")
        stop_btn = gr.Button("stop and output")
        output_midi_seq = gr.State()
        output_midi_visualizer = gr.HTML(elem_id="midi_visualizer_container")
        output_audio = gr.Audio(label="output audio", format="mp3", elem_id="midi_audio")
        output_midi = gr.File(label="output midi", file_types=[".mid"])
        run_event = run_btn.click(run, [tab_select, input_instruments, input_drum_kit, input_bpm,
                                        input_midi, input_midi_events, input_seed, input_seed_rand, input_gen_events,
                                        input_temp, input_top_p, input_top_k, input_allow_cc],
                                  [output_midi_seq, output_midi, output_audio, input_seed, js_msg],
                                  concurrency_limit=3)
        stop_btn.click(cancel_run, [output_midi_seq], [output_midi, output_audio, js_msg], cancels=run_event, queue=False)
    try:
        port = opt.port
        if port == -1:
            port = None
        app.launch(server_port=port, share=opt.share, inbrowser=True)
    except Exception as e:
        print(e)
        input("Failed to launch webui.\nPress any key to continue...")
        exit(-1)
