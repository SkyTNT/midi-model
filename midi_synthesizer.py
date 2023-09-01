import fluidsynth
import numpy as np


def synthesis(midi_opus, soundfont_path, sample_rate=44100):
    ticks_per_beat = midi_opus[0]
    event_list = []
    for track_idx, track in enumerate(midi_opus[1:]):
        abs_t = 0
        for event in track:
            abs_t += event[1]
            event_new = [*event]
            event_new[1] = abs_t
            event_list.append(event_new)
    event_list = sorted(event_list, key=lambda e: e[1])

    tempo = int((60 / 120) * 10 ** 6)  # default 120 bpm
    ss = np.empty((0, 2), dtype=np.int16)
    fl = fluidsynth.Synth(samplerate=float(sample_rate))
    sfid = fl.sfload(soundfont_path)
    last_t = 0
    for c in range(16):
        fl.program_select(c, sfid, 128 if c == 9 else 0, 0)
    for event in event_list:
        name = event[0]
        sample_len = int(((event[1] / ticks_per_beat) * tempo / (10 ** 6)) * sample_rate)
        sample_len -= int(((last_t / ticks_per_beat) * tempo / (10 ** 6)) * sample_rate)
        last_t = event[1]
        if sample_len > 0:
            sample = fl.get_samples(sample_len).reshape(sample_len, 2)
            ss = np.concatenate([ss, sample])
        if name == "set_tempo":
            tempo = event[2]
        elif name == "patch_change":
            c, p = event[2:4]
            fl.program_select(c, sfid, 128 if c == 9 else 0, p)
        elif name == "control_change":
            c, cc, v = event[2:5]
            fl.cc(c, cc, v)
        elif name == "note_on" and event[3] > 0:
            c, p, v = event[2:5]
            fl.noteon(c, p, v)
        elif name == "note_off" or (name == "note_on" and event[3] == 0):
            c, p = event[2:4]
            fl.noteoff(c, p)

    fl.delete()
    if ss.shape[0] > 0:
        max_val = np.abs(ss).max()
        if max_val != 0:
            ss = (ss / max_val) * np.iinfo(np.int16).max
    ss = ss.astype(np.int16)
    return ss