import fluidsynth
import numpy as np


def synthesis(midi_opus, soundfont_path, sample_rate=44100):
    ticks_per_beat = midi_opus[0]
    tempo = int((60 / 140) * 10 ** 6)  # default 140 bpm

    fl = fluidsynth.Synth(samplerate=float(sample_rate))
    sfid = fl.sfload(soundfont_path)
    waveforms = []
    for track_idx, track in enumerate(midi_opus[1:]):
        ss = np.empty((0, 2), dtype=np.int16)
        cur_t = 0
        last_t = 0
        fl.system_reset()
        fl.get_samples(int(2 * sample_rate))  # get 2 seconds sample to eliminate reverb sound from previous track
        for c in range(16):
            fl.program_select(c, sfid, 128 if c == 9 else 0, 0)
        for event in track:
            name = event[0]
            cur_t += event[1]
            sample_len = int(((cur_t / ticks_per_beat) * tempo / (10 ** 6)) * sample_rate)
            sample_len -= int(((last_t / ticks_per_beat) * tempo / (10 ** 6)) * sample_rate)
            last_t = cur_t
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
        waveforms.append(ss)
    fl.delete()
    synthesized = np.zeros((max([w.shape[0] for w in waveforms]), 2), dtype=np.int32)
    for waveform in waveforms:
        synthesized[:waveform.shape[0]] += waveform
    max_val = np.abs(synthesized).max()
    if max_val != 0:
        synthesized = (synthesized / max_val) * np.iinfo(np.int16).max
    synthesized = synthesized.astype(np.int16)

    return synthesized
