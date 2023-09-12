import random

import PIL
import numpy as np


class MIDITokenizer:
    def __init__(self):
        self.vocab_size = 0

        def allocate_ids(size):
            ids = [self.vocab_size + i for i in range(size)]
            self.vocab_size += size
            return ids

        self.pad_id = allocate_ids(1)[0]
        self.bos_id = allocate_ids(1)[0]
        self.eos_id = allocate_ids(1)[0]
        self.events = {
            "note": ["time1", "time2", "track", "duration", "channel", "pitch", "velocity"],
            "patch_change": ["time1", "time2", "track", "channel", "patch"],
            "control_change": ["time1", "time2", "track", "channel", "controller", "value"],
            "set_tempo": ["time1", "time2", "track", "bpm"],
        }
        self.event_parameters = {
            "time1": 128, "time2": 16, "duration": 2048, "track": 128, "channel": 16, "pitch": 128, "velocity": 128,
            "patch": 128, "controller": 128, "value": 128, "bpm": 256
        }
        self.event_ids = {e: allocate_ids(1)[0] for e in self.events.keys()}
        self.id_events = {i: e for e, i in self.event_ids.items()}
        self.parameter_ids = {p: allocate_ids(s) for p, s in self.event_parameters.items()}
        self.max_token_seq = max([len(ps) for ps in self.events.values()]) + 1

    def tempo2bpm(self, tempo):
        tempo = tempo / 10 ** 6  # us to s
        bpm = 60 / tempo
        return bpm

    def bpm2tempo(self, bpm):
        if bpm == 0:
            bpm = 1
        tempo = int((60 / bpm) * 10 ** 6)
        return tempo

    def tokenize(self, midi_score, add_bos_eos=True):
        ticks_per_beat = midi_score[0]
        event_list = {}
        for track_idx, track in enumerate(midi_score[1:129]):
            last_notes = {}
            for event in track:
                t = round(16 * event[1] / ticks_per_beat)  # quantization
                new_event = [event[0], t // 16, t % 16, track_idx] + event[2:]
                if event[0] == "note":
                    new_event[4] = max(1, round(16 * new_event[4] / ticks_per_beat))
                elif event[0] == "set_tempo":
                    new_event[4] = int(self.tempo2bpm(new_event[4]))
                if event[0] == "note":
                    key = tuple(new_event[:4] + new_event[5:-1])
                else:
                    key = tuple(new_event[:-1])
                if event[0] == "note":  # to eliminate note overlap due to quantization
                    cp = tuple(new_event[5:7])
                    if cp in last_notes:
                        last_note_key, last_note = last_notes[cp]
                        last_t = last_note[1] * 16 + last_note[2]
                        last_note[4] = max(0, min(last_note[4], t - last_t))
                        if last_note[4] == 0:
                            event_list.pop(last_note_key)
                    last_notes[cp] = (key, new_event)
                event_list[key] = new_event
        event_list = list(event_list.values())
        event_list = sorted(event_list, key=lambda e: e[1:4])
        midi_seq = []

        last_t1 = 0
        for event in event_list:
            name = event[0]
            if name in self.event_ids:
                params = event[1:]
                cur_t1 = params[0]
                params[0] = params[0] - last_t1
                if not all([0 <= params[i] < self.event_parameters[p] for i, p in enumerate(self.events[name])]):
                    continue
                tokens = [self.event_ids[name]] + [self.parameter_ids[p][params[i]]
                                                   for i, p in enumerate(self.events[name])]
                tokens += [self.pad_id] * (self.max_token_seq - len(tokens))
                midi_seq.append(tokens)
                last_t1 = cur_t1

        if add_bos_eos:
            bos = [self.bos_id] + [self.pad_id] * (self.max_token_seq - 1)
            eos = [self.eos_id] + [self.pad_id] * (self.max_token_seq - 1)
            midi_seq = [bos] + midi_seq + [eos]
        return midi_seq

    def event2tokens(self, event):
        name = event[0]
        params = event[1:]
        tokens = [self.event_ids[name]] + [self.parameter_ids[p][params[i]]
                                           for i, p in enumerate(self.events[name])]
        tokens += [self.pad_id] * (self.max_token_seq - len(tokens))
        return tokens

    def tokens2event(self, tokens):
        if tokens[0] in self.id_events:
            name = self.id_events[tokens[0]]
            if len(tokens) <= len(self.events[name]):
                return []
            params = tokens[1:]
            params = [params[i] - self.parameter_ids[p][0] for i, p in enumerate(self.events[name])]
            if not all([0 <= params[i] < self.event_parameters[p] for i, p in enumerate(self.events[name])]):
                return []
            event = [name] + params
            return event
        return []

    def detokenize(self, midi_seq):
        ticks_per_beat = 480
        tracks_dict = {}
        t1 = 0
        for tokens in midi_seq:
            if tokens[0] in self.id_events:
                name = self.id_events[tokens[0]]
                if len(tokens) <= len(self.events[name]):
                    continue
                params = tokens[1:]
                params = [params[i] - self.parameter_ids[p][0] for i, p in enumerate(self.events[name])]
                if not all([0 <= params[i] < self.event_parameters[p] for i, p in enumerate(self.events[name])]):
                    continue
                event = [name] + params
                if name == "set_tempo":
                    event[4] = self.bpm2tempo(event[4])
                if event[0] == "note":
                    event[4] = int(event[4] * ticks_per_beat / 16)
                t1 += event[1]
                t = t1 * 16 + event[2]
                t = int(t * ticks_per_beat / 16)
                track_idx = event[3]
                if track_idx not in tracks_dict:
                    tracks_dict[track_idx] = []
                tracks_dict[track_idx].append([event[0], t] + event[4:])
        tracks = list(tracks_dict.values())

        for i in range(len(tracks)):  # to eliminate note overlap
            track = tracks[i]
            track = sorted(track, key=lambda e: e[1])
            last_note_t = {}
            zero_len_notes = []
            for e in reversed(track):
                if e[0] == "note":
                    t, d, c, p = e[1:5]
                    key = (c, p)
                    if key in last_note_t:
                        d = min(d, max(last_note_t[key] - t, 0))
                    last_note_t[key] = t
                    e[2] = d
                    if d == 0:
                        zero_len_notes.append(e)
            for e in zero_len_notes:
                track.remove(e)
            tracks[i] = track
        return [ticks_per_beat, *tracks]

    def midi2img(self, midi_score):
        ticks_per_beat = midi_score[0]
        notes = []
        max_time = 1
        track_num = len(midi_score[1:])
        for track_idx, track in enumerate(midi_score[1:]):
            for event in track:
                t = round(16 * event[1] / ticks_per_beat)
                if event[0] == "note":
                    d = max(1, round(16 * event[2] / ticks_per_beat))
                    c, p = event[3:5]
                    max_time = max(max_time, t + d + 1)
                    notes.append((track_idx, c, p, t, d))
        img = np.zeros((128, max_time, 3), dtype=np.uint8)
        colors = {(i, j): np.random.randint(50, 256, 3) for i in range(track_num) for j in range(16)}
        for note in notes:
            tr, c, p, t, d = note
            img[p, t: t + d] = colors[(tr, c)]
        img = PIL.Image.fromarray(np.flip(img, 0))
        return img

    def augment(self, midi_seq, max_pitch_shift=4, max_vel_shift=10, max_cc_val_shift=10, max_bpm_shift=10,
                max_track_shift=1, max_channel_shift=16):
        pitch_shift = random.randint(-max_pitch_shift, max_pitch_shift)
        vel_shift = random.randint(-max_vel_shift, max_vel_shift)
        cc_val_shift = random.randint(-max_cc_val_shift, max_cc_val_shift)
        bpm_shift = random.randint(-max_bpm_shift, max_bpm_shift)
        track_shift = random.randint(0, max_track_shift)
        channel_shift = random.randint(0, max_channel_shift)
        midi_seq_new = []
        for tokens in midi_seq:
            tokens_new = [*tokens]
            if tokens[0] in self.id_events:
                name = self.id_events[tokens[0]]
                for i, pn in enumerate(self.events[name]):
                    if pn == "track":
                        tr = tokens[1 + i] - self.parameter_ids[pn][0]
                        tr += track_shift
                        tr = tr % self.event_parameters[pn]
                        tokens_new[1 + i] = self.parameter_ids[pn][tr]
                    elif pn == "channel":
                        c = tokens[1 + i] - self.parameter_ids[pn][0]
                        c0 = c
                        c += channel_shift
                        c = c % self.event_parameters[pn]
                        if c0 == 9:
                            c = 9
                        elif c == 9:
                            c = (9 + channel_shift) % self.event_parameters[pn]
                        tokens_new[1 + i] = self.parameter_ids[pn][c]

                if name == "note":
                    c = tokens[5] - self.parameter_ids["channel"][0]
                    p = tokens[6] - self.parameter_ids["pitch"][0]
                    v = tokens[7] - self.parameter_ids["velocity"][0]
                    if c != 9:  # no shift for drums
                        p += pitch_shift
                    if not 0 <= p < 128:
                        return midi_seq
                    v += vel_shift
                    v = max(1, min(127, v))
                    tokens_new[6] = self.parameter_ids["pitch"][p]
                    tokens_new[7] = self.parameter_ids["velocity"][v]
                elif name == "control_change":
                    cc = tokens[5] - self.parameter_ids["controller"][0]
                    val = tokens[6] - self.parameter_ids["value"][0]
                    if cc in [1, 2, 7, 11]:
                        val += cc_val_shift
                        val = max(1, min(127, val))
                    tokens_new[6] = self.parameter_ids["value"][val]
                elif name == "set_tempo":
                    bpm = tokens[4] - self.parameter_ids["bpm"][0]
                    bpm += bpm_shift
                    bpm = max(1, min(255, bpm))
                    tokens_new[4] = self.parameter_ids["bpm"][bpm]
            midi_seq_new.append(tokens_new)
        return midi_seq_new

    def check_alignment(self, midi_seq, threshold=0.3):
        total = 0
        hist = [0] * 16
        for tokens in midi_seq:
            if tokens[0] in self.id_events and self.id_events[tokens[0]] == "note":
                t2 = tokens[2] - self.parameter_ids["time2"][0]
                total += 1
                hist[t2] += 1
        if total == 0:
            return False
        hist = sorted(hist, reverse=True)
        p = sum(hist[:2]) / total
        return p > threshold
