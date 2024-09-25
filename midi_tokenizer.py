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

    def tokenize(self, midi_score, add_bos_eos=True, cc_eps=4, tempo_eps=4,
                 remap_track_channel=False, add_default_instr=False, remove_empty_channels=False):
        ticks_per_beat = midi_score[0]
        event_list = {}
        track_idx_map = {i: dict() for i in range(16)}
        track_idx_dict = {}
        channels = []
        patch_channels = []
        for track_idx, track in enumerate(midi_score[1:129]):
            last_notes = {}
            patch_dict = {}
            control_dict = {}
            last_tempo = 0
            for event in track:
                if event[0] not in self.events:
                    continue
                c = -1
                t = round(16 * event[1] / ticks_per_beat)  # quantization
                new_event = [event[0], t // 16, t % 16, track_idx] + event[2:]
                if event[0] == "note":
                    c = event[3]
                    track_idx_dict.setdefault(c, track_idx)
                    tr_map = track_idx_map[c]  # put track_idx_map when note event to avoid empty track
                    if track_idx not in tr_map:
                        tr_map[track_idx] = 0
                    new_event[4] = max(1, round(16 * new_event[4] / ticks_per_beat))
                elif event[0] == "set_tempo":
                    if new_event[4] == 0: # invalid tempo
                        continue
                    bpm = int(self.tempo2bpm(new_event[4]))
                    new_event[4] = min(bpm, 255)
                if event[0] == "note":
                    key = tuple(new_event[:4] + new_event[5:-1])
                else:
                    key = tuple(new_event[:-1])
                if event[0] == "patch_change":
                    c, p = event[2:]
                    last_p = patch_dict.setdefault(c, None)
                    if last_p == p:
                        continue
                    patch_dict[c] = p
                    if c not in patch_channels:
                        patch_channels.append(c)
                elif event[0] == "control_change":
                    c, cc, v = event[2:]
                    last_v = control_dict.setdefault((c, cc), 0)
                    if abs(last_v - v) < cc_eps:
                        continue
                    control_dict[(c, cc)] = v
                elif event[0] == "set_tempo":
                    tempo = new_event[-1]
                    if abs(last_tempo - tempo) < tempo_eps:
                        continue
                    last_tempo = tempo

                if c != -1 and c not in channels:
                    channels.append(c)

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

        empty_channels = [c for c, tr_map in track_idx_map.items() if len(tr_map) == 0]

        if remap_track_channel:
            patch_channels = []
            channels_count = 0
            channels_map = {9: 9} if 9 in channels else {}
            for c in channels:
                if c == 9:
                    continue
                channels_map[c] = channels_count
                channels_count += 1
                if channels_count == 9:
                    channels_count = 10
            channels = list(channels_map.values())

            track_count = 0
            mapped_channels_order = [k for k,v in sorted(list(channels_map.items()), key=lambda x: x[1])]
            for c in mapped_channels_order:
                tr_map = track_idx_map[c]
                for key in tr_map:
                    track_count += 1
                    tr_map[key] = track_count

            for event in event_list:
                name = event[0]
                track_idx = event[3]
                if name == "note":
                    c = event[5]
                    event[5] = channels_map[c]
                    event[3] = track_idx_map[c][track_idx]
                    track_idx_dict[event[5]] = event[3]
                elif name == "set_tempo":
                    event[3] = 0
                elif name == "control_change" or name == "patch_change":
                    c = event[4]
                    event[4] = channels_map[c]
                    tr_map = track_idx_map[c]
                    if len(tr_map) == 0: # empty channel
                        track_count += 1
                        tr_map[track_idx] = track_count
                    event[3] = next(iter(tr_map.values())) # move patch_change and control_change to first track of the channel
                    if event[4] not in patch_channels:
                        patch_channels.append(event[4])

        if add_default_instr:
            for c in channels:
                if c not in patch_channels:
                    event_list.append(["patch_change", 0,0, track_idx_dict[c], c, 0])

        event_list = sorted(event_list, key=lambda e: e[1:4])
        midi_seq = []
        setup_events = {}
        notes_in_setup = False
        for i, event in enumerate(event_list):  # optimise setup
            new_event = [*event]
            if event[0] != "note":
                new_event[1] = 0
                new_event[2] = 0
            has_next = False
            has_pre = False
            if i < len(event_list) - 1:
                next_event = event_list[i + 1]
                has_next = event[1] + event[2] == next_event[1] + next_event[2]
            if notes_in_setup and i > 0:
                pre_event = event_list[i - 1]
                has_pre = event[1] + event[2] == pre_event[1] + pre_event[2]
            if (event[0] == "note" and not has_next) or (notes_in_setup and not has_pre) :
                event_list = sorted(setup_events.values(), key=lambda e: 1 if e[0] == "note" else 0) + event_list[i:]
                break
            else:
                if event[0] == "note":
                    notes_in_setup = True
                key = tuple(event[3:-1])
            setup_events[key] = new_event

        last_t1 = 0
        for event in event_list:
            if remove_empty_channels and event[0] in ["control_change", "patch_change"] and event[4] in empty_channels:
                continue
            cur_t1 = event[1]
            event[1] = event[1] - last_t1
            tokens = self.event2tokens(event)
            if not tokens:
                continue
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
        if not all([0 <= params[i] < self.event_parameters[p] for i, p in enumerate(self.events[name])]):
            return []
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
                event = self.tokens2event(tokens)
                if not event:
                    continue
                name = event[0]
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
        tracks = [tr for idx, tr in sorted(list(tracks_dict.items()), key=lambda it: it[0])]

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
                max_track_shift=0, max_channel_shift=16):
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

    def check_quality(self, midi_seq, alignment_min=0.4, tonality_min=0.8, piano_max=0.7, notes_bandwidth_min=3, notes_density_max=30, notes_density_min=2.5, total_notes_max=10000, total_notes_min=500, note_window_size=16):
        total_notes = 0
        channels = []
        time_hist = [0] * 16
        note_windows = {}
        notes_sametime = []
        notes_density_list = []
        tonality_list = []
        notes_bandwidth_list = []
        instruments = {}
        piano_channels = []
        undef_instrument = False
        abs_t1 = 0
        last_t = 0
        for tsi, tokens in enumerate(midi_seq):
            event = self.tokens2event(tokens)
            if not event:
                continue
            t1, t2, tr = event[1:4]
            abs_t1 += t1
            t = abs_t1 * 16 + t2
            c = None
            if event[0] == "note":
                d, c, p, v = event[4:]
                total_notes += 1
                time_hist[t2] += 1
                if c != 9:  # ignore drum channel
                    if c not in instruments:
                        undef_instrument = True
                    note_windows.setdefault(abs_t1 // note_window_size, []).append(p)
                if last_t != t:
                    notes_sametime = [(et, p_) for et, p_ in notes_sametime if et > last_t]
                    notes_sametime_p = [p_ for _, p_ in notes_sametime]
                    if len(notes_sametime) > 0:
                        notes_bandwidth_list.append(max(notes_sametime_p) - min(notes_sametime_p))
                notes_sametime.append((t + d - 1, p))
            elif event[0] == "patch_change":
                c, p = event[4:]
                instruments[c] = p
                if p == 0 and c not in piano_channels:
                    piano_channels.append(c)
            if c is not None and c not in channels:
                channels.append(c)
            last_t = t
        reasons = []
        if total_notes < total_notes_min:
            reasons.append("total_min")
        if total_notes > total_notes_max:
            reasons.append("total_max")
        if undef_instrument:
            reasons.append("undef_instr")
        if len(note_windows) == 0 and total_notes > 0:
            reasons.append("drum_only")
        if reasons:
            return False, reasons
        time_hist = sorted(time_hist, reverse=True)
        alignment = sum(time_hist[:2]) / total_notes
        for notes in note_windows.values():
            key_hist = [0] * 12
            for p in notes:
                key_hist[p % 12] += 1
            key_hist = sorted(key_hist, reverse=True)
            tonality_list.append(sum(key_hist[:7]) / len(notes))
            notes_density_list.append(len(notes) / note_window_size)
        tonality_list = sorted(tonality_list)
        tonality = sum(tonality_list)/len(tonality_list)
        notes_bandwidth = sum(notes_bandwidth_list)/len(notes_bandwidth_list) if notes_bandwidth_list else 0
        notes_density = max(notes_density_list) if notes_density_list else 0
        piano_ratio = len(piano_channels) / len(channels)
        if len(channels) <=3:  # ignore piano threshold if it is a piano solo midi
            piano_max = 1
        if alignment < alignment_min: # check weather the notes align to the bars (because some midi files are recorded)
            reasons.append("alignment")
        if tonality < tonality_min:  # check whether the music is tonal
            reasons.append("tonality")
        if notes_bandwidth < notes_bandwidth_min:  # check whether music is melodic line only
            reasons.append("bandwidth")
        if not notes_density_min < notes_density < notes_density_max:
            reasons.append("density")
        if piano_ratio > piano_max: # check whether most instruments is piano (because some midi files don't have instruments assigned correctly)
            reasons.append("piano")
        return not reasons, reasons
