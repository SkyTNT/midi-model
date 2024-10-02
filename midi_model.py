from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import lightning  as pl
from transformers import LlamaModel, LlamaConfig

from midi_tokenizer import MIDITokenizerV1, MIDITokenizerV2, MIDITokenizer

config_name_list = ["tv1-medium", "tv2-medium", "tv2o-medium", "tv2-large", "tv2o-large"]


class MIDIModelConfig:
    def __init__(self, tokenizer: Union[MIDITokenizerV1, MIDITokenizerV2],
                 net_config: LlamaConfig, net_token_config: LlamaConfig):
        self.tokenizer = tokenizer
        self.net_config = net_config
        self.net_token_config = net_token_config
        self.n_embd = net_token_config.hidden_size

    @staticmethod
    def get_config(tokenizer_ver="v2", optimise_midi=True, n_layer=12, n_head=16, n_embd=1024, n_inner=4096):
        tokenizer = MIDITokenizer(tokenizer_ver)
        tokenizer.set_optimise_midi(optimise_midi)
        net_config = LlamaConfig(vocab_size=tokenizer.vocab_size,
                                 hidden_size=n_embd, num_attention_heads=n_head,
                                 num_hidden_layers=n_layer, intermediate_size=n_inner,
                                 pad_token_id=tokenizer.pad_id, max_position_embeddings=4096)
        net_token_config = LlamaConfig(vocab_size=tokenizer.vocab_size,
                                       hidden_size=n_embd, num_attention_heads=n_head // 4,
                                       num_hidden_layers=n_layer // 4, intermediate_size=n_inner // 4,
                                       pad_token_id=tokenizer.pad_id, max_position_embeddings=4096)
        return MIDIModelConfig(tokenizer, net_config, net_token_config)

    @staticmethod
    def from_name(name="tv2o-medium"):
        tv, size = name.split("-")
        tv = tv[1:]
        if tv[-1] == "o":
            o = True
            tv = tv[:-1]
        else:
            o = False
        if tv not in ["v1", "v2"]:
            raise ValueError(f"Unknown tokenizer version {tv}")
        if size == "medium":
            return MIDIModelConfig.get_config(tokenizer_ver=tv, optimise_midi=o,
                                              n_layer=12, n_head=16, n_embd=1024, n_inner=4096)
        elif size == "large":
            return MIDIModelConfig.get_config(tokenizer_ver=tv, optimise_midi=o,
                                              n_layer=24, n_head=16, n_embd=1024, n_inner=4096)
        else:
            raise ValueError(f"Unknown model size {size}")


class MIDIModel(pl.LightningModule):
    def __init__(self, config: MIDIModelConfig, flash=False, *args, **kwargs):
        super(MIDIModel, self).__init__()
        if flash:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
        self.tokenizer = config.tokenizer
        self.net = LlamaModel(config.net_config)
        self.net_token = LlamaModel(config.net_token_config)
        self.lm_head = nn.Linear(config.n_embd, self.tokenizer.vocab_size, bias=False)

    def forward_token(self, hidden_state, x=None):
        """

        :param hidden_state: (batch_size, n_embd)
        :param x: (batch_size, token_sequence_length)
        :return: (batch_size, 1 + token_sequence_length, vocab_size)
        """
        hidden_state = hidden_state.unsqueeze(1)  # (batch_size, 1, n_embd)
        if x is not None:
            x = self.net_token.embed_tokens(x)
            hidden_state = torch.cat([hidden_state, x], dim=1)
        hidden_state = self.net_token.forward(inputs_embeds=hidden_state).last_hidden_state
        return self.lm_head(hidden_state)

    def forward(self, x):
        """
        :param x: (batch_size, midi_sequence_length, token_sequence_length)
        :return: hidden (batch_size, midi_sequence_length, n_embd)
        """

        # merge token sequence
        x = self.net.embed_tokens(x)
        x = x.sum(dim=-2)
        x = self.net.forward(inputs_embeds=x)
        return x.last_hidden_state

    def sample_top_p_k(self, probs, p, k, generator=None):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        mask = torch.zeros(probs_sort.shape[-1], device=probs_sort.device)
        mask[:k] = 1
        probs_sort = probs_sort * mask
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        shape = probs_sort.shape
        next_token = torch.multinomial(probs_sort.reshape(-1, shape[-1]),
                                       num_samples=1, generator=generator).reshape(*shape[:-1], 1)
        next_token = torch.gather(probs_idx, -1, next_token).reshape(*shape[:-1])
        return next_token

    @torch.inference_mode()
    def generate(self, prompt=None, batch_size=1, max_len=512, temp=1.0, top_p=0.98, top_k=20, generator=None):
        tokenizer = self.tokenizer
        max_token_seq = tokenizer.max_token_seq
        if prompt is None:
            input_tensor = torch.full((1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=self.device)
            input_tensor[0, 0] = tokenizer.bos_id  # bos
        else:
            prompt = prompt[:, :max_token_seq]
            if prompt.shape[-1] < max_token_seq:
                prompt = np.pad(prompt, ((0, 0), (0, max_token_seq - prompt.shape[-1])),
                                mode="constant", constant_values=tokenizer.pad_id)
            input_tensor = torch.from_numpy(prompt).to(dtype=torch.long, device=self.device)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = torch.cat([input_tensor]*batch_size, dim=0)
        cur_len = input_tensor.shape[1]
        bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
        with bar:
            while cur_len < max_len:
                end = [False] * batch_size
                hidden = self.forward(input_tensor)[:, -1]
                next_token_seq = None
                event_names = [""] * batch_size
                for i in range(max_token_seq):
                    mask = torch.zeros((batch_size, tokenizer.vocab_size), dtype=torch.int64, device=self.device)
                    for b in range(batch_size):
                        if end[b]:
                            mask[b, tokenizer.pad_id] = 1
                            continue
                        if i == 0:
                            mask[b, list(tokenizer.event_ids.values()) + [tokenizer.eos_id]] = 1
                        else:
                            param_names = tokenizer.events[event_names[b]]
                            if i > len(param_names):
                                mask[b, tokenizer.pad_id] = 1
                                continue
                            mask[b, tokenizer.parameter_ids[param_names[i - 1]]] = 1
                    mask = mask.unsqueeze(1)
                    logits = self.forward_token(hidden, next_token_seq)[:, -1:]
                    scores = torch.softmax(logits / temp, dim=-1) * mask
                    samples = self.sample_top_p_k(scores, top_p, top_k, generator=generator)
                    if i == 0:
                        next_token_seq = samples
                        for b in range(batch_size):
                            if end[b]:
                                continue
                            eid = samples[b].item()
                            if eid == tokenizer.eos_id:
                                end[b] = True
                            else:
                                event_names[b] = tokenizer.id_events[eid]
                    else:
                        next_token_seq = torch.cat([next_token_seq, samples], dim=1)
                        if all([len(tokenizer.events[event_names[b]]) == i for b in range(batch_size) if not end[b]]):
                            break

                if next_token_seq.shape[1] < max_token_seq:
                    next_token_seq = F.pad(next_token_seq, (0, max_token_seq - next_token_seq.shape[1]),
                                           "constant", value=tokenizer.pad_id)
                next_token_seq = next_token_seq.unsqueeze(1)
                input_tensor = torch.cat([input_tensor, next_token_seq],
                                         dim=1)
                cur_len += 1
                bar.update(1)

                if all(end):
                    break
        return input_tensor.cpu().numpy()
