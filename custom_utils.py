from collections import UserDict
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple, Union

import fire
import torch
import torch.nn as nn
import transformers
from transformers import GenerationConfig, LlamaForCausalLM
from transformers import StoppingCriteriaList
from transformers.generation.utils import GenerationMixin
import torch.nn.functional as F
from transformers.generation import BeamScorer, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateBeamOutput, _split_model_inputs, stack_model_outputs, GenerateBeamEncoderDecoderOutput, GenerateBeamDecoderOnlyOutput
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.utils import ModelOutput
from transformers.cache_utils import DynamicCache

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Run selection beam", '#'*10)

class CustomGenerationConfig(GenerationConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminator_ids = kwargs.pop("terminator_ids", None)

@dataclass
class CustomGenerateBeamOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    best_cache: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None

def PatchProcess(
        self,
        original_process,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[Union[int, torch.Tensor]] = None,
        eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
        decoder_prompt_len: Optional[int] = 0,
        candidate_cache = None,
        cache = None,
        beam_length = None,
        num_steps = 1,
        terminator_ids = None,
) -> Dict[str, torch.Tensor]:
    # add up to the length which the next_scores is calculated on (including decoder prompt)
    cur_len = input_ids.shape[-1] + 1
    batch_size = len(self._beam_hyps) // self.num_beam_groups

    if not (batch_size == (input_ids.shape[0] // self.group_size)):
        if self.num_beam_groups > 1:
            raise ValueError(
                f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                f"size of {self.group_size} is expected by the beam scorer."
            )
        else:
            raise ValueError(
                f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                f"{self.group_size} is expected by the beam scorer."
            )

    device = input_ids.device
    next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
    next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
    next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)
    next_beam_length = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

    if eos_token_id is not None and not isinstance(eos_token_id, torch.Tensor):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id = torch.tensor(eos_token_id)

    for batch_idx in range(batch_size):
        batch_group_idx = batch_idx * self.num_beam_groups + group_index
        if self._done[batch_group_idx]:
            if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
            if eos_token_id is None or pad_token_id is None:
                raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
            # pad the batch
            next_beam_scores[batch_idx, :] = 0
            next_beam_tokens[batch_idx, :] = pad_token_id
            next_beam_indices[batch_idx, :] = 0
            continue

        # next tokens for this sentence
        beam_idx = 0
        for beam_token_rank, (next_token, next_score, next_index) in enumerate(
            zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
        ):
            batch_beam_idx = batch_idx * self.group_size + next_index
            # add to generated hypotheses if end of sentence
            if (terminator_ids is not None) and (next_token.item() in terminator_ids) and (num_steps == 1 or beam_length[batch_beam_idx] + 1 == num_steps):
                # if beam_token does not belong to top num_beams tokens, it should not be added
                is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                if is_beam_token_worse_than_top_num_beams:
                    continue
                if beam_indices is not None:
                    beam_index = beam_indices[batch_beam_idx]
                    beam_index = beam_index + (batch_beam_idx,)
                else:
                    beam_index = None

                worst_score = self._beam_hyps[batch_group_idx].worst_score
                self._beam_hyps[batch_group_idx].add(
                    torch.cat((input_ids[batch_beam_idx].clone(), next_token.unsqueeze(0)), dim=-1),
                    # torch.cat((input_ids[batch_beam_idx].clone(), torch.tensor([627], device=next_token.device)), dim=-1),
                    next_score.item(),
                    beam_indices=beam_index,
                    generated_len=cur_len - decoder_prompt_len,
                )
                # ended_batch_beam_indices[batch_idx].append(batch_beam_idx)
                generated_len = cur_len - decoder_prompt_len
                if generated_len is not None:
                    score = next_score.item() / (generated_len ** self._beam_hyps[batch_group_idx].length_penalty)
                else:
                    score = next_score.item() / (input_ids[batch_beam_idx].shape[-1] ** self._beam_hyps[batch_group_idx].length_penalty)
                if len(candidate_cache[batch_group_idx]) < self._beam_hyps[batch_group_idx].num_beams or score > worst_score:
                    ended_cache = [(cache[i][0][batch_beam_idx, ...], cache[i][1][batch_beam_idx, ...]) for i in range(len(cache))]
                    candidate_cache[batch_group_idx].append((score, ended_cache))
                    if len(candidate_cache[batch_group_idx]) > self._beam_hyps[batch_group_idx].num_beams:
                        sorted_cache = sorted([(s, idx) for idx, (s, c) in enumerate(candidate_cache[batch_group_idx])])
                        del candidate_cache[batch_group_idx][sorted_cache[0][1]]

            else:
                # add next predicted token since it is not eos_token
                next_beam_scores[batch_idx, beam_idx] = next_score
                next_beam_tokens[batch_idx, beam_idx] = next_token
                next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                if (terminator_ids is not None) and (next_token.item() in terminator_ids):
                    next_beam_length[batch_idx, beam_idx]  = beam_length[batch_beam_idx] + 1
                beam_idx += 1

            # once the beam for next step is full, don't add more tokens to it.
            if beam_idx == self.group_size:
                break

        if beam_idx < self.group_size:
            raise ValueError(
                f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
            )

        # Check if we are done so that we can save a pad step if all(done)
        self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
            next_scores[batch_idx].max().item(), cur_len, decoder_prompt_len
        )

    return UserDict(
        {
            "next_beam_scores": next_beam_scores.view(-1),
            "next_beam_tokens": next_beam_tokens.view(-1),
            "next_beam_indices": next_beam_indices.view(-1),
            "next_beam_length": next_beam_length.view(-1),
        }
    )

def PatchFinalize(
    self,
    input_ids: torch.LongTensor,
    final_beam_scores: torch.FloatTensor,
    final_beam_tokens: torch.LongTensor,
    final_beam_indices: torch.LongTensor,
    max_length: int,
    pad_token_id: Optional[Union[int, torch.Tensor]] = None,
    eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None,
    beam_indices: Optional[torch.LongTensor] = None,
    decoder_prompt_len: Optional[int] = 0,
    candidate_cache = None,
    cache = None,
) -> Tuple[torch.LongTensor]:
    batch_size = len(self._beam_hyps) // self.num_beam_groups

    if eos_token_id is not None and not isinstance(eos_token_id, torch.Tensor):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id = torch.tensor(eos_token_id)

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
        if self._done[batch_group_idx]:
            continue

        # all open beam hypotheses are added to the beam hypothesis
        # beam hypothesis class automatically keeps the best beams
        for index_per_group in range(self.group_size):
            batch_beam_idx = batch_group_idx * self.group_size + index_per_group
            final_score = final_beam_scores[batch_beam_idx].item()
            final_tokens = input_ids[batch_beam_idx]
            beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
            generated_len = final_tokens.shape[-1] - decoder_prompt_len
            worst_score = beam_hyp.worst_score
            beam_hyp.add(final_tokens, final_score, beam_indices=beam_index, generated_len=generated_len)

            if generated_len is not None:
                score = final_score / (generated_len ** beam_hyp.length_penalty)
            else:
                score = final_score / (final_tokens.shape[-1] ** beam_hyp.length_penalty)
            if len(candidate_cache[batch_group_idx]) < beam_hyp.num_beams or score > worst_score:
                ended_cache = [(cache[i][0][batch_beam_idx, ...], cache[i][1][batch_beam_idx, ...]) for i in range(len(cache))]
                candidate_cache[batch_group_idx].append((score, ended_cache))
                if len(candidate_cache[batch_group_idx]) > beam_hyp.num_beams:
                    sorted_cache = sorted([(s, idx) for idx, (s, c) in enumerate(candidate_cache[batch_group_idx])])
                    del candidate_cache[batch_group_idx][sorted_cache[0][1]]

    # select the best hypotheses
    sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
    best = []
    best_indices = []
    best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

    # retrieve best hypotheses
    for i in range(batch_size):
        beam_hyps_in_batch = self._beam_hyps[i * self.num_beam_groups : (i + 1) * self.num_beam_groups]
        candidate_beams = [beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams]
        sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
        for j in range(self.num_beam_hyps_to_keep):
            best_hyp_tuple = sorted_hyps.pop()
            best_score = best_hyp_tuple[0]
            best_hyp = best_hyp_tuple[1]
            best_index = best_hyp_tuple[2]
            sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

            # append hyp to lists
            best.append(best_hyp)

            # append indices to list
            best_indices.append(best_index)

            best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

    # prepare for adding eos
    sent_lengths_max = sent_lengths.max().item()
    sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
    decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)

    if len(best_indices) > 0 and best_indices[0] is not None:
        indices: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
    else:
        indices = None

    # shorter batches are padded if needed
    if sent_lengths.min().item() != sent_lengths.max().item():
        if pad_token_id is None:
            raise ValueError("`pad_token_id` has to be defined")
        decoded.fill_(pad_token_id)

    if indices is not None:
        indices.fill_(-1)

    # fill with hypotheses and eos_token_id if the latter fits in
    for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
        decoded[i, : sent_lengths[i]] = hypo

        if indices is not None:
            indices[i, : len(best_idx)] = torch.tensor(best_idx)

        # if sent_lengths[i] < sent_max_len:
        #     # inserting only the first eos_token_id
        #     decoded[i, sent_lengths[i]] = eos_token_id[0]

    return UserDict(
        {
            "sequences": decoded,
            "sequence_scores": best_scores,
            "beam_indices": indices,
        }
    )


def prepare_input(input_ids, cache, beam_idx, sample_size, startends=None, special_token_id=None):
    '''
        Prepare input for next step
    '''
    assert len(input_ids) == len(cache)
    dtype = input_ids[0].dtype
    device = input_ids[0].device
    # print(len(input_ids), len(cache), beam_idx)
    selected_input_ids = [input_ids[i] for i in beam_idx]
    selected_cache = [cache[i] for i in beam_idx]

    if startends is not None and special_token_id is not None:
        selected_startends = [startends[i] for i in beam_idx]
        special_token_input_ids = []
        for i in range(len(selected_input_ids)):
            seq_ids = selected_input_ids[i]
            processed_seq_ids = seq_ids.clone()
            for j, (start, end) in enumerate(selected_startends[i]):
                processed_seq_ids = torch.cat([
                    processed_seq_ids[:start],
                    torch.tensor([special_token_id], device=seq_ids.device),
                    processed_seq_ids[start:end],
                    torch.tensor([special_token_id], device=seq_ids.device),
                    processed_seq_ids[end:]
                ])
            special_token_input_ids.append(processed_seq_ids)
        selected_input_ids = special_token_input_ids

    """Left padding for input_ids"""
    real_pos = [(0, 0) for _ in range(len(selected_input_ids))]
    for i, ids in enumerate(selected_input_ids):
        start, end = None, None
        for j, id in enumerate(ids):
            if id == 0:
                continue
            start = j
            break
        for j in range(len(ids)-1, -1, -1):
            if ids[j] == 0:
                continue
            end = j+1
            break
        real_pos[i] = (start, end)
    max_len = max([x[1]-x[0] for x in real_pos])
    next_input_ids = torch.zeros(len(selected_input_ids), max_len, dtype=dtype, device=device)
    for i, (start, end) in enumerate(real_pos):
        next_input_ids[i, max_len-(end-start):] = selected_input_ids[i][start:end]

    """Prepare attention mask"""
    next_attention_mask = torch.zeros(len(selected_input_ids), max_len, dtype=dtype, device=device)
    for i, (start, end) in enumerate(real_pos):
        next_attention_mask[i, max_len-(end-start):] = 1

    if startends is not None and special_token_id is not None:
        return next_input_ids, None, next_attention_mask

    """Check cache shape"""
    for i in range(len(selected_cache)):
        ids_len = real_pos[i][1]
        cache_len = selected_cache[i][0][0].shape[-2]
        # assert ids_len == cache_len + 1, f"[ERROR] real_ids_len: {ids_len}, cache_len: {cache_len}"

    """Prepare cache"""
    # origin cache has shape List[List[Tuple[torch.Tensor, torch.Tensor]]] [Batch_size * num_beams, n_layers, 2, num_heads, seq_len, head_dim]
    cache_dtype = cache[0][0][0].dtype
    cache_device = cache[0][0][0].device
    selected_batch_size = len(selected_cache)
    n_layers = len(cache[0])
    n_heads = cache[0][0][0].shape[0]
    head_dim = cache[0][0][0].shape[-1]
    real_cache_pos = [(x[0], x[1]-1) for x in real_pos]
    # turn to shape Tuple[Tuple[torch.Tensor, torch.Tensor]] [n_layers, 2, slected_batch_size, num_heads, seq_len, head_dim]
    cache_max_len = max_len - 1 # cache len is always one less than input_ids
    next_cache = tuple((torch.zeros(selected_batch_size, n_heads, cache_max_len, head_dim, device=cache_device, dtype=cache_dtype), torch.zeros(selected_batch_size, n_heads, cache_max_len, head_dim, device=cache_device, dtype=cache_dtype)) for _ in range(n_layers))

    for i in range(n_layers):
        for j, (start, end) in enumerate(real_cache_pos):
            next_cache[i][0][j, :, cache_max_len-(end-start):, :] = selected_cache[j][i][0][:, start:end, :]
            next_cache[i][1][j, :, cache_max_len-(end-start):, :] = selected_cache[j][i][1][:, start:end, :]
    next_cache = DynamicCache.from_legacy_cache(next_cache)
    next_cache.batch_repeat_interleave(sample_size)

    return next_input_ids, next_cache, next_attention_mask

class CustomModelForCausalLM(LlamaForCausalLM, GenerationMixin):
    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        logits_warper: Optional[LogitsProcessorList] = None,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        # print("Custom beam search called")
        # init values
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample
        terminator_ids = generation_config.terminator_ids
        num_steps = 1
        # print("Num steps", num_steps)
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        candidate_cache = [[] for _ in range(batch_size)]
        beam_length = None

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                    model_name in self.__class__.__name__.lower()
                    for model_name in [
                        "fsmt",
                        "reformer",
                        "bloom",
                        "ctrl",
                        "gpt_bigcode",
                        "transo_xl",
                        "xlnet",
                        "cpm",
                        "jamba",
                    ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
                )
                outputs_per_sub_batch = [
                    self(
                        **inputs_per_sub_batch,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch)

            else:  # Unchanged original behavior
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            if do_sample:
                next_token_scores_processed = logits_warper(input_ids, next_token_scores_processed)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
            # non eos token per beam.
            # n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            n_eos_tokens = len(terminator_ids)
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            original_process = BeamSearchScorer.process
            beam_outputs = PatchProcess(
                beam_scorer,
                original_process,
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
                candidate_cache=candidate_cache,
                cache=model_kwargs["past_key_values"],
                beam_length=beam_length,
                num_steps=num_steps,
                terminator_ids=terminator_ids,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            beam_length = beam_outputs["next_beam_length"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = PatchFinalize(
            beam_scorer,
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
            candidate_cache=candidate_cache,
            cache=model_kwargs["past_key_values"],
        )

        # reorder candidate cache to accord with the final sequence order
        best = [] # List[List[Tuple[torch.Tensor, torch.Tensor]]] [Batch_size * num_beams, n_layers, 2, num_heads, seq_len, head_dim]
        for i in range(batch_size):
            sorted_cache = sorted(candidate_cache[i], key=lambda x: x[0])
            for j in range(beam_scorer.num_beam_hyps_to_keep):
                best_cache = sorted_cache.pop()
                best.append(best_cache[1])

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return CustomGenerateBeamOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    best_cache=best,
                )
        else:
            return sequence_outputs["sequences"]  