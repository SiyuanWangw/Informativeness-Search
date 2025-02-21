import os
import sys
import time
import re
import numpy as np
import random
from tqdm import tqdm
import math
from utils.select_steps import *
import json
from collections import Counter
import fire
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F
from transformers import BitsAndBytesConfig
from utils.custom_utils import CustomModelForCausalLM, prepare_input, CustomGenerationConfig
from transformers.cache_utils import DynamicCache
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
option_ids = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def main(
    load_8bit: bool = False,
    load_4bit: bool = False,
    base_model: str = "",
    data_path: str = "",
    baseline: bool = False,
    batch_size: int = 1,
    beam_size: int = 3,
    sample_size: int = 2,
    run_log="",
    is_selfselect=False,
    premise_info_thres = 0.7,
    start_step = 3, #4
    conc_threshold = 0.3,
    output_file = ""
):
    output_file = output_file.replace(".log", ".json")
    print(run_log)
    # =================== Load model ===================
    """assure model config according to model_name"""
    if "llama" and "8b" in base_model.lower():
        layer_num = 32
        head_num = 32
        start_step = 3
    elif "llama" and "3b" in base_model.lower():
        layer_num = 28
        head_num = 24
        start_step = 4
    elif 'phi' in base_model.lower():
        layer_num = 40
        head_num = 40

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    model = CustomModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    print(model.__class__)

    if "llama" in base_model.lower():
        model.config.pad_token_id = tokenizer.pad_token_id = 0  
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    start_time = time.perf_counter()

    def insert_tokens(inputs, inserted_tokens):
        '''
            Insert tokens after <|assistant|>
        '''
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        # Add leading tokens to the input_ids
        ids_list = []
        mask_list = []
        for i in range(input_ids.shape[0]):
            if inserted_tokens is None or inserted_tokens[i] == "":
                ids_list.append(input_ids[i])
                if attention_mask is not None:
                    mask_list.append(attention_mask[i])
            else:
                inserted_ids = torch.tensor(tokenizer(inserted_tokens[i], add_special_tokens=False).input_ids)
                        
                inserted_ids = inserted_ids.to(input_ids.device)
                ids_list.append(torch.cat([input_ids[i], inserted_ids], dim=-1))
                if attention_mask is not None:
                    insert_attention_mask = torch.ones(inserted_ids.shape[0], device=input_ids.device)
                    mask_list.append(torch.cat([attention_mask[i], insert_attention_mask], dim=-1))
        # Left padding
        max_len = max([len(ids) for ids in ids_list])
        for i in range(len(ids_list)):
            ids_list[i] = torch.cat([torch.zeros(max_len-len(ids_list[i]), device=input_ids.device, dtype=ids_list[i].dtype), ids_list[i]], dim=-1)
            if attention_mask is not None:
                mask_list[i] = torch.cat([torch.zeros(max_len-len(mask_list[i]), device=input_ids.device, dtype=ids_list[i].dtype), mask_list[i]], dim=-1)
        inputs['input_ids'] = torch.stack(ids_list)
        if attention_mask is not None:
            inputs['attention_mask'] = torch.stack(mask_list)
        return inputs

    def generate_one_step(input_ids, input_attn_mask, input_cache, max_new_tokens, num_beam, sample_size, terminators):
        '''
            Generate one step for beam search
        '''
        generation_config = CustomGenerationConfig(
            temperature=0,
            pad_token_id=0,
            return_dict=True,
            early_stopping=False,
            terminator_ids=terminators,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=input_attn_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                output_attentions=(not baseline),
                output_hidden_states=False,
                num_beams=num_beam,
                num_return_sequences=sample_size,
                past_key_values=input_cache,
                use_cache=True,
            )
        seq_ids = generation_output.sequences
        seq_scores = generation_output.sequences_scores
        beam_indices = generation_output.beam_indices
        best_cache = generation_output.best_cache
        if baseline:
            del generation_output, input_ids, input_attn_mask
            torch.cuda.empty_cache()
            return seq_ids, seq_scores, None, best_cache
        output_attn = generation_output.attentions
        input_len = input_ids.shape[-1]
        del generation_output, input_ids, input_attn_mask
        torch.cuda.empty_cache()
        first_token_score = [output_attn[0][i] for i in range(layer_num)]
        whole_size = first_token_score[0].shape[0]
        dtype = first_token_score[0].dtype
        output_len = len(output_attn) - 1
        seq_len = output_len + input_len
        attn_scores = torch.zeros(layer_num, whole_size, head_num, output_len, seq_len, dtype=dtype, device=first_token_score[0].device)
        for i in range(1, len(output_attn)):
            for j in range(layer_num):
                attn_scores[j, :, :, i-1:i, :i+input_len] = output_attn[i][j]
        del output_attn, first_token_score
        torch.cuda.empty_cache()
        for i in range(0, output_len):
            attn_scores[:, :, :, i, :] = attn_scores[:, beam_indices[:, i+1], :, i, :]
        return seq_ids, seq_scores, attn_scores, best_cache


    def parse_output(output, input_len, batch_size, terminator, num_hype_to_keep, sample_size, total_cost=0):
        '''
            Parse output
        '''
        raw_seq_ids, seq_scores = output
        raw_seq_tokens = [tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in raw_seq_ids]
        raw_output_tokens = [raw_seq_tokens[i][input_len:] for i in range(len(raw_seq_tokens))]
        batch_pos = [0]
        for i in range(batch_size):
            batch_pos.append(batch_pos[-1]+num_hype_to_keep[i]*sample_size)
        batched_seq_scores = []
        batched_output_tokens = []
        batched_seq_ids = []
        for i in range(batch_size):
            batched_seq_scores.append(seq_scores[batch_pos[i]:batch_pos[i+1]])
            batched_output_tokens.append(raw_output_tokens[batch_pos[i]:batch_pos[i+1]])
            batched_seq_ids.append(raw_seq_ids[batch_pos[i]:batch_pos[i+1]])
        del seq_scores
        torch.cuda.empty_cache()
        no_pad_output_tokens = [[] for _ in range(batch_size)]
        step_len = [[] for _ in range(batch_size)]
        for i, one_batch in enumerate(batched_output_tokens):
            for j, tokens in enumerate(one_batch):
                found = False
                for k, token in enumerate(tokens):
                    if terminator in token:
                        no_pad_output_tokens[i].append(tokens[:k+1])
                        step_len[i].append(k+1)
                        found = True
                        break
                if not found:
                    no_pad_output_tokens[i].append(tokens)
                    step_len[i].append(len(tokens))
                    
        raw_output_seq = [[tokenizer.convert_tokens_to_string(x) for x in tokens] for tokens in no_pad_output_tokens]
        whole_seq = [tokenizer.batch_decode(batched_seq_ids[i]) for i in range(batch_size)]
        whole_seq = [[y.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip() for y in x] for x in whole_seq]
        finish = []
        for seq in whole_seq:
            seq_finish = []
            for x in seq: 
                total_cost += len(tokenizer(x)[0]) + 2   
                last_seq = x.split("\n")[-1]    
                if "So the answer is" in last_seq and any(char.isalnum() for char in last_seq.split("So the answer is")[-1].strip()):
                    cur_finish = True
                else:
                    cur_finish = False
                seq_finish.append(cur_finish)
            finish.append(seq_finish)
        return raw_output_seq, batched_seq_scores, step_len, finish, raw_seq_tokens, total_cost

    def compute_scores_step_startend(output_attn, cur_step_len, step_startends, batch_size, num_hype_to_keep, sample_size, beam_size, response_start):
        '''
            Compute attention scores

            |<---last_step_len--->|input_len|<---cur_step_len--->|
        '''
        if output_attn is None:
            return None
        start = min(response_start)
        attn_scores = output_attn[:,:,:,:,start:]
        useful_mask = torch.zeros_like(attn_scores)
        batch_id = 0
        case_id = 0
        info_steps_len_list = []
        for i in range(attn_scores.shape[1]):
            if num_hype_to_keep[batch_id] == 0:
                batch_id += 1
                case_id = 0
            info_steps_len = 0
            for j in range(len(step_startends[case_id+batch_id*(sample_size*beam_size)])):
                useful_mask[:, i, :, 0:cur_step_len[batch_id][case_id], step_startends[case_id+batch_id*(sample_size*beam_size)][j][0]-start:step_startends[case_id+batch_id*(sample_size*beam_size)][j][1]-start+1] = 1
                info_steps_len += step_startends[case_id+batch_id*(sample_size*beam_size)][j][1] - step_startends[case_id+batch_id*(sample_size*beam_size)][j][0]+1
            info_steps_len = max(1, info_steps_len)
            info_steps_len_list.append(info_steps_len)
            case_id += 1
            if case_id == num_hype_to_keep[batch_id]*sample_size:
                batch_id += 1
                case_id = 0
        useful_scores = attn_scores * useful_mask
        useful_scores = useful_scores.sum(dim=-2)

        batch_id = 0
        case_id = 0
        for i in range(attn_scores.shape[1]):
            if num_hype_to_keep[batch_id] == 0:
                batch_id += 1
                case_id = 0
            useful_scores[:, i] = useful_scores[:, i] / cur_step_len[batch_id][case_id]
            case_id += 1
            if case_id == num_hype_to_keep[batch_id]*sample_size:
                batch_id += 1
                case_id = 0
        useful_scores = torch.transpose(useful_scores, 0, 1)
        useful_scores = useful_scores.reshape(useful_scores.size(0), -1)
        useful_scores = torch.topk(useful_scores, 50, dim=-1)[0]
        useful_scores = useful_scores.mean(dim=-1)
        useful_scores = useful_scores.reshape(batch_size, -1)
        del attn_scores, useful_mask
        torch.cuda.empty_cache()
        return useful_scores


    def select_best_k(seq_scores, attn_scores, num_hype_to_keep, batch_size):
        '''
            Select the best k
        '''
        print(seq_scores, attn_scores)
        if attn_scores is None:
            total_best_beam_idx = []
            for i in range(batch_size):
                _, best_beam_idx = torch.topk(seq_scores[i], num_hype_to_keep[i], dim=-1)
                best_beam_idx = best_beam_idx.tolist()
                total_best_beam_idx.append(best_beam_idx)
            return total_best_beam_idx
        else:
            total_best_beam_idx = []
            for i in range(batch_size):
                score = seq_scores[i] + 2 * attn_scores[i]
                _, best_beam_idx = torch.topk(score, num_hype_to_keep[i], dim=-1)
                best_beam_idx = best_beam_idx.tolist()
                total_best_beam_idx.append(best_beam_idx)
            return total_best_beam_idx      

    def generate_beam_search(
        prompt,
        questions,
        options,
        batch_size,
        beam_size,
        sample_size,
        terminator,
        start_step=3,
        thres = 0.7,
        conc_threshold=0.3,
    ):
        '''
            Step-wise beam search
        '''
        # Create terminators
        terminators = [
            token_id for token, token_id in tokenizer.vocab.items()
            if terminator in token
        ]
        terminators.append(tokenizer.eos_token_id)
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        terminators = list(set(terminators))
        terminator_id = tokenizer.convert_tokens_to_ids(terminator)
        if terminator_id in terminators:
            terminators.remove(terminator_id)
            terminators.insert(0, terminator_id)

        if is_selfselect:
            max_new_tokens = 300
        else:
            max_new_tokens = 100
        all_finished = False
        n_steps = 0
        candidates = [[] for _ in range(batch_size)]
        num_hype_to_keep = [beam_size for _ in range(batch_size)]
        batch_finish = [False for _ in range(batch_size)]
        remain_scores = None
        remain_attn_scores = None

        """Prepare inputs for the first step"""
        texts = []
        for each_ques, each_op in zip(questions, options):
            messages = [
                {"role": "user", "content": prompt.format(question=each_ques, option=each_op)},
            ]
            encodeds = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(encodeds)
        input_tensors = tokenizer(texts, padding=True, return_tensors='pt').to(device)
        input_tensors = insert_tokens(input_tensors, ["" for _ in range(batch_size)])

        input_ids = input_tensors['input_ids']
        total_cost = 0
        original_input_length = input_ids.shape[-1]
        input_attn_mask = input_tensors['attention_mask']
        input_cache = DynamicCache()
        while not all_finished and input_ids.shape[-1] < 2600 and input_ids.shape[-1]-original_input_length <= 1024:
            print("#"*50+f"generating step {n_steps}"+ "#"*50)
            input_len = input_ids.shape[-1]
            if n_steps == 0:
                if original_input_length < 1100:
                    new_sample_size = beam_size * sample_size
                else:
                    # for saving costs
                    new_sample_size = max(beam_size, sample_size)
            else:
                new_sample_size = sample_size
            
            seq_ids, output_seq_scores, output_attn, best_cache = generate_one_step(input_ids, input_attn_mask, input_cache, max_new_tokens, new_sample_size, new_sample_size, terminators)

            if n_steps > start_step-1 and not baseline:
                beam_informative_steps_start_ends = []
                response_start = []
                for bs in range(seq_ids.size(0)):
                    output_start = (seq_ids[bs] == 78191).nonzero(as_tuple=True)[0] + 2
                    seq_splits = torch.isin(seq_ids[bs], torch.tensor(terminators, device=seq_ids[bs].device)).nonzero(as_tuple=True)[0].tolist()
                    raw_step_split_indice = seq_splits[:-1]
                    last_step = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(seq_ids[bs][seq_splits[-2]+1:seq_splits[-1]+1]))
                    
                    raw_step_split_indice = [_ for _ in raw_step_split_indice if _ > output_start[0]]
                    raw_step_split_indice = output_start.tolist() + raw_step_split_indice
                    
                    if seq_splits[-1] - seq_splits[-2] < 7 or "END" in last_step or seq_ids[bs][output_start[0]:seq_splits[-1]+1] is None:
                        output_attn = None
                        break
                    all_setps = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(seq_ids[bs][output_start[0]:seq_splits[-1]+1]))
                    if "So the answer is" in all_setps:
                        output_attn = None
                        break

                    step_split_indice = []
                    for jth in range(1, len(raw_step_split_indice)):
                        if raw_step_split_indice[jth] - raw_step_split_indice[jth-1] >= 8:
                            step_split_indice.append([raw_step_split_indice[jth-1]+1, raw_step_split_indice[jth]])
                        else:
                            pass

                    if len(step_split_indice) > 0:
                        informative_steps_start_ends= [[step_split_indice[-1][0], step_split_indice[-1][1]]]
                    else:
                        informative_steps_start_ends = []

                    all_steps = []
                    for k in range(len(step_split_indice)-1, -1, -1):
                        all_steps.append(seq_ids[bs][step_split_indice[k][0]:step_split_indice[k][1]].tolist())
                    if len(all_steps) >= 2:
                        informative_steps = get_all_step_infogain(all_steps, tokenizer, strategy = "all", thres=thres)
                        for n in informative_steps:
                            informative_steps_start_ends.append([step_split_indice[-1-n][0], step_split_indice[-1-n][1]])
                    beam_informative_steps_start_ends.append(informative_steps_start_ends)
                    response_start.append(output_start[0])

            if n_steps > start_step-1 and not baseline:
                beam_conc_information_gain = []
                for bs in range(seq_ids.size(0)):
                    output_start = (seq_ids[bs] == 78191).nonzero(as_tuple=True)[0] + 2
                    seq_splits = torch.isin(seq_ids[bs], torch.tensor(terminators, device=seq_ids[bs].device)).nonzero(as_tuple=True)[0].tolist()
                    raw_step_split_indice = seq_splits[:-1]
                    last_step = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(seq_ids[bs][seq_splits[-2]+1:seq_splits[-1]+1]))
                    
                    raw_step_split_indice = [_ for _ in raw_step_split_indice if _ > output_start[0]]
                    raw_step_split_indice = output_start.tolist() + raw_step_split_indice

                    if seq_splits[-1] - seq_splits[-2] < 7 or "END" in last_step:
                        beam_conc_information_gain.append(1)
                        continue
                    all_setps = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(seq_ids[bs][output_start[0]:seq_splits[-1]+1]))
                    if "So the answer is" in all_setps:
                        beam_conc_information_gain.append(1)
                        continue

                    # get conclusion information gain
                    last_step_ids = seq_ids[bs][seq_splits[-2]+1:seq_splits[-1]+1].tolist()
                    previous_steps_ids = []
                    for jth in range(1, len(raw_step_split_indice)):
                        previous_steps_ids.append(seq_ids[bs][raw_step_split_indice[jth-1]+1:raw_step_split_indice[jth]+1].tolist())
                    beam_conc_information_gain.append(get_new_conclusion(last_step_ids, previous_steps_ids, tokenizer))

            n_steps += 1
            parse_start = time.perf_counter()
            output_one_step, seq_scores, step_len, raw_finish, raw_seq_tokens, total_cost = parse_output((seq_ids, output_seq_scores), input_len, batch_size, terminator, num_hype_to_keep, sample_size, total_cost)
            del output_seq_scores, input_cache, input_ids, input_attn_mask
            torch.cuda.empty_cache()

            compute_start = time.perf_counter()
            if n_steps > start_step and not baseline:
                attn_scores = compute_scores_step_startend(output_attn, step_len, beam_informative_steps_start_ends, batch_size, num_hype_to_keep, sample_size, beam_size, response_start) 
            else:
                attn_scores = None

            base_num = [0]
            for i in range(batch_size):
                base_num.append(base_num[-1]+num_hype_to_keep[i]*sample_size)

            # add previous sequence scores for stepwsie beam search
            if n_steps > 1:
                for i in range(batch_size):
                    for j, score_n in enumerate(remain_scores[i]):
                        seq_scores[i][j*int(len(seq_scores[i])/len(remain_scores[i])):(j+1)*int(len(seq_scores[i])/len(remain_scores[i]))] += score_n

            if n_steps > start_step and not baseline:
                beam_conc_information_gain = torch.tensor(beam_conc_information_gain, device=seq_scores[0].device)
                bs_start = 0
                for i in range(batch_size):
                    if len(seq_scores[i]) > 0:
                        cur_beam_conc_information_gain = beam_conc_information_gain[bs_start:bs_start+len(seq_scores[i])]
                        bs_start += len(seq_scores[i])
                        if sum(cur_beam_conc_information_gain < conc_threshold) < len(seq_scores[i]):
                            seq_scores[i][(cur_beam_conc_information_gain < conc_threshold) & (cur_beam_conc_information_gain >= 0)] = -100
            best_beam_idx = select_best_k(seq_scores, attn_scores, num_hype_to_keep, batch_size)

            """Add finished candidates"""
            # Each batch has different number of sequences to continue
            continue_beam_idx = []

            remain_scores = [[] for _ in range(batch_size)]
            remain_attn_scores = [[] for _ in range(batch_size)]
            for i, indices in enumerate(best_beam_idx):
                for idx in indices:
                    if raw_finish[i][idx]:
                        if num_hype_to_keep[i] == 0:
                            sorted_candidates = sorted(candidates[i], key=lambda x: x[1])
                            if seq_scores[i][idx] > sorted_candidates[0][1]:
                                sorted_candidates[0] = (seq_ids[idx+base_num[i]], seq_scores[i][idx], attn_scores[i][idx])
                            candidates[i] = sorted_candidates
                        else:
                            if attn_scores is not None:
                                candidates[i].append((seq_ids[idx+base_num[i]], seq_scores[i][idx], attn_scores[i][idx]))
                            else:
                                candidates[i].append((seq_ids[idx+base_num[i]], seq_scores[i][idx], torch.tensor(0, device=device)))
                            num_hype_to_keep[i] -= 1
                            if num_hype_to_keep[i] == 0:
                                batch_finish[i] = True
                    else:
                        continue_beam_idx.append(idx+base_num[i])
                        remain_scores[i].append(seq_scores[i][idx])
                        if attn_scores is not None:
                            remain_attn_scores[i].append(attn_scores[i][idx])
                        else:
                            remain_attn_scores[i].append(torch.tensor(0, device=device))

            postproc_start = time.perf_counter()
            """Prepare next inputs"""
            input_ids, input_cache, input_attn_mask = prepare_input(seq_ids, best_cache, continue_beam_idx, sample_size) if len(continue_beam_idx) > 0 else (None, None, None)
            del output_attn, best_cache
            torch.cuda.empty_cache()
            all_finished = all(batch_finish)
            del raw_seq_tokens, output_one_step
            torch.cuda.empty_cache()

        if n_steps == 0:
            return "", 0, "", total_cost

        left_best_beam_idx = [list(set(range(len(seq_scores[0]))) - set(best_beam_idx[0]))]
        for i, indices in enumerate(left_best_beam_idx):
            for idx in indices:
                if raw_finish[i][idx]:
                    if attn_scores is not None:
                        candidates[i].append((seq_ids[idx+base_num[i]], seq_scores[i][idx], attn_scores[i][idx]))
                    else:
                        candidates[i].append((seq_ids[idx+base_num[i]], seq_scores[i][idx], torch.tensor(0, device=device)))
                else:
                    remain_scores[i].append(seq_scores[i][idx])
                    if attn_scores is not None:
                        remain_attn_scores[i].append(attn_scores[i][idx])
                    else:
                        remain_attn_scores[i].append(torch.tensor(0, device=device))

        if not all_finished:
            base_num = 0
            for i in range(batch_size):
                if len(candidates[i]) == 0:
                    for j in range(base_num, base_num+num_hype_to_keep[i]):
                        candidates[i].append((input_ids[j], remain_scores[i][j-base_num], remain_attn_scores[i][j-base_num]))
                    base_num += num_hype_to_keep[i]

        final_output = []
        sc_final_output = []
        candidate_ids, candidate_scores, candidate_attn_scores = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
        for i, each_candidates in enumerate(candidates):
            for candidate in each_candidates:
                for j in range(1, len(candidate[0])+1):
                    if candidate[0][-j] == 0:
                        continue
                    candidate_ids[i].append(candidate[0][:len(candidate[0])-j+1])
                    break
                candidate_scores[i].append(candidate[1])
                candidate_attn_scores[i].append(candidate[2])
            candidate_scores[i] = torch.stack(candidate_scores[i])
            candidate_attn_scores[i] = torch.stack(candidate_attn_scores[i])

        best_beam_idx = select_best_k(candidate_scores, None, [1 for _ in range(batch_size)], batch_size)
        for i, indices in enumerate(best_beam_idx):
            for idx in indices:
                final_output.append(tokenizer.decode(candidate_ids[i][idx].tolist()))
        for i, x in enumerate(final_output):
            final_output[i] = x.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].split("<|end_of_text|>")[0].strip()
        return final_output, n_steps, total_cost

    def ref_whole_process(output_file):
        '''
            Test on MMLU-Pro
        '''
        if not is_selfselect:
            prompt = '''You will receive a query and ten options. Your task is to select an option to answer the query.

#### Examples
Query: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?
Options: A.24, B.54, C.40, D.32, E.64, F.8, G.16, H.60, I.100, J.74
Thought: 
Because one glass costs $5, and every second glass costs only 60% of the price, so the discount price of every second glass is 60/100 * 5 = $3.
Because every second glass is discounted at $3, and Kylar wants to buy 16 glasses, so Kylar is going to buy 16 / 2 = 8 discounted glasses and 16 - 8 = 8 regular-priced glasses.
Because Kylar is going to buy 8 discounted glasses, and every discounted glass is $3, so Kylar is going to pay 8 * 3 = $24.
Because Kylar is also going to buy 8 regular-priced glasses, and one glass costs $5, so Kylar will pay 8 * 5 = $40.
Because Kylar will pay $24 for 8 discounted glasses, and $40 for 8 regular-priced glasses, so in total Kylar needs to pay 24 + 40 = $64 for the glasses he wants to buy.
END.
So the answer is: E.
------
Query: A refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is ?
Options: A.10, B.40, C.6, D.25, E.15, F.50, G.30, H.4, I.5, J.20
Thought: 
Because in a refracting telescope both lenses are converging, so their focus must be between the two lenses.
Because the focus of both lenses must lie between them, so their focal lengths must add up to their separation. 
Because the two lenses are separated by 100 cm, and one lens has a focal length of 20 cm, so the other lens must have a focal length of 80 cm.
Because one lens has a focal length of 20 cm and the other 80 cm, so the magnification is the ratio of their focal lengths, which is 4.
END.
So the answer is: H. 

#### Here's what you need to do. Please first think step-by-step, presenting each of your step in a new line. Then end your thought with "END.". Finally respond with an option from "A", "B", "C", "D", "E", "F", "G", "H", "I" or "J" in a newline, strictly starting with "So the answer is: ".
Query: {question}
Options: {option}
Thought: '''

        else:
            prompt = '''You will receive a query and ten options. Your task is to select an option to answer the query.

#### Examples
Query: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?
Options: A.24, B.54, C.40, D.32, E.64, F.8, G.16, H.60, I.100, J.74
Thought: 
[Step-1] From Query, because one glass costs $5, and every second glass costs only 60% of the price, so the discount price of every second glass is 60/100 * 5 = $3.
[Step-2] From Step-1 and Query, because every second glass is discounted at $3, and Kylar wants to buy 16 glasses, so Kylar is going to buy 16 / 2 = 8 discounted glasses and 16 - 8 = 8 regular-priced glasses.
[Step-3] From Step-1 and Step-2, because Kylar is going to buy 8 discounted glasses, and every discounted glass is $3, so Kylar is going to pay 8 * 3 = $24.
[Step-4] From Step-2 and Query, because Kylar is also going to buy 8 regular-priced glasses, and one glass costs $5, so Kylar will pay 8 * 5 = $40.
[Step-5] From Step-3 and Step-4, because Kylar will pay $24 for 8 discounted glasses, and $40 for 8 regular-priced glasses, so in total Kylar needs to pay 24 + 40 = $64 for the glasses he wants to buy.
So the answer is: E.
------
Query: A refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is ?
Options: A.10, B.40, C.6, D.25, E.15, F.50, G.30, H.4, I.5, J.20
Thought: 
[Step-1] From Query, because in a refracting telescope both lenses are converging, so their focus must be between the two lenses.
[Step-2] From Step-1, because the focus of both lenses must lie between them, so their focal lengths must add up to their separation. 
[Step-3] From Step-2 and Query, because the two lenses are separated by 100 cm, and one lens has a focal length of 20 cm, so the other lens must have a focal length of 80 cm.
[Step-4] From Step-3 and Query, because one lens has a focal length of 20 cm and the other 80 cm, so the magnification is the ratio of their focal lengths, which is 4.
So the answer is: H.

#### Here's what you need to do. Please first think step-by-step, presenting each of your step in a new line starting with "[Step-i]", and cite the sources (e.g., Step-i, Query) of your premises at the begining of each step. Finally respond with an option from "A", "B", "C", "D", "E", "F", "G", "H", "I" or "J" in a newline, strictly starting with "So the answer is: ".
Query: {question}
Options: {option}
Thought: '''

        print("testing on mmlu-pro\n")
        terminator = "Ċ"

        ## data preparation
        all_data = load_dataset("TIGER-Lab/MMLU-Pro")['test']
        import random
        data = [[] for _ in range(14)]
        categories = ['math', 'health', 'physics', 'business', 'biology', 'chemistry', 'computer science', 'economics',
                    'engineering', 'philosophy', 'other', 'history', 'psychology', 'law']
        for i in range(len(all_data)):
            if all_data[i]['category'] in categories:
                cate_id = categories.index(all_data[i]['category'])
                data[cate_id].append(all_data[i])
        val_data = []
        for i, each_data in enumerate(data):
            random.seed(i)
            val_data += random.sample(each_data, 20)
        print("val num", len(val_data))

        ## evaluate
        accu_num = 0
        total_num = 0
        output_dict = {}
        all_lens = []
        all_token_num = []
        all_step_num = []
        total_cost = []
        for bs in range(math.ceil(len(val_data)/batch_size)):
            i = bs * batch_size
            questions = []
            options = []
            ref_answers = []
            for j in range(i, min(i+batch_size, len(val_data))):
                questions.append(val_data[j]['question'].strip())
                option_list = []
                for k, _ in enumerate(val_data[j]["options"]):
                    option_list.append(option_ids[k]+"."+_)
                options.append(", ".join(option_list))
                ref_answers.append(val_data[j]['answer'])
                
            input_len = len(val_data[bs]['question'].split(". "))
            output, step_num, cost = generate_beam_search(prompt=prompt, questions=questions, options=options, batch_size=len(questions), beam_size=beam_size, sample_size=sample_size, terminator=terminator, 
                                        start_step=start_step, thres=premise_info_thres, conc_threshold=conc_threshold)
            all_lens += [len(_.split()) for _ in output]
            all_token_num += [len(tokenizer(_)[0]) for _ in output]
            all_step_num.append(step_num)
            total_cost.append(cost)

            for k in range(len(output)):
                pred_answer = output[k]
                cur_ref_answer = ref_answers[k]
                output_dict[i+k] = pred_answer

                if "So the answer is" in pred_answer:
                    pred_answer = pred_answer.split("So the answer is:")[-1].split("So the answer is")[-1].split("<|eot_id|>")[0].split("<|end_of_text|>")[0].strip()
                    pred_answer = pred_answer.split(".")[0][:1]
                    if cur_ref_answer == pred_answer:
                        accu_num += 1
                    else:
                        print(f"Wrong answer: {pred_answer} | ref: {cur_ref_answer}")
                else:
                    print("[ERROR] No answer\n")

            print("#"*100)
            torch.cuda.empty_cache()
            total_num += batch_size
            print(f"**Step{i}**", (time.perf_counter()-start_time)/60)
            print(f"Cur Accuracy: {accu_num/total_num}, {cost} tokens")

            if len(output_file) > 0:
                with open(output_file, 'w') as file:
                    json.dump(output_dict, file, indent=2)
                print("Saved!")
        print(f"Accuracy: {accu_num}/{total_num}  {accu_num/total_num}")
        print("Avg.len", np.mean(all_lens), np.mean(all_token_num), np.mean(total_cost))
        print(f"Average Step Number: {sum(all_step_num)/len(all_step_num)}")

    ref_whole_process(output_file)


if __name__ == "__main__":
    fire.Fire(main)
