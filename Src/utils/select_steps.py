from collections import Counter
import re

def split_steps(ids_list, tokenizer):
    cur_step_str = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids_list))
    cur_step_str = cur_step_str.split(". (Source")[0].split(". [Source")[0].split(". (Query")[0].split(". (Step-")[0].strip()
    cur_step_str = cur_step_str.replace("<|begin_of_text|>", "")

    pattern = r"\[Step-\d+\]\s+(?:From\s+(?:Step-\d+|the\s+[Qq]uery))?(?:\s+and\s+(?:[Qq]uery|Step-\d+))?,?"
    pattern2 = r"\[Step-\d+,\s*Query\]"
    combined_pattern = f"(?:{pattern})|(?:{pattern2})"
    cleaned_parts = re.split(combined_pattern, cur_step_str)
    if len(cleaned_parts) > 0:
        cur_step_str = cleaned_parts[-1]
        cur_step = tokenizer(cur_step_str, add_special_tokens=False)["input_ids"]
        return cur_step
    else:
        return ids_list

def get_joint_str(l):
    l = " ".join([str(_) for _ in l])
    return l

def compute_max_infogain(cur_step, previous_steps):
    max_overlap_rate = 0
    cur_step_trigram = Counter([get_joint_str(cur_step[i:i+3]) for i in range(len(cur_step) - 2)])
    if sum(cur_step_trigram.values()) == 0:
        return -1
    for each_prev_step in previous_steps:
        each_prev_step_trigram = Counter([get_joint_str(each_prev_step[i:i+3]) for i in range(len(each_prev_step) - 2)])
        overlap_rate = sum((cur_step_trigram & each_prev_step_trigram).values()) / sum(cur_step_trigram.values())
        if overlap_rate > max_overlap_rate:
            max_overlap_rate = overlap_rate
    return 1 - max_overlap_rate

def get_all_step_infogain(all_steps, tokenizer, strategy="all", thres=0.7, step_len=15):
    all_steps_infogain = []
    for i in range(1, len(all_steps)):
        previous_steps = [split_steps(_, tokenizer) for _ in all_steps[:i]]
        cur_step = split_steps(all_steps[i], tokenizer)

        if tokenizer.convert_tokens_to_ids("Ġso") in cur_step:
            cur_step_start = cur_step[::-1].index(tokenizer.convert_tokens_to_ids("Ġso"))
        elif tokenizer.convert_tokens_to_ids("ĠSo") in cur_step:
            cur_step_start = cur_step[::-1].index(tokenizer.convert_tokens_to_ids("ĠSo"))
        elif tokenizer.convert_tokens_to_ids("Ġthus") in cur_step:
            cur_step_start = cur_step[::-1].index(tokenizer.convert_tokens_to_ids("Ġthus"))
        elif tokenizer.convert_tokens_to_ids(",") in cur_step:
            cur_step_start = cur_step[::-1].index(tokenizer.convert_tokens_to_ids(","))
            if cur_step_start+1 < len(cur_step) and cur_step_start-1 >= 0:
                prev_later_tokens = tokenizer.convert_ids_to_tokens([cur_step[::-1][cur_step_start-1], cur_step[::-1][cur_step_start+1]])
                invalid_comma = prev_later_tokens[0].isnumeric() and prev_later_tokens[1].isnumeric()
            else:
                invalid_comma = False
            while (cur_step_start < step_len and cur_step_start < len(cur_step)/2) or invalid_comma:
                if tokenizer.convert_tokens_to_ids(",") in cur_step[:-cur_step_start-1]:
                    cur_step_start = cur_step[::-1].index(tokenizer.convert_tokens_to_ids(","), cur_step_start+1)
                    if cur_step_start+1 < len(cur_step) and cur_step_start-1 >= 0:
                        prev_later_tokens = tokenizer.convert_ids_to_tokens([cur_step[::-1][cur_step_start-1], cur_step[::-1][cur_step_start+1]])
                        invalid_comma = prev_later_tokens[0].isnumeric() and prev_later_tokens[1].isnumeric()
                    else:
                        invalid_comma = False
                else:
                    cur_step_start = step_len
                    invalid_comma = False
        else:
            cur_step_start = step_len
        cur_step = cur_step[-cur_step_start:]

        information_gain = compute_max_infogain(cur_step, previous_steps)
        all_steps_infogain.append(information_gain)

    if strategy == "all":
        indices = [j+1 for j, x in enumerate(all_steps_infogain) if x >= thres and len(all_steps[j+1]) >= 7]
        return indices


def get_conclusion(cur_step, tokenizer, step_len=15):
    if len(cur_step) == 0:
        return cur_step

    cur_step = split_steps(cur_step, tokenizer)
    if tokenizer.convert_tokens_to_ids("Ġso") in cur_step:
        conclusion_start = cur_step[::-1].index(tokenizer.convert_tokens_to_ids("Ġso"))
    elif tokenizer.convert_tokens_to_ids("ĠSo") in cur_step:
        conclusion_start = cur_step[::-1].index(tokenizer.convert_tokens_to_ids("ĠSo"))
    elif tokenizer.convert_tokens_to_ids("Ġthus") in cur_step:
        conclusion_start = cur_step[::-1].index(tokenizer.convert_tokens_to_ids("Ġthus"))
    elif tokenizer.convert_tokens_to_ids(",") in cur_step:
        conclusion_start = cur_step[::-1].index(tokenizer.convert_tokens_to_ids(","))
        if conclusion_start+1 < len(cur_step) and conclusion_start-1 >= 0:
            prev_later_tokens = tokenizer.convert_ids_to_tokens([cur_step[::-1][conclusion_start-1], cur_step[::-1][conclusion_start+1]])
            invalid_comma = prev_later_tokens[0].isnumeric() and prev_later_tokens[1].isnumeric()
        else:
            invalid_comma = False
        while (conclusion_start < step_len and conclusion_start < len(cur_step)/2) or invalid_comma:
            if tokenizer.convert_tokens_to_ids(",") in cur_step[:-conclusion_start-1]:
                conclusion_start = cur_step[::-1].index(tokenizer.convert_tokens_to_ids(","), conclusion_start+1)
                if conclusion_start+1 < len(cur_step) and conclusion_start-1 >= 0:
                    prev_later_tokens = tokenizer.convert_ids_to_tokens([cur_step[::-1][conclusion_start-1], cur_step[::-1][conclusion_start+1]])
                    invalid_comma = prev_later_tokens[0].isnumeric() and prev_later_tokens[1].isnumeric()
                else:
                    invalid_comma = False
            else:
                conclusion_start = step_len
                invalid_comma = False
    else:
        conclusion_start = step_len 
    conclusion = cur_step[-conclusion_start:]
    return conclusion

    
def get_new_conclusion(cur_step, previous_steps, tokenizer, step_len=15):
    cur_step_conc = get_conclusion(cur_step, tokenizer, step_len)
    previous_concs = []
    for each_prev_step in previous_steps:
        previous_concs.append(get_conclusion(each_prev_step, tokenizer, step_len))
    
    return compute_max_infogain(cur_step_conc, previous_concs)



        
    
