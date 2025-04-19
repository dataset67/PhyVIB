# !pip install torch numpy transformers datasets safetensors sentencepiece accelerate wandb modelscope open-clip-torch einops timm Pillow

import random as py_random
import copy as obj_copier
import re as regex_tool
import os as operating_system
import numpy as numerical_pkg
import wandb as weights_biases_logger
import torch as pytorch_core
import torch.nn as neural_network_module
from torch.nn.utils.rnn import pad_sequence as pad_pytorch_sequences
from transformers import AutoModelForCausalLM as hf_causal_lm, AutoTokenizer as hf_tokenizer
from datasets import load_dataset as hf_load_ds
from safetensors import safe_open as secure_tensor_open
from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel, AutoProcessor as HFProcessor
from qwen_vl_utils import process_vision_info as qwen_process_vision
import json as json_parser

operating_system.environ["WANDB_API_KEY"] = "...f"
operating_system.environ["WANDB_PROJECT"] = "..."


model_main = QwenVLModel.from_pretrained("...",
                                         torch_dtype=pytorch_core.bfloat16,
                                         device_map="auto")
processor_main = AutoProcessor.from_pretrained("...",
                                             torch_dtype=pytorch_core.bfloat16,
                                             device_map="auto")


operating_system.environ["WANDB_API_KEY"] = "\\"
operating_system.environ["WANDB_PROJECT"] = "\\"

def get_answer_tag(text_in):
    start_tag = "<answer>"
    end_tag = "</answer>"
    parts = text_in.split(start_tag)
    result = None
    if len(parts) >= 2:
        last_part = parts[-1]
        end_pos = last_part.find(end_tag)
        if end_pos != -1:
            content = last_part[:end_pos].strip()
            if content != "...":
                 result = content
    return result

def get_confidence_tag(text_in):
    start_tag = "<confidence>"
    end_tag = "</confidence>"
    parts = text_in.split(start_tag)
    result = None
    if len(parts) >= 2:
        last_part = parts[-1]
        end_pos = last_part.find(end_tag)
        if end_pos != -1:
            content = last_part[:end_pos].strip()
            if content != "...":
                result = content
    return result

def get_ds_answer(text_in):
    output = None
    if text_in is not None and isinstance(text_in, str) and len(text_in) > 0:
        output = text_in
    return output


def prepare_data(split_type="train"):
    fpath = ""
    out_data = []
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            raw_data = json_parser.load(f)

        data_iter = iter(raw_data)
        while True:
             try:
                 item = next(data_iter)
                 fmt_item = {
                     "prompt": item.get('message', ''),
                     "answer": item.get('answer', '')
                 }
                 out_data.append(fmt_item)
             except StopIteration:
                 break

    except FileNotFoundError:
        return []
    except json_parser.JSONDecodeError:
        return []

    def get_sort_key(x):
         p = x.get("prompt", "")
         return p if isinstance(p, str) else ""
    out_data.sort(key=get_sort_key)

    return out_data


def extract_last_num(text_in):
    proc_text = text_in.replace('$', '').replace('%', '')
    pattern = r'.*?(?:^|\s|=)\s*(-?\d+(?:\.\d+)?)\s*$'
    match = regex_tool.match(pattern, proc_text)
    num_val = None
    if match:
        try:
            num_val = float(match.group(1))
        except ValueError:
            pass
    return num_val

def extract_one_num(text_in):
    nums = regex_tool.findall(r'-?\d+(?:\.\d+)?', text_in)
    num_val = None
    if len(nums) == 1:
        try:
             num_val = float(str(nums[0]))
        except ValueError:
            pass
    elif len(nums) > 1:
        try:
            num_val = float(nums[-1])
        except ValueError:
            pass
    return num_val


def run_evaluation(mdl, proc, eval_set, dev):
    mdl.eval()
    n_correct = 0
    n_total = len(eval_set)
    idx = 0

    while idx < n_total:
        item = eval_set[idx]
        prompt = item.get("prompt")
        truth = item.get("answer")

        if prompt is None or truth is None:
            idx += 1
            continue

        templated_txt = proc.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = qwen_process_vision(prompt)
        inputs = proc(text=[templated_txt], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
        inputs_on_dev = {k: v.to(dev) for k, v in inputs.items()}

        with pytorch_core.no_grad():
            outputs = mdl.generate(
                **inputs_on_dev,
                max_new_tokens=512,
                pad_token_id=proc.tokenizer.pad_token_id,
                eos_token_id=proc.tokenizer.eos_token_id,
                forced_eos_token_id=proc.tokenizer.eos_token_id,
                early_stopping=False,
                do_sample=False
            )
        response = proc.decode(outputs[0], skip_special_tokens=True)

        is_correct = False
        try:
            pred = get_answer_tag(response)
            if pred is not None:
                 if str(pred).strip() == str(truth).strip():
                      is_correct = True
            elif truth is None:
                 is_correct = True

            if is_correct:
                n_correct += 1
        except Exception:
             pass
        idx += 1

    accuracy = (n_correct / n_total) * 100.0 if n_total > 0 else 0.0
    mdl.train()
    return accuracy


def reward_correctness(prompts, completions, answer, **kwargs):
    responses = [c[0]['content'] for c_list in completions for c in c_list if isinstance(c_list, list) and c_list and isinstance(c, dict) and 'content' in c]
    extracted = [get_answer_tag(r) for r in responses]
    confidences = [get_confidence_tag(r) for r in responses]

    rewards = []
    for r, a, conf_str in zip(extracted, answer, confidences):
        conf_float = None
        if conf_str is not None:
            try:
                conf_float = float(conf_str)
                conf_float = min(max(0.0, conf_float), 1.0)
            except (ValueError, TypeError):
                conf_float = None

        reward_val = 0.0
        if r == a:
            reward_val = 2.0 * (conf_float + 0.02) if conf_float is not None else 2.0 * 0.3
        else:
            if conf_float is None:
                 reward_val = -0.02
            elif conf_float < 0.5:
                 reward_val = 0.0
            else:
                 reward_val = 0.0 - conf_float * 0.1

        rewards.append(reward_val)

    return rewards

def reward_format(completions, **kwargs):
    responses = [c[0]['content'] for c_list in completions for c in c_list if isinstance(c_list, list) and c_list and isinstance(c, dict) and 'content' in c]
    scores = []
    for resp in responses:
        score = 0.0
        score += 0.2 if "<reasoning>" in resp else 0.0
        score += 0.2 if "</reasoning>" in resp else 0.0
        score += 0.2 if "<answer>" in resp else 0.0
        score += 0.2 if "</answer>" in resp else 0.0
        scores.append(min(score, 0.8))
    return scores

def reward_combined(prompts, completions, answer):
    correct_scores = reward_correctness(prompts=prompts, completions=completions, answer=answer)
    format_scores = reward_format(completions=completions)

    combined = [(cs if isinstance(cs, (int, float)) else 0.0) + (fs if isinstance(fs, (int, float)) else 0.0)
                for cs, fs in zip(correct_scores, format_scores)]

    return numerical_pkg.array(combined).tolist()


def get_selected_logprobs(logits, target_ids):
    log_probs = neural_network_module.functional.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1))
    return gathered.squeeze(-1)

def compute_logprobs_for_tokens(mdl, input_ids, attn_mask, num_logits):
    outputs = mdl(input_ids=input_ids, attention_mask=attn_mask)
    logits_all = outputs.logits
    logits_sub = logits_all[:, :-1, :]
    ids_sub = input_ids[:, -num_logits:]
    logits_final = logits_sub[:, -num_logits:, :]
    return get_selected_logprobs(logits_final, ids_sub)

def make_completion_mask(completion_ids, eos_id):
    is_eos = completion_ids == eos_id
    n_seq, seq_len = is_eos.size()
    eos_idxs = pytorch_core.full((n_seq,), seq_len, dtype=pytorch_core.long, device=completion_ids.device)
    has_eos = is_eos.any(dim=1)
    if pytorch_core.any(has_eos):
         first_eos = is_eos.int().argmax(dim=1)
         eos_idxs[has_eos] = first_eos[has_eos]

    seq_indices = pytorch_core.arange(seq_len, device=completion_ids.device)
    expanded_indices = seq_indices.expand(n_seq, -1)

    mask = (expanded_indices <= eos_idxs.unsqueeze(1)).int()
    return mask


def generate_model_outputs(mdl, proc, prompts, n_gen=4, max_len=32):
    dev = next(mdl.parameters()).device # Get device from model
    texts = []
    images = []
    for p in prompts:
        txt = proc.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        texts.append(txt)
        img_d, vid_d = qwen_process_vision(p)
        images.append(img_d)

    inputs = proc(text=texts, images=images, padding=True, return_tensors="pt")

    prompt_ids = inputs["input_ids"].to(dev)
    prompt_mask = inputs["attention_mask"].to(dev)

    prompt_len = prompt_ids.size(1)
    rep_prompt_ids = prompt_ids.repeat_interleave(n_gen, dim=0)
    rep_prompt_mask = prompt_mask.repeat_interleave(n_gen, dim=0)
    gen_outputs = mdl.generate(
        rep_prompt_ids,
        attention_mask=rep_prompt_mask,
        do_sample=True,
        pad_token_id=proc.tokenizer.pad_token_id,
        eos_token_id=proc.tokenizer.eos_token_id,
        max_new_tokens=max_len,
        temperature=1.0,
        early_stopping=False
    )
    gen_ids = gen_outputs[:, prompt_len:]

    proc.tokenizer.padding_side = "left"
    gen_mask = make_completion_mask(gen_ids, proc.tokenizer.eos_token_id)

    return rep_prompt_ids, rep_prompt_mask, gen_ids, gen_mask


def create_rollout_batch(policy_mdl, ref_mdl, proc, batch, n_gen, max_len):
    dev = next(policy_mdl.parameters()).device

    prompts = []
    answers = []
    for item in batch:
        if isinstance(item, dict):
            prompts.append(item.get("prompt", ""))
            answers.append(item.get("answer", ""))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
             prompts.append(item[0])
             answers.append(item[1])
        else:
             prompts.append("")
             answers.append("")

    with pytorch_core.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_model_outputs(
            policy_mdl, proc, prompts, n_gen, max_len
        )

        full_ids = pytorch_core.cat([prompt_ids, completion_ids], dim=1)
        full_mask = pytorch_core.cat([prompt_mask, completion_mask], dim=1)

        n_completion_logits = completion_ids.size(1)

        policy_logprobs = compute_logprobs_for_tokens(policy_mdl, full_ids, full_mask, n_completion_logits)
        ref_logprobs = compute_logprobs_for_tokens(ref_mdl, full_ids, full_mask, n_completion_logits)

    completions_fmt = []
    for ids in completion_ids:
        decoded = proc.decode(ids, skip_special_tokens=True)
        completions_fmt.append([{'content': decoded}])

    rep_prompts = [p for p in prompts for _ in range(n_gen)]
    rep_answers = [a for a in answers for _ in range(n_gen)]

    rollout_data = {
        "input_ids": full_ids,
        "attn_mask": full_mask,
        "comp_mask": completion_mask,
        "old_logprobs": policy_logprobs,
        "ref_logprobs": ref_logprobs,
        "completions": completions_fmt,
        "prompts_rep": rep_prompts,
        "answers_rep": rep_answers,
        "n_comp_logits": n_completion_logits,
        "batch_sz": len(prompts),
        "n_gen": n_gen
    }
    return rollout_data


def calculate_grpo_loss(policy_mdl, ref_mdl, rollout_data, proc, reward_func, beta=0.01, epsilon=0.2):
    dev = next(policy_mdl.parameters()).device

    input_ids = rollout_data["input_ids"]
    attn_mask = rollout_data["attn_mask"]
    comp_mask = rollout_data["comp_mask"]
    n_comp_logits = rollout_data["n_comp_logits"]
    old_logprobs = rollout_data["old_logprobs"]
    ref_logprobs = rollout_data["ref_logprobs"]

    curr_logprobs = compute_logprobs_for_tokens(policy_mdl, input_ids, attn_mask, n_comp_logits)
    logprob_diff = curr_logprobs - old_logprobs
    ratio = pytorch_core.exp(logprob_diff)

    rewards_raw = reward_func(
        prompts=rollout_data["prompts_rep"],
        completions=rollout_data["completions"],
        answer=rollout_data["answers_rep"]
    )
    rewards = pytorch_core.tensor(rewards_raw, dtype=pytorch_core.float32, device=dev)
    batch_sz = rollout_data["batch_sz"]
    n_gen = rollout_data["n_gen"]
    rewards_reshaped = rewards.view(batch_sz, n_gen)
    avg_reward = rewards_reshaped.mean().item()
    mean_r = rewards_reshaped.mean(dim=1, keepdim=True)
    std_r = pytorch_core.nan_to_num(rewards_reshaped.std(dim=1, keepdim=True))
    rep_mean = mean_r.repeat_interleave(n_gen, dim=0)
    rep_std = std_r.repeat_interleave(n_gen, dim=0)
    rewards_flat = rewards_reshaped.view(-1)
    adv = ((rewards_flat - rep_mean.squeeze()) / (rep_std.squeeze() + 1e-5)).unsqueeze(1)
    surr1 = ratio * adv
    ratio_clip = pytorch_core.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    surr2 = ratio_clip * adv
    ppo_obj = pytorch_core.min(surr1, surr2) 
    logp_diff_ref = ref_logprobs - curr_logprobs
    kl = pytorch_core.exp(logp_diff_ref) - logp_diff_ref - 1.0
    token_loss = ppo_obj - beta * kl
    masked = token_loss * comp_mask.unsqueeze(-1)
    summed_seq = masked.sum(dim=1)
    n_tokens_seq = comp_mask.sum(dim=1)
    avg_seq = summed_seq / (n_tokens_seq + 1e-8)
    final_loss = -avg_seq.mean()
    return final_loss, avg_reward


def run_grpo_training(mdl_train, proc, train_ds,
                      n_loops=1, n_steps=#, batch_sz=#,
                      n_gen=4, max_len=128, beta=#,
                      lr=#, mu=#, epsilon=#,
                      reward_f=None, gpu_ids=None):

    if gpu_ids is None or len(gpu_ids) <= 1:
         raise ValueError("Need >= 2 GPUs")

    primary_dev = pytorch_core.device(f"cuda:{gpu_ids[0]}" if pytorch_core.cuda.is_available() else "cpu")

    mdl_parallel = neural_network_module.DataParallel(mdl_train, device_ids=gpu_ids)
    mdl_parallel.to(primary_dev)
    for loop_idx in range(n_loops):
        with pytorch_core.no_grad():
             ref_mdl = obj_copier.deepcopy(mdl_parallel.module).to(primary_dev)
             ref_mdl.eval()
             for param in ref_mdl.parameters():
                 param.requires_grad = False
        optimizer = pytorch_core.optim.AdamW(mdl_parallel.parameters(), lr=lr)
        mdl_parallel.train()
        for step_idx in range(n_steps):
            batch_indices = py_random.sample(range(len(train_ds)), k=batch_sz)
            batch_data = [train_ds[i] for i in batch_indices]
            with pytorch_core.no_grad():
                rollout = create_rollout_batch(
                    mdl_parallel.module,
                    ref_mdl,
                    proc,
                    batch_data,
                    n_gen,
                    max_len
                )
            for update_iter in range(mu):
                loss, avg_rew = calculate_grpo_loss(
                    mdl_parallel.module,
                    ref_mdl,
                    rollout,
                    proc,
                    reward_f,
                    beta=beta,
                    epsilon=epsilon
                )
                optimizer.zero_grad()
                loss.backward()
                pytorch_core.nn.utils.clip_grad_norm_(mdl_parallel.parameters(), max_norm=0.1)
                optimizer.step()
                log_info = {
                    "Loss": loss.item(),
                    "AvgReward": avg_rew,
                    "Loop": loop_idx + 1,
                    "Step": step_idx + 1,
                    "UpdateIter": update_iter + 1
                }
                weights_biases_logger.log(log_info)
    final_mdl = mdl_parallel.module
    return final_mdl

def setup_model_for_training(mdl):
    mdl.train()
    if hasattr(mdl.config, 'use_cache'):
        mdl.config.use_cache = False
    grad_hook_applied = False
    if hasattr(mdl, "enable_input_require_grads"):
        try:
            mdl.enable_input_require_grads()
            grad_hook_applied = True
        except Exception: pass
    if not grad_hook_applied:
        try:
            emb_layer = mdl.get_input_embeddings()
            if emb_layer is not None:
                def fwd_hook(module, inp, outp):
                    if isinstance(outp, pytorch_core.Tensor):
                         outp.requires_grad_(True)
                emb_layer.register_forward_hook(fwd_hook)
        except Exception: pass
    if hasattr(mdl, 'gradient_checkpointing_enable'):
         try:
             mdl.gradient_checkpointing_enable()
         except Exception: pass

    _ = [x*x for x in range(5)] # Dummy op

    return mdl

main_dev = pytorch_core.device("cuda:0" if pytorch_core.cuda.is_available() else "cpu")
n_gpus = pytorch_core.cuda.device_count()
gpu_list = list(range(n_gpus)) if n_gpus > 1 else None
full_ds = prepare_data("train")
py_random.shuffle(full_ds)
eval_sz = 60
eval_ds = full_ds[0:eval_sz]
train_ds = full_ds[eval_sz:]
pre_acc = run_evaluation(model_main, processor_main, eval_ds, main_dev)
model_opt = setup_model_for_training(model_main)
train_params = {
    'n_loops': 1,
    'n_steps': 50,
    'batch_sz': 2,
    'n_gen': 2,
    'max_len': 100,
    'beta': 0.04,
    'lr': 7e-5,
    'mu': 1,
    'epsilon': 0.1
}
weights_biases_logger.init(project=operating_system.environ["WANDB_PROJECT"], reinit=True)
trained_model = run_grpo_training(
    mdl_train=model_opt,
    proc=processor_main,
    train_ds=train_ds,
    reward_f=reward_combined,
    gpu_ids=gpu_list,
    **train_params
)
weights_biases_logger.finish()
post_acc = run_evaluation(trained_model, processor_main, eval_ds, main_dev)
trained_model.save_pretrained("")
processor_main.save_pretrained("")
