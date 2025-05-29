import argparse, gc, os, pickle

import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from lpaaj.data import TextDataset
from lpaaj.constants import MODEL_PATH, LORA_RESULTS_PATH, FULL_RESULTS_PATH



prompt_keys = {
    'newsroom': ['coherence', 'fluency', 'informativeness', 'relevance'],
    'summeval': ['coherence', 'consistency', 'fluency', 'relevance'],
    'hanna': ['coherence', 'complexity', 'empathy', 'engagement', 'relevance', 'surprise'],
    'rocstories': ['consistency']
} # default: prompt

label_keys = {
    'mctaco': 'correct',
    'caters': 'first',
    'rocstories': 'correct'
} # default: prompt-key


def load_model_and_tokenizer(
        model_name: str,
        max_num_seqs: int,
) -> tuple[LLM, AutoTokenizer]:
    # === LOAD TOKENIZER ===
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model_name}", trust_remote_code=True)
    # === LOAD MODEL ===
    if "qwen-2.5-0.5b" in model_name:
        tp_size = 2
    elif "qwen-2.5-1.5b" in model_name:
        tp_size = 4
    elif "qwen-2.5-7b" in model_name:
        tp_size = 4
    else:
        tp_size = t.cuda.device_count()
    llm_kwargs = {
        "model": f"{MODEL_PATH}/{model_name}",
        "gpu_memory_utilization": 0.98,
        "tensor_parallel_size": tp_size,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_num_seqs": max_num_seqs,
        "max_model_len": 8192,
        "enable_prefix_caching": True,
        "seed": 123456,
        "task": "generate",
        "enforce_eager": True,
        "enable_lora": True,
        "max_lora_rank": 8
    }
    model = LLM(**llm_kwargs)
    return model, tokenizer


def main(
        args: argparse.Namespace,
) -> None:
    # === TEXT QUALITY ===
    datasets = ["newsroom", "summeval", "hanna"]
    if args.lora:
        model_name = args.model
    else:
        model_name = f"{args.model}-full-text_quality"
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        args.max_num_seqs,
    )
    for dataset in datasets:
        pks = prompt_keys.get(dataset, ["prompt"])
        for pk in pks:
            lk = label_keys.get(dataset, pk)
            inference(
                model,
                tokenizer,
                dataset,
                pk,
                lk,
                args
            )
    del model, tokenizer
    gc.collect()
    t.cuda.empty_cache()

    # === COMMON SENSE ===
    datasets = ["mctaco", "caters", "rocstories"]
    if args.lora:
        model_name = args.model
    else:
        model_name = f"{args.model}-full-common_sense"
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        args.max_num_seqs,
    )
    for dataset in datasets:
        pks = prompt_keys.get(dataset, ["prompt"])
        for pk in pks:
            lk = label_keys.get(dataset, pk)
            inference(
                model,
                tokenizer,
                dataset,
                pk,
                lk,
                args
            )
    del model, tokenizer
    gc.collect()
    t.cuda.empty_cache()


def inference(
        model: LLM,
        tokenizer: AutoTokenizer,
        dataset: str,
        prompt_key: str,
        label_key: str,
        args: argparse.Namespace,
) -> None:
    # determine lora path
    if dataset in ["newsroom", "summeval", "hanna"]:
        lora_path = f"{MODEL_PATH}/{args.model}-lora-text_quality"
    elif dataset in ["mctaco", "caters", "rocstories"]:
        lora_path = f"{MODEL_PATH}/{args.model}-lora-common_sense"
    # === CHECK FOR EXISTING RESULTS ===
    RESULT_PATH = LORA_RESULTS_PATH if args.lora else FULL_RESULTS_PATH
    outpath = f"{RESULT_PATH}/{dataset}/{args.model}/{prompt_key}.pkl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return
    else:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # === LOAD DATASET ===
    dataset = TextDataset(
        task="compare",
        dataset=dataset,
        prompt_key=prompt_key,
        label_key=label_key,
    )
    dataset.preprocess_prompts(tokenizer)
    # === GENERATE ===
    gen_kwargs = {
        "prompts": list(dataset.prompts),
        "sampling_params": SamplingParams(
            max_tokens=1,
            skip_special_tokens=False,
            logprobs=5,
            temperature=0.0
        ),
        "lora_request": LoRARequest("adapter", 1, lora_path=lora_path) if args.lora else None,
        "use_tqdm": True,
    }
    outputs = model.generate(**gen_kwargs)
    # === PREDICTIONS ===
    predictions = []
    for output in outputs:
        # grab logits
        valid_tks = ["1", "2"]
        prediction = -1
        logprobs = output.outputs[0].logprobs
        if logprobs:
            for _, logprob in logprobs[0].items():
                if logprob.decoded_token.strip() in valid_tks:
                    prediction = int(logprob.decoded_token.strip())
                    break
        predictions.append(prediction)
    # === SAVE RESULTS ===
    with open(outpath, "wb") as f:
        pickle.dump(predictions, f)
    print(f"saved predictions to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma-2-2b-it")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--max_num_seqs", type=int, default=2048)
    args = parser.parse_args()
    main(args)