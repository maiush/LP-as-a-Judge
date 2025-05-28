import os, argparse, pickle

from lpaaj.data import TextDataset, MTBench, LLMBar
from lpaaj.constants import MODEL_PATH, RESULT_PATH

import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def generate_vllm(
        args: argparse.Namespace
) -> None:
    # === ARGS ===
    model_name = args.model
    lora = args.lora
    max_num_seqs = args.max_num_seqs
    max_new_tokens = args.max_new_tokens
    task = args.task
    dataset = args.dataset
    prompt_key = args.prompt_key
    label_key = args.label_key
    reverse = args.reverse
    contrast_choice = args.contrast_choice
    if dataset in ["newsroom", "summeval", "hanna"]:
        dataset_category = "text_quality"
    elif dataset in ["mctaco", "caters", "rocstories"]:
        dataset_category = "common_sense"
    else:
        dataset_category = ""

    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{RESULT_PATH}/{dataset}/{model_name}/"
    if dataset.startswith("llmbar"): dataset, subset = dataset.split("-")
    if dataset in ["mtbench", "llmbar"]: outpath += f"{task}"
    else: outpath += f"{prompt_key}_{task}"
    if reverse: outpath += "_reversed"
    if contrast_choice: outpath += f"_{contrast_choice}.pt"
    else: outpath += ".pkl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return
    else:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # === LOAD TOKENIZER ===
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model_name}", trust_remote_code=True)
    # === LOAD MODEL ===
    llm_kwargs = {
        "model": f"{MODEL_PATH}/{model_name}",
        "gpu_memory_utilization": 0.98,
        "tensor_parallel_size": 2 if model_name == "qwen-2.5-0.5b" else t.cuda.device_count(),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_num_seqs": max_num_seqs,
        "max_model_len": 8192,
        "enable_prefix_caching": True,
        "seed": 123456,
        "task": "embed" if task == "contrast" else "generate",
        "enforce_eager": True
    }
    if args.lora:
        print(f"applying LoRA adapter: {args.lora}")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 8
    model = LLM(**llm_kwargs)
    # === SET SAMPLING PARAMS ===
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        skip_special_tokens=False,
        logprobs=5,
        temperature=0.0
    )

    # === LOAD DATASET ===
    if dataset == "mtbench":
        dataset = MTBench(
            task=task,
            reverse=reverse,
            contrast_choice=contrast_choice
        )
    elif dataset == "llmbar":
        dataset = LLMBar(
            subset=subset,
            task=task,
            reverse=reverse,
            contrast_choice=contrast_choice
        )
    else:
        dataset = TextDataset(
            task=task,
            dataset=dataset,
            prompt_key=prompt_key,
            label_key=label_key,
            reverse=reverse,
        contrast_choice=contrast_choice
    )
    dataset.preprocess_prompts(tokenizer)

    # === GENERATE ===
    if task == "contrast":
        outputs = model.encode(list(dataset.prompts))
    else:
        gen_kwargs = {
            "prompts": list(dataset.prompts),
            "sampling_params": sampling_params,
            "lora_request": LoRARequest("adapter", 1, lora_path=f"{MODEL_PATH}/{model_name}-lora-{dataset_category}") if lora else None,
            "use_tqdm": True,
        }
        outputs = model.generate(**gen_kwargs)

    # === PREDICTIONS ===
    predictions = []
    if task in ["score", "compare"]:
        for output in outputs:
            # grab logits
            valid_tks = ["1", "2"] if task == "score" else ["1", "2", "3", "4", "5"]
            prediction = -1
            logprobs = output.outputs[0].logprobs
            if logprobs:
                for _, logprob in logprobs[0].items():
                    if logprob.decoded_token.strip() in valid_tks:
                        prediction = int(logprob.decoded_token.strip())
                        break
            predictions.append(prediction)
    # harvest activations
    harvest = None
    if task == "contrast":
        harvest = [x.outputs.data for x in outputs]
        assert len(harvest) == len(dataset)
    # save results
    if predictions:
        with open(outpath, "wb") as f:
            pickle.dump(predictions, f)
        print(f"saved predictions to {outpath}")
    if harvest:
        t.save(t.stack(harvest, dim=0), outpath)
        print(f"saved activations to {outpath}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-2.5-0.5b")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--max_num_seqs", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--task", type=str, default="compare")
    parser.add_argument("--dataset", type=str, default="newsroom")
    parser.add_argument("--prompt_key", type=str, default="coherence")
    parser.add_argument("--label_key", type=str, default=None)
    parser.add_argument("--reverse", action="store_true", default=False)
    parser.add_argument("--contrast_choice", type=str, default=None)
    args = parser.parse_args()

    generate_vllm(args)
