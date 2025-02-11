import os, argparse, pickle

from lpaaj.data import TextDataset, MTBench, LLMBar
from lpaaj.constants import MODELS, RESULTS_DIR

import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def generate_vllm(
        args: argparse.Namespace
) -> None:
    model = args.model
    max_num_seqs = args.max_num_seqs
    max_new_tokens = args.max_new_tokens
    task = args.task
    dataset = args.dataset
    prompt_key = args.prompt_key
    label_key = args.label_key
    reverse = args.reverse
    contrast_choice = args.contrast_choice
    # check for existing results
    outpath = f"{RESULTS_DIR}/{dataset}/{model}/"
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
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model], cache_dir=os.getenv("HF_HOME", None))
    # load model
    model = LLM(
        model=MODELS[model],
        dtype="bfloat16",
        tensor_parallel_size = 2 if model == "qwen-2.5-0.5b" else t.cuda.device_count(),
        max_num_seqs=max_num_seqs,
        max_model_len=4096,
        enable_prefix_caching=True,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.98,
        enable_chunked_prefill=True,
        download_dir=os.getenv("HF_HOME", None),
        seed=123456
    )
    # add activation hooks
    transformer = model.llm_engine.model_executor.driver_worker.model_runner.model
    if task == "contrast":
        activations = {}
        def harvest(name):
            def hook(module, input, output):
                acts = activations.get(name, [])
                current_acts = output[0].detach().cpu()
                if current_acts.ndim == 3:
                    current_acts = current_acts[:, -1, :]
                    acts.extend([x for x in current_acts])
                elif current_acts.ndim == 2:
                    current_acts = current_acts[-1, :]
                    acts.extend([current_acts])
                else:
                    raise ValueError(f"unexpected shape for activations: {current_acts.shape}")                
                activations[name] = acts
            return hook
        for name, module in transformer.named_modules():
            name_parts = name.split(".")
            if len(name_parts) == 3 and name_parts[-1].isdigit():
                module.register_forward_hook(harvest(name))
    # sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        skip_special_tokens=False,
        logprobs=5
    )
    # load dataset
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
    # generate
    outputs = model.generate(list(dataset.prompts), sampling_params)
    predictions, harvest = [], []
    for output in outputs:
        # grab logits
        if task in ["score", "compare"]:
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
    if task == "contrast":
        # NOTE: stacking all activations in case we decide we want them at a later date
        # keep in mind the size of these can get quite large if we do so
        all_acts = sorted(activations.items(), key = lambda x: int(x[0].split(".")[-1]))
        all_acts = [t.stack(acts, dim=0) for _, acts in all_acts]
        # getting the last layer (code supports all layers)
        all_acts = all_acts[-1]
        harvest.append(all_acts)
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
    parser.add_argument("--max_num_seqs", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--task", type=str, default="score")
    parser.add_argument("--dataset", type=str, default="newsroom")
    parser.add_argument("--prompt_key", type=str, default="coherence")
    parser.add_argument("--label_key", type=str, default=None)
    parser.add_argument("--reverse", action="store_true", default=False)
    parser.add_argument("--contrast_choice", type=str, default=None)
    args = parser.parse_args()

    generate_vllm(args)
