import os
from typing import Tuple
from tqdm import trange
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM

from lpaaj.constants import MODELS, RESULTS_DIR
from lpaaj.data import MTBench


HF_HOME = os.getenv("HF_HOME", None)

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_name],
        torch_dtype=t.bfloat16,
        device_map="auto",
        cache_dir=HF_HOME,
        trust_remote_code=True
    ); model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS[model_name],
        cache_dir=HF_HOME
    )
    return model, tokenizer


def harvest_all(model_name: str, contrast_choice: str) -> None:
    outpath = f"{RESULTS_DIR}/mtbench/{model_name}/contrast_{contrast_choice}_all.pt"
    if os.path.exists(outpath):
        print("results already exist at ", outpath)
        return
    model, tokenizer = load_model_and_tokenizer(model_name)
    dataset = MTBench(
        task="contrast",
        contrast_choice="1"
    )
    dataset.preprocess_prompts(tokenizer)
    harvest = []
    for idx in trange(len(dataset)):
        prompt = dataset.prompts[idx]
        tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with t.inference_mode(): 
            out = model(**tks, output_hidden_states=True, use_cache=False)
            activations = t.cat(out["hidden_states"], dim=0)[:, -1, :].cpu()
            harvest.append(activations)
    harvest = t.stack(harvest, dim=0)
    t.save(harvest, outpath)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--contrast_choice", type=str, required=True)
    args = parser.parse_args()
    harvest_all(args.model, args.contrast_choice)
