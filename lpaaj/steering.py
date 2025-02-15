import os, pickle
from typing import Tuple
from tqdm import trange
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score
from lpaaj.constants import MODELS, RESULTS_DIR
from lpaaj.data import MTBench

HF_HOME = os.getenv("HF_HOME", None)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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

def main(model_name: str, config: str) -> None:
    outpath = f"{RESULTS_DIR}/mtbench/{model_name}/steering-{config}.pkl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return
    dataset = MTBench(
        task="compare",
    )
    model, tokenizer = load_model_and_tokenizer(model_name)
    dataset.preprocess_prompts(tokenizer)
    # probes
    probe_type, probe_layer, steer_layer = config.split("-")
    probe_path = f"{RESULTS_DIR}/mtbench/{model_name}/{probe_type[0]}-probes.pkl"
    probes = pickle.load(open(probe_path, "rb"))
    probes = t.stack([t.Tensor(x).bfloat16() for x in probes], dim=0)
    nlayers = probes.shape[0]
    if probe_layer == "last":
        probes = probes[[-1], :]
        if steer_layer == "all":
            probes = probes.repeat(nlayers, 1)
    # hooks
    # define hook that orthogonalizes against a probe direction
    def get_orthogonalize_hook(probe):
        def hook(module, input, output):
            is_tuple = type(output) == tuple    
            if is_tuple:
                output = output[0]
            x = output[:, -1:, :] 
            normalized_probe = probe.to(x.device) / probe.to(x.device).norm(dim=-1, keepdim=True)
            proj = (x @ normalized_probe) * normalized_probe
            output[:, -1:, :] = x - proj
            if is_tuple:
                return (output,)
            return output
        return hook
    # add hooks starting from second-last layer
    for i in range(len(probes)-1):
        layer_idx = -(i + 2)  
        probe = probes[layer_idx]  
        # transformer block for this layer
        block = model.model.layers[len(probes)+layer_idx]
        # hook after block
        block.register_forward_hook(get_orthogonalize_hook(probe))
    # add last hook
    model.model.norm.register_forward_hook(get_orthogonalize_hook(probes[-1]))
    # predictions
    skipped, predictions = [], []
    one_id = tokenizer.encode("1", add_special_tokens=False)[0]
    two_id = tokenizer.encode("2", add_special_tokens=False)[0]
    for idx in trange(len(dataset), desc="processing predictions"):
        prompt = dataset.prompts[idx]
        # run the prompts
        tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        if tks.input_ids.shape[1] > 2000 and model_name == "llama-3.1-70b":
            skipped.append(idx)
            continue
        with t.inference_mode(): 
            out = model(**tks, use_cache=False)
            logits = out.logits[:, -1, :]
            choice_logits = logits[:, [one_id, two_id]]
            prediction = choice_logits.flatten().argmax().item() + 1
            predictions.append(prediction)
    # calculate f1
    labels = [dataset.labels[i] for i in range(len(dataset)) if i not in skipped]
    score = f1_score(labels, predictions, average="weighted", labels=[1, 2])
    # save results
    out = (labels, predictions, score)
    with open(outpath, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.model, args.config)