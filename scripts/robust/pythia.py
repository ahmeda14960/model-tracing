import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

import argparse
import pickle
import timeit
import subprocess

from tracing.utils.evaluate import prepare_hf_dataset, prepare_hf_dataloader, evaluate
from tracing.utils.utils import output_hook, get_submodule

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument("--model_id", default="EleutherAI/pythia-1.4b-deduped", type=str)
parser.add_argument("--step", default=0, type=int)
parser.add_argument("--layer", default=10, type=int)

parser.add_argument("--dataset_id", default="dlwh/wikitext_103_detokenized", type=str)
parser.add_argument("--block_size", default=512, type=int)
parser.add_argument("--batch_size", default=6, type=int)

parser.add_argument("--save", default="results.p", type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--token", default="", type=str)

args = parser.parse_args()

from huggingface_hub import login

login(token=args.token)

start = timeit.default_timer()

results = {}
results["args"] = args
results["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

torch.manual_seed(args.seed)

model = GPTNeoXForCausalLM.from_pretrained(
    args.model_id,
    revision=f"step{args.step}",
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_id,
    revision=f"step{args.step}",
)

print("model loaded")

dataset = prepare_hf_dataset(args.dataset_id, args.block_size, tokenizer)
dataloader = prepare_hf_dataloader(dataset, args.batch_size)

print("dataset loaded")

block = get_submodule(model, f"gpt_neox.layers.{args.layer}")

feats, hooks = {}, {}
for layer in [
    "input_layernorm",
    "post_attention_layernorm",
    "mlp.dense_h_to_4h",
    "mlp.dense_4h_to_h",
]:
    hooks[layer] = lambda m, inp, op, layer=layer, feats=feats: output_hook(
        m, inp, op, layer, feats
    )
    get_submodule(block, layer).register_forward_hook(hooks[layer])

print("hooks created")

evaluate(model, dataloader)

print("models evaluated")

end = timeit.default_timer()
results["time"] = end - start

results["weights"] = block.state_dict()
results["feats"] = feats

print(results)
pickle.dump(results, open(args.save, "wb"))
