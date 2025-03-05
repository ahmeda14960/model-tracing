import torch
from transformers import AutoModelForCausalLM

from tracing.utils.llama.model import rotate_model_t5

torch.set_default_dtype(torch.bfloat16)

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")

model_rot_name = "yahma/llama-7b-hf"
model_rotated = AutoModelForCausalLM.from_pretrained(model_rot_name, torch_dtype=torch.bfloat16).to(
    "cuda"
)
rotate_model_t5(model_rotated)
# rotate_model(model_rotated)

# Fixing the layer norms to 1's (HUReF works)
# """
# fix_layer_norm(model)
# fix_layer_norm(model_rotated)
# """

# base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,base_tokenizer)
# dataloader = prepare_hf_dataloader(dataset,1)

# evaluate_model = evaluate(model, dataloader)
# evaluate_rotated = evaluate(model_rotated, dataloader)

# print("outputs are aligned: ")
# print([abs(evaluate_model[i] - evaluate_rotated[i]) <= 0.01 for i in range(len(evaluate_model))])

weights = model.state_dict()
weights_rotated = model_rotated.state_dict()

# model.to('cuda')
# print("invariant 1")
# print(weights['model.embed_tokens.weight']@weights['model.layers.0.self_attn.q_proj.weight'].T@weights['model.layers.0.self_attn.k_proj.weight']@weights['model.embed_tokens.weight'].T)
# print("invariant 2")
# print(weights['model.embed_tokens.weight']@weights['model.layers.0.self_attn.v_proj.weight'].T@weights['model.layers.0.self_attn.o_proj.weight'].T@weights['model.embed_tokens.weight'].T)
# print("invariant 3")
# print(weights['model.embed_tokens.weight']@weights[f'model.layers.0.mlp.up_proj.weight'].T@weights[f'model.layers.0.mlp.down_proj.weight'].T@weights['model.embed_tokens.weight'].T)
# print()
# model.to('cpu')

# model_rotated.to('cuda')
# print("rotated")
# print("invariant 1")
# print(weights_rotated['model.embed_tokens.weight']@weights_rotated['model.layers.0.self_attn.q_proj.weight'].T@weights_rotated['model.layers.0.self_attn.k_proj.weight']@weights_rotated['model.embed_tokens.weight'].T)


# print("invariant 2")
# print(weights_rotated['model.embed_tokens.weight']@weights_rotated['model.layers.0.self_attn.v_proj.weight'].T@weights_rotated['model.layers.0.self_attn.o_proj.weight'].T@weights_rotated['model.embed_tokens.weight'].T)
# print("invariant 3")
# print(weights_rotated['model.embed_tokens.weight']@weights_rotated[f'model.layers.0.mlp.up_proj.weight'].T@weights_rotated[f'model.layers.0.mlp.down_proj.weight'].T@weights_rotated['model.embed_tokens.weight'].T)
# print()
# model_rotated.to('cpu')

# Cosine similarity

print("cosine similarity")

model.to("cuda")
print("invariant 1")
invariant = (
    weights["model.embed_tokens.weight"]
    @ weights["model.layers.0.self_attn.q_proj.weight"].T
    @ weights["model.layers.0.self_attn.k_proj.weight"]
    @ (weights["model.embed_tokens.weight"]).T
)
model.to("cpu")
model_rotated.to("cuda")
invariant_rotated = (
    weights_rotated["model.embed_tokens.weight"]
    @ weights_rotated["model.layers.0.self_attn.q_proj.weight"].T
    @ weights_rotated["model.layers.0.self_attn.k_proj.weight"]
    @ (weights_rotated["model.embed_tokens.weight"]).T
)
model_rotated.to("cpu")
invariant.to("cuda")
invariant_rotated.to("cuda")
invariant = torch.flatten(invariant)
invariant_rotated = torch.flatten(invariant_rotated)
print(
    torch.dot(invariant, invariant_rotated)
    / (torch.norm(invariant) * torch.norm(invariant_rotated))
)

model.to("cuda")
print("invariant 2")
invariant = (
    weights["model.embed_tokens.weight"]
    @ weights["model.layers.0.self_attn.v_proj.weight"].T
    @ weights["model.layers.0.self_attn.o_proj.weight"].T
    @ weights["model.embed_tokens.weight"].T
)
model.to("cpu")
model_rotated.to("cuda")
invariant_rotated = (
    weights_rotated["model.embed_tokens.weight"]
    @ weights_rotated["model.layers.0.self_attn.v_proj.weight"].T
    @ weights_rotated["model.layers.0.self_attn.o_proj.weight"].T
    @ weights_rotated["model.embed_tokens.weight"].T
)
model_rotated.to("cpu")
invariant.to("cuda")
invariant_rotated.to("cuda")
invariant = torch.flatten(invariant)
invariant_rotated = torch.flatten(invariant_rotated)
print(
    torch.dot(invariant, invariant_rotated)
    / (torch.norm(invariant) * torch.norm(invariant_rotated))
)

model.to("cuda")
print("invariant 3")
invariant = (
    weights["model.embed_tokens.weight"]
    @ weights["model.layers.0.mlp.up_proj.weight"].T
    @ weights["model.layers.0.mlp.down_proj.weight"].T
    @ weights["model.embed_tokens.weight"].T
)
model.to("cpu")
model_rotated.to("cuda")
invariant_rotated = (
    weights_rotated["model.embed_tokens.weight"]
    @ weights_rotated["model.layers.0.mlp.up_proj.weight"].T
    @ weights_rotated["model.layers.0.mlp.down_proj.weight"].T
    @ weights_rotated["model.embed_tokens.weight"].T
)
model_rotated.to("cpu")
invariant.to("cuda")
invariant_rotated.to("cuda")
invariant = torch.flatten(invariant)
invariant_rotated = torch.flatten(invariant_rotated)
print(
    torch.dot(invariant, invariant_rotated)
    / (torch.norm(invariant) * torch.norm(invariant_rotated))
)
