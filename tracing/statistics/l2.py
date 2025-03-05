import torch


def statistic(base_model, ft_model):
    return calculate_l2_distance(base_model, ft_model)


def calculate_l2_distance(model1, model2):
    total_squared_diff = 0
    num_layers = 0

    all_layers = []

    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2:
            raise ValueError(f"Model parameter names do not match: {name1} != {name2}")
        elif param1.shape != param2.shape:
            if name1 == "model.embed_tokens.weight" or name1 == "lm_head.weight":
                print(
                    f"Skipping {name1} because of shape mismatch: {param1.shape} != {param2.shape}"
                )
                continue
            raise ValueError(
                f"Model parameter shapes do not match for {name1}: {param1.shape} != {param2.shape}"
            )

        l2_diff = torch.sum((param1 - param2) ** 2) ** 0.5
        total_squared_diff += l2_diff.item()
        all_layers.append(l2_diff.item())
        num_layers += 1

    avg_l2_distance = total_squared_diff / num_layers
    # print(avg_l2_distance)
    # print(all_layers)
    return avg_l2_distance
