import typing
import torch


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):

    # Embedding fusion
    # for W in [model.model.embed_tokens]:
    #     W_ = W.weight.data.double()
    #     W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = [layer for layer in model.model.layers]

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )
        W_norm = layer.input_layernorm.weight.data
        layer.input_layernorm.weight.data = torch.ones_like(W_norm)
        if hasattr(layer.input_layernorm, "bias"):
            bias_norm = layer.input_layernorm.bias.data
            layer.input_layernorm.bias.data = torch.zeros_like(bias_norm)

        # bake_mean_into_linear(layer.self_attn.o_proj)

        fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])

        W_norm = layer.post_attention_layernorm.weight.data
        layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
        if hasattr(layer.post_attention_layernorm, "bias"):
            bias_norm = layer.post_attention_layernorm.bias.data
            layer.post_attention_layernorm.bias.data = torch.zeros_like(bias_norm)

        # bake_mean_into_linear(layer.mlp.down_proj)

    fuse_ln_linear(
        model.model.norm,
        [model.lm_head],
    )

    W_norm = model.model.norm.weight.data
    model.model.norm.weight.data = torch.ones_like(W_norm)
    if hasattr(model.model.norm, "bias"):
        bias_norm = model.model.norm.bias.data
        model.model.norm.bias.data = torch.zeros_like(bias_norm)
