import torch
from tqdm import tqdm
import math
import fast_hadamard_transform

from .hadamard import apply_exact_had_to_linear, get_orthogonal_matrix, matmul_hadU_cuda, hadamard_matrix
from .hadamard_utils import HadamardTransform, get_hadK, is_pow2


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    W = model.model.embed_tokens
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1) -> None:
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(b, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1):
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(b, R1).to(device="cpu", dtype=dtype)


def rotate_R1(model):
    R1 = get_orthogonal_matrix(model.config.hidden_size, "hadamard")

    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating R1")):

        rotate_attention_inputs(layers[idx], R1)
        rotate_attention_output(layers[idx], R1)
        rotate_mlp_input(layers[idx], R1)
        rotate_mlp_output(layers[idx], R1)


def rotate_R2_offline(model):
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating R2")):
        R2 = get_orthogonal_matrix(head_dim, "hadamard")

        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj

        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

        # apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
        # apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)


def rotate_R2_online(x, had_dim):

    init_shape = x.shape

    # x = HadamardTransform.apply(
    #     x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
    # ) / math.sqrt(self.had_dim)

    x = fast_hadamard_transform.hadamard_transform(
        x.reshape(-1, init_shape[-1] // had_dim, had_dim).transpose(1, 2),
        scale=1 / math.sqrt(init_shape[-1] // had_dim),
    ).transpose(1, 2)

    # x = fast_hadamard_transform.hadamard_transform(
    #     x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim),
    #     scale=1 / math.sqrt(self.had_dim),
    # ).reshape(init_shape)

    x = x.reshape(init_shape)

    return x


def rotate_R3_online(x):
    x_dtype = x.dtype
    # x = (HadamardTransform.apply(x.float()) / math.sqrt(x.shape[-1])).to(x_dtype)
    x = HadamardTransform.apply(x.contiguous()) / torch.tensor(x.shape[-1]).sqrt().to(x_dtype)
    return x


def rotate_R4_offline(model):
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating R4")):
        W = layer.mlp.dense_4h_to_h
        apply_exact_had_to_linear(W, had_dim=-1, output=False)


def rotate_R4_online(x):

    had_K, K = get_hadK(x.shape[-1])
    x_dtype = x.dtype
    x = matmul_hadU_cuda(x.float(), had_K, K).to(x_dtype)
    
    return x



def rotate_R2_v(x, had_dim):
    
    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    hadK = hadamard_matrix(had_dim, "cuda").to(torch.float64)
    x_shape = x.shape
    x_dtype = x.dtype
    
    x = x.reshape(-1, x.shape[-1] // had_dim, had_dim)
    x = x.to(torch.float64) @ hadK
    x = x.reshape(x_shape).to(x_dtype)
    
    return x


def rotate_R2_o(W, had_dim):

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    hadK = hadamard_matrix(had_dim, "cuda").to(torch.float64)
    
    W_shape = W.shape
    W_dtype = W.dtype
    
    W = W.reshape(-1, W.shape[-1] // had_dim, had_dim)
    W = W.to(torch.float64) @ hadK
    W = W.reshape(W_shape).to(W_dtype)
    
    return W


def rotate_R4_act(x):

    had_K, K = get_hadK(x.shape[-1])
    x_dtype = x.dtype
    x = matmul_hadU_cuda(x.float(), had_K, K).to(x_dtype)
    
    return x

def rotate_R4_weight(W):

    had_K, K = get_hadK(W.shape[-1])
    W_dtype = W.dtype
    W = matmul_hadU_cuda(W.float(), had_K, K).to(W_dtype)
    
    return W
