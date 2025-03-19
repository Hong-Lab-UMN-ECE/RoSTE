import torch
from .hadamard_utils import (
    get_hadK,
    is_pow2,
    matmul_hadU,
    matmul_hadUt,
    matmul_hadU_cuda,
    matmul_hadUt_cuda,
)


def hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.eye(size)
    return matmul_hadU(Q).to(device)


def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def apply_exact_had_to_linear(module, had_dim=-1, output=False, R2=None):
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()

    if module.bias is not None:
        b = module.bias.data
        b = b.float().cuda()

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K).t()
        if not output:
            ###############################################################
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
    else:
        hadK = hadamard_matrix(had_dim, "cuda").to(torch.float64)
        if R2 is not None:
            hadK = R2.to(torch.float64)
        if output:
            ###############################################################
            W_ = W_.t()
            transposed_shape = W_.shape
            temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(transposed_shape).t()
            # W_ = fast_hadamard_transform.hadamard_transform(
            #     W_.reshape(-1, transposed_shape[-1]//had_dim, had_dim),
            #     scale=1.0/math.sqrt(had_dim)
            #     ).reshape(transposed_shape).t()

            if module.bias is not None:
                b_shape = b.shape
                tmp = b.reshape(b_shape[-1] // had_dim, had_dim)
                tmp = tmp.to(torch.float64) @ hadK
                b = tmp.reshape(b_shape)
                # b = fast_hadamard_transform.hadamard_transform(
                #     b.reshape(b_shape[-1]//had_dim, had_dim),
                #     scale=1.0/math.sqrt(had_dim)
                #     ).reshape(b_shape)

        else:
            init_shape = W_.shape
            temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(init_shape)

            # W_ = fast_hadamard_transform.hadamard_transform(
            #     W_.reshape(-1, init_shape[-1]//had_dim, had_dim),
            #     scale=1/math.sqrt(had_dim)
            #     ).reshape(init_shape)

    module.weight.data = W_.to(device=dev, dtype=dtype)
    if module.bias is not None:
        module.bias.data = b.to(device=dev, dtype=dtype)
