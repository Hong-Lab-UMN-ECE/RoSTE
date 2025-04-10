import torch
import torch.nn as nn
from torch.autograd import Function
import sys


class QuantizePerTensor(Function):
    @staticmethod
    def forward(ctx, input, num_bits, symmetric, clip_ratio):
        if num_bits >= 16:
            return input
        else:
            if symmetric:
                q_min = -(2 ** (num_bits - 1))
                q_max = (2 ** (num_bits - 1)) - 1
                x_max = input.abs().max() * clip_ratio
                scale = x_max / q_max
                zero_point = 0
            else:
                q_min = 0
                q_max = (2**num_bits) - 1
                x_max = input.max() * clip_ratio
                x_min = input.min() * clip_ratio
                scale = (x_max - x_min) / (q_max - q_min)
                zero_point = torch.round(-x_min / scale)

            quantized = torch.clamp(torch.round(input / scale) + zero_point, q_min, q_max)

            output = (quantized - zero_point) * scale
            return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients unaltered
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class QuantizePerChannel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, symmetric, clip_ratio):
        if num_bits >= 16:
            return input
        else:
            if symmetric:
                q_min = -(2 ** (num_bits - 1))
                q_max = (2 ** (num_bits - 1)) - 1
                x_max = input.abs().max(dim=-1, keepdim=True)[0] * clip_ratio
                scales = x_max / q_max
                zero_points = 0
            else:
                q_min = 0
                q_max = (2**num_bits) - 1
                x_max = input.max(dim=-1, keepdim=True)[0] * clip_ratio
                x_min = input.min(dim=-1, keepdim=True)[0] * clip_ratio
                scales = (x_max - x_min) / (q_max - q_min)
                zero_points = torch.round(-x_min / scales)

            quantized = torch.clamp(
                torch.round(input / scales) + zero_points, q_min, q_max
            )

            output = (quantized - zero_points) * scales
            return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients unaltered
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class QuantizePerToken(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, symmetric, clip_ratio):
        if num_bits >= 16:
            return input
        else:
            if symmetric:
                q_min = -(2 ** (num_bits - 1))
                q_max = (2 ** (num_bits - 1)) - 1
                x_max = input.abs().max(dim=-1, keepdim=True)[0] * clip_ratio
                scales = x_max / q_max
                zero_points = 0
            else:
                q_min = 0
                q_max = (2**num_bits) - 1
                x_max = input.max(dim=-1, keepdim=True)[0] * clip_ratio
                x_min = input.min(dim=-1, keepdim=True)[0] * clip_ratio
                scales = (x_max - x_min) / (q_max - q_min)
                zero_points = torch.round(-x_min / scales)

            quantized = torch.clamp(
                torch.round(input / scales) + zero_points, q_min, q_max
            )

            output = (quantized - zero_points) * scales
            return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients unaltered
        grad_input = grad_output.clone()
        return grad_input, None, None, None

class Quantize_K(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, symmetric, clip_ratio):
        if num_bits >= 16:
            return input
        else:
            if symmetric:
                q_min = -(2 ** (num_bits - 1))
                q_max = (2 ** (num_bits - 1)) - 1
                x_max = input.abs().max(dim=-1, keepdim=True)[0] * clip_ratio
                scales = x_max / q_max
                zero_points = 0
            else:
                q_min = 0
                q_max = (2**num_bits) - 1
                x_max = input.max(dim=-1, keepdim=True)[0] * clip_ratio
                x_min = input.min(dim=-1, keepdim=True)[0] * clip_ratio
                scales = (x_max - x_min) / (q_max - q_min)
                zero_points = torch.round(-x_min / scales)

            quantized = torch.clamp(
                torch.round(input / scales) + zero_points, q_min, q_max
            )

            output = (quantized - zero_points) * scales
            return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients unaltered
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class Quantize_V(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, symmetric, clip_ratio):
        if num_bits >= 16:
            return input
        else:
            if symmetric:
                q_min = -(2 ** (num_bits - 1))
                q_max = (2 ** (num_bits - 1)) - 1
                x_max = input.abs().max(dim=-1, keepdim=True)[0] * clip_ratio
                scales = x_max / q_max
                zero_points = 0
            else:
                q_min = 0
                q_max = (2**num_bits) - 1
                x_max = input.max(dim=-1, keepdim=True)[0] * clip_ratio
                x_min = input.min(dim=-1, keepdim=True)[0] * clip_ratio
                scales = (x_max - x_min) / (q_max - q_min)
                zero_points = torch.round(-x_min / scales)

            quantized = torch.clamp(
                torch.round(input / scales) + zero_points, q_min, q_max
            )

            output = (quantized - zero_points) * scales
            return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients unaltered
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class QuantLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        config=None,
        layer_idx=None,
    ):
        super(QuantLinear, self).__init__(in_features, out_features, bias)

        self.config = config
        self.layer_idx = layer_idx
        self.w_bits = config.quant_config["w_bits"]
        self.a_bits = config.quant_config["a_bits"]
        self.w_quant_type = "per-channel"
        self.a_quant_type = "per-token"
        self.w_sym = config.quant_config["w_sym"]
        self.a_sym = config.quant_config["a_sym"]
        self.w_clip_ratio = config.quant_config["w_clip_ratio"]
        self.a_clip_ratio = config.quant_config["a_clip_ratio"]

        if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            self.head_size = config.hidden_size // config.num_attention_heads

        assert 2 <= self.w_bits <= 16, "Weight bitwidth must be between 2 and 16"
        assert 2 <= self.a_bits <= 16, "Activation bitwidth must be between 2 and 16"
        assert self.w_quant_type in ["per-tensor", "per-channel"]
        assert self.a_quant_type in ["per-tensor", "per-token"]
        assert 0.0 <= self.w_clip_ratio <= 1.0
        assert 0.0 <= self.a_clip_ratio <= 1.0 
        
    def forward(self, x):
        weight = self.weight
        w_q = self._quantize_weights(weight, self.w_bits)
        x_q = self._quantize_activations(x, self.a_bits)

        return nn.functional.linear(x_q, w_q, self.bias)

    def _quantize_weights(self, tensor, num_bits):
        if self.w_quant_type == "per-tensor":
            return QuantizePerTensor.apply(
                tensor, num_bits, self.w_sym, self.w_clip_ratio
            )
        elif self.w_quant_type == "per-channel":
            return QuantizePerChannel.apply(
                tensor, num_bits, self.w_sym, self.w_clip_ratio
            )
        else:
            raise ValueError(
                f"Unsupported quantization type for weights: {self.w_quant_type}"
            )

    def _quantize_activations(self, tensor, num_bits):
        if self.a_quant_type == "per-tensor":
            return QuantizePerTensor.apply(
                tensor, num_bits, self.a_sym, self.a_clip_ratio
            )
        elif self.a_quant_type == "per-token":
            return QuantizePerToken.apply(
                tensor, num_bits, self.a_sym, self.a_clip_ratio
            )
        else:
            raise ValueError(
                f"Unsupported quantization type for activations: {self.a_quant_type}"
            )

    def __repr__(self):
        return (
            f"QuantLinear(in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, w_bits={self.w_bits}, a_bits={self.a_bits}, "
            f"w_quant_type={self.w_quant_type}, w_sym={self.w_sym}, w_clip_ratio={self.w_clip_ratio}, "
            f"a_quant_type={self.a_quant_type}, a_sym={self.a_sym}, a_clip_ratio={self.a_clip_ratio})"
        )




def convert_model_to_quant(
    model,
    w_bits=16,
    a_bits=16,
    w_quant_type="per-channel",
    a_quant_type="per-token",
    w_sym=True,
    a_sym=True,
    w_clip_ratio=1.0,
    a_clip_ratio=1.0,
):
    for name, module in model.named_modules():
        if "mlp" in name.lower() or "attention" in name.lower():
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, nn.Linear):
                    quant_layer = QuantLinear(
                        sub_module.in_features,
                        sub_module.out_features,
                        w_bits=w_bits,
                        a_bits=a_bits,
                        bias=sub_module.bias is not None,
                        w_quant_type=w_quant_type,
                        a_quant_type=a_quant_type,
                        w_sym=w_sym,
                        a_sym=a_sym,
                        w_clip_ratio=w_clip_ratio,
                        a_clip_ratio=a_clip_ratio,
                    )
                    quant_layer.weight = sub_module.weight
                    if sub_module.bias is not None:
                        quant_layer.bias = sub_module.bias
                    setattr(module, sub_name, quant_layer)
    return model