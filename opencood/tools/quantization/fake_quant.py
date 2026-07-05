import torch
import torch.nn as nn
import torch.nn.functional as F


def _symmetric_qparams(x, num_bits=8, eps=1e-8):
    qmax = 2 ** (num_bits - 1) - 1
    max_abs = x.detach().abs().amax()
    scale = max(float(max_abs.cpu()) / qmax, eps)
    return scale, 0, -qmax - 1, qmax


def fake_quant_symmetric(x, num_bits=8):
    if x.numel() == 0:
        return x
    scale, zero_point, qmin, qmax = _symmetric_qparams(
        x, num_bits=num_bits)
    return torch.fake_quantize_per_tensor_affine(
        x,
        scale=scale,
        zero_point=zero_point,
        quant_min=qmin,
        quant_max=qmax,
    )


class W8A8FakeQuantWrapper(nn.Module):
    """Wrap Conv2d or Linear with per-tensor symmetric W8A8 fake quant."""

    def __init__(self, module, weight_bits=8, activation_bits=8):
        super().__init__()
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            raise TypeError(f"Unsupported module type: {type(module)}")
        self.module = module
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits

    def forward(self, x):
        x_q = fake_quant_symmetric(x, self.activation_bits)
        w_q = fake_quant_symmetric(self.module.weight, self.weight_bits)
        bias = self.module.bias

        if isinstance(self.module, nn.Conv2d):
            return F.conv2d(
                x_q,
                w_q,
                bias,
                stride=self.module.stride,
                padding=self.module.padding,
                dilation=self.module.dilation,
                groups=self.module.groups,
            )

        if isinstance(self.module, nn.Linear):
            return F.linear(x_q, w_q, bias)

        raise TypeError(f"Unsupported module type: {type(self.module)}")
