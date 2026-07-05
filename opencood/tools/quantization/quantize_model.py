import re

import torch.nn as nn

from .fake_quant import W8A8FakeQuantWrapper


SUPPORTED_TYPES = (nn.Conv2d, nn.Linear)

SCOPE_PATTERNS = {
    "full": [r".*"],
    "pillar_vfe": [
        r"pillar_vfe",
        r"pfn_layers",
    ],
    "backbone": [
        r"backbone",
        r"resnet",
        r"blocks",
        r"deblocks",
    ],
    "shrink_compression": [
        r"shrink_conv",
        r"naive_compressor",
        r"compression",
    ],
    "fusion_attention": [
        r"fusion_net\.fuse_modules",
        r"fusion_net\.fuse_network",
        r"encode_layer",
        r"linear1",
        r"linear2",
        r"attention",
        r"attn",
    ],
    "comm_confidence": [
        r"fusion_net\.naive_communication",
        r"gaussian_filter",
    ],
    "head": [
        r"cls_head",
        r"reg_head",
    ],
    "combined": [
        r"backbone",
        r"resnet",
        r"blocks",
        r"deblocks",
        r"shrink_conv",
        r"naive_compressor",
        r"compression",
        r"cls_head",
        r"reg_head",
    ],
}


def _matches_scope(module_name, scope):
    if scope not in SCOPE_PATTERNS:
        raise ValueError(f"Unknown quant_scope: {scope}")
    return any(re.search(pattern, module_name, re.IGNORECASE)
               for pattern in SCOPE_PATTERNS[scope])


def list_quantizable_modules(model, scope="full"):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, SUPPORTED_TYPES):
            rows.append({
                "name": name,
                "type": module.__class__.__name__,
                "selected": _matches_scope(name, scope),
            })
    return rows


def _set_child_module(root, module_name, new_module):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_w8a8_fake_quant(model, scope="full",
                          weight_bits=8, activation_bits=8):
    selected = []

    # Snapshot first; do not mutate while iterating named_modules().
    candidates = list(model.named_modules())
    for name, module in candidates:
        if name == "":
            continue
        if not isinstance(module, SUPPORTED_TYPES):
            continue
        if not _matches_scope(name, scope):
            continue
        _set_child_module(
            model,
            name,
            W8A8FakeQuantWrapper(
                module,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
            ),
        )
        selected.append(name)

    skipped = []
    for name, module in model.named_modules():
        type_name = module.__class__.__name__.lower()
        if "conv" in type_name and not isinstance(
                module, (nn.Conv2d, W8A8FakeQuantWrapper)):
            skipped.append(f"{name}:{module.__class__.__name__}")

    return {
        "scope": scope,
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "num_quantized": len(selected),
        "quantized_modules": selected,
        "skipped_conv_like_modules": skipped,
    }
