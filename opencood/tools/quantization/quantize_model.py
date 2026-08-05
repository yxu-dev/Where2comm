import hashlib
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

SINGLE_SCOPE_CHOICES = tuple(SCOPE_PATTERNS)
MULTI_SCOPE_CHOICES = tuple(
    scope for scope in SCOPE_PATTERNS if scope != "combined"
)


def normalize_scopes(scope="full", scopes=None):
    if scopes is None:
        components = [scope or "full"]
    else:
        components = list(scopes)

    if not components:
        raise ValueError("At least one quantization scope is required")

    unknown = sorted(set(components) - set(SCOPE_PATTERNS))
    if unknown:
        raise ValueError("Unknown quant_scope(s): {}".format(
            ", ".join(unknown)))

    components = sorted(set(components))
    if len(components) > 1 and "full" in components:
        raise ValueError("full cannot be combined with other quantization scopes")
    if len(components) > 1 and "combined" in components:
        raise ValueError(
            "legacy combined cannot be combined with other quantization scopes")
    return components


def module_set_sha256(module_names):
    payload = "".join("{}\n".format(name)
                      for name in sorted(module_names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _matches_scopes(module_name, scope_components):
    return any(
        re.search(pattern, module_name, re.IGNORECASE)
        for scope in scope_components
        for pattern in SCOPE_PATTERNS[scope]
    )


def list_quantizable_modules(model, scope="full", scopes=None):
    scope_components = normalize_scopes(scope=scope, scopes=scopes)
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, SUPPORTED_TYPES):
            rows.append({
                "name": name,
                "type": module.__class__.__name__,
                "selected": _matches_scopes(name, scope_components),
            })
    return rows


def _set_child_module(root, module_name, new_module):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_w8a8_fake_quant(model, scope="full",
                          weight_bits=8, activation_bits=8, scopes=None):
    scope_components = normalize_scopes(scope=scope, scopes=scopes)
    selected = []

    # Snapshot first; do not mutate while iterating named_modules().
    candidates = list(model.named_modules())
    for name, module in candidates:
        if name == "":
            continue
        if not isinstance(module, SUPPORTED_TYPES):
            continue
        if not _matches_scopes(name, scope_components):
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
        "scope": (scope_components[0] if len(scope_components) == 1
                  else "combined"),
        "scope_components": scope_components,
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "num_quantized": len(selected),
        "quantized_modules": sorted(selected),
        "module_set_sha256": module_set_sha256(selected),
        "skipped_conv_like_modules": skipped,
    }
