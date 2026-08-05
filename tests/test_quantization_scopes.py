import unittest

import torch.nn as nn

from opencood.tools.quantization.fake_quant import W8A8FakeQuantWrapper
from opencood.tools.quantization.quantize_model import (
    apply_w8a8_fake_quant,
    list_quantizable_modules,
    module_set_sha256,
    normalize_scopes,
)


class DummyWhere2Comm(nn.Module):
    def __init__(self):
        super().__init__()
        self.pillar_vfe = nn.Module()
        self.pillar_vfe.pfn_layers = nn.ModuleList([nn.Linear(2, 2)])
        self.backbone = nn.Sequential(nn.Conv2d(1, 1, 1))
        self.shrink_conv = nn.Conv2d(1, 1, 1)
        self.fusion_net = nn.Module()
        self.fusion_net.naive_communication = nn.Module()
        self.fusion_net.naive_communication.gaussian_filter = nn.Conv2d(1, 1, 1)
        self.cls_head = nn.Conv2d(1, 1, 1)
        self.reg_head = nn.Conv2d(1, 1, 1)


BASE_SCOPES = [
    "pillar_vfe",
    "backbone",
    "shrink_compression",
    "comm_confidence",
    "head",
]


def selected_names(model, scopes):
    return {
        row["name"]
        for row in list_quantizable_modules(model, scopes=scopes)
        if row["selected"]
    }


class QuantizationScopeTests(unittest.TestCase):
    def test_normalize_sorts_and_deduplicates(self):
        self.assertEqual(
            normalize_scopes(scopes=["head", "backbone", "head"]),
            ["backbone", "head"],
        )

    def test_full_and_legacy_combined_cannot_be_mixed(self):
        with self.assertRaises(ValueError):
            normalize_scopes(scopes=["full", "head"])
        with self.assertRaises(ValueError):
            normalize_scopes(scopes=["combined", "pillar_vfe"])

    def test_base_scopes_are_disjoint_and_cover_full(self):
        model = DummyWhere2Comm()
        groups = [selected_names(model, [scope]) for scope in BASE_SCOPES]
        for index, group in enumerate(groups):
            for other in groups[index + 1:]:
                self.assertFalse(group & other)
        union = set().union(*groups)
        self.assertEqual(union, selected_names(model, ["full"]))

    def test_apply_reports_deterministic_union_hash(self):
        expected_names = selected_names(
            DummyWhere2Comm(), ["pillar_vfe", "backbone"])
        model = DummyWhere2Comm()
        report = apply_w8a8_fake_quant(
            model, scopes=["pillar_vfe", "backbone"])
        reverse_report = apply_w8a8_fake_quant(
            DummyWhere2Comm(), scopes=["backbone", "pillar_vfe"])

        self.assertEqual(report["scope_components"], ["backbone", "pillar_vfe"])
        self.assertEqual(report["num_quantized"], len(expected_names))
        self.assertEqual(report["module_set_sha256"],
                         module_set_sha256(expected_names))
        self.assertEqual(report["module_set_sha256"],
                         reverse_report["module_set_sha256"])
        wrapped = sum(
            isinstance(module, W8A8FakeQuantWrapper)
            for module in model.modules()
        )
        self.assertEqual(wrapped, len(expected_names))


if __name__ == "__main__":
    unittest.main()
