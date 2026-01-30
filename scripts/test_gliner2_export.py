#!/usr/bin/env python3
"""
Unit tests for GLiNER2 ONNX export wrapper.

This test verifies the wrapper logic works correctly without needing
to download the actual GLiNER2 model from HuggingFace.

Usage:
    python test_gliner2_export.py
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

# Add the scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_gliner2_onnx import GLiNER2SpanWrapper, create_dummy_inputs


class MockProjection(nn.Module):
    """Mock projection layer (like SpanMarkerV0's project layers)."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear(x)


class MockSpanRepLayer(nn.Module):
    """Mock SpanRepLayer.span_rep_layer that mimics SpanMarkerV0."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.project_start = MockProjection(hidden_size)
        self.project_end = MockProjection(hidden_size)
        self.out_project = nn.Linear(hidden_size * 2, hidden_size)


class MockSpanRep(nn.Module):
    """Mock SpanRep module that contains span_rep_layer."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.span_rep_layer = MockSpanRepLayer(hidden_size)


class MockEncoder(nn.Module):
    """Mock encoder that simulates a DeBERTa-like encoder."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(30000, hidden_size)

        # Add config for the wrapper to read
        class Config:
            hidden_size = 768

        self.config = Config()

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embedding(input_ids)

        class Output:
            def __init__(self, hidden):
                self.last_hidden_state = hidden

        return Output(hidden_states)


class MockClassifier(nn.Module):
    """Mock classifier (hidden -> 1)."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, x):
        return self.classifier(x)


class MockGLiNER2:
    """Mock GLiNER2 model for testing."""

    def __init__(self, hidden_size=768, max_width=8):
        self.hidden_size = hidden_size
        self.max_width = max_width
        self.encoder = MockEncoder(hidden_size=hidden_size)
        self.span_rep = MockSpanRep(hidden_size=hidden_size)
        self.classifier = MockClassifier(hidden_size=hidden_size)


class TestGLiNER2SpanWrapper(unittest.TestCase):
    """Test cases for the GLiNER2 ONNX wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.max_width = 8
        self.hidden_size = 768

        self.mock_model = MockGLiNER2(
            hidden_size=self.hidden_size, max_width=self.max_width
        )
        self.wrapper = GLiNER2SpanWrapper(self.mock_model, max_width=self.max_width)
        self.wrapper.eval()

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 1
        seq_len = 64
        num_tokens = 10
        num_spans = num_tokens * self.max_width

        inputs = create_dummy_inputs(batch_size, seq_len, num_tokens, self.max_width)

        with torch.no_grad():
            output = self.wrapper(**inputs)

        # GLiNER2SpanWrapper outputs [batch, num_spans, 1]
        expected_shape = (batch_size, num_spans, 1)
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {output.shape}",
        )

    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 2, 4]:
            with self.subTest(batch_size=batch_size):
                num_tokens = 10
                inputs = create_dummy_inputs(
                    batch_size, 64, num_tokens, self.max_width
                )

                with torch.no_grad():
                    output = self.wrapper(**inputs)

                self.assertEqual(output.shape[0], batch_size)

    def test_forward_pass_different_num_tokens(self):
        """Test forward pass with different numbers of tokens."""
        for num_tokens in [5, 10, 20]:
            with self.subTest(num_tokens=num_tokens):
                inputs = create_dummy_inputs(1, 64, num_tokens, self.max_width)
                num_spans = num_tokens * self.max_width

                with torch.no_grad():
                    output = self.wrapper(**inputs)

                # Output shape: [batch, num_spans, 1]
                self.assertEqual(output.shape[1], num_spans)
                self.assertEqual(output.shape[2], 1)

    def test_text_offset_calculation(self):
        """Test that text token offset is correctly calculated from words_mask."""
        batch_size = 1
        seq_len = 64
        num_tokens = 10
        text_start = 7  # create_dummy_inputs uses this offset

        inputs = create_dummy_inputs(batch_size, seq_len, num_tokens, self.max_width)

        # Verify words_mask has text tokens starting at position 7
        words_mask = inputs["words_mask"]
        self.assertEqual(words_mask[0, text_start].item(), 1)  # First text token
        self.assertEqual(words_mask[0, text_start - 1].item(), 0)  # Before text

    def test_onnx_export(self):
        """Test that model can be exported to ONNX format."""
        num_tokens = 10
        num_spans = num_tokens * self.max_width
        inputs = create_dummy_inputs(1, 64, num_tokens, self.max_width)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")

            input_names = [
                "input_ids",
                "attention_mask",
                "words_mask",
                "text_lengths",
                "span_idx",
                "span_mask",
            ]

            dynamic_axes = {
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "words_mask": {0: "batch", 1: "seq_len"},
                "text_lengths": {0: "batch"},
                "span_idx": {0: "batch", 1: "num_spans"},
                "span_mask": {0: "batch", 1: "num_spans"},
                "logits": {0: "batch", 1: "num_spans"},
            }

            torch.onnx.export(
                self.wrapper,
                (
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["words_mask"],
                    inputs["text_lengths"],
                    inputs["span_idx"],
                    inputs["span_mask"],
                ),
                onnx_path,
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=17,
                dynamo=False,  # Use TorchScript export
            )

            self.assertTrue(os.path.exists(onnx_path))
            self.assertGreater(os.path.getsize(onnx_path), 0)

    def test_onnx_runtime_inference(self):
        """Test that exported ONNX model works with ONNX Runtime."""
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")

        num_tokens = 10
        num_spans = num_tokens * self.max_width
        inputs = create_dummy_inputs(1, 64, num_tokens, self.max_width)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")

            input_names = [
                "input_ids",
                "attention_mask",
                "words_mask",
                "text_lengths",
                "span_idx",
                "span_mask",
            ]

            torch.onnx.export(
                self.wrapper,
                (
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["words_mask"],
                    inputs["text_lengths"],
                    inputs["span_idx"],
                    inputs["span_mask"],
                ),
                onnx_path,
                input_names=input_names,
                output_names=["logits"],
                opset_version=17,
                dynamo=False,
            )

            # Load with ONNX Runtime
            session = ort.InferenceSession(onnx_path)

            # Get actual input names from the model
            model_input_names = {inp.name for inp in session.get_inputs()}

            # Build inputs dict with only what the model expects
            ort_inputs = {}
            for name in model_input_names:
                if name in inputs:
                    val = inputs[name].numpy()
                    if name == "span_mask":
                        val = val.astype(np.bool_)
                    else:
                        val = val.astype(np.int64)
                    ort_inputs[name] = val

            outputs = session.run(None, ort_inputs)

            self.assertEqual(len(outputs), 1)
            # Output shape: [batch, num_spans, 1]
            self.assertEqual(outputs[0].shape[0], 1)  # batch
            self.assertEqual(outputs[0].shape[-1], 1)  # single score per span


if __name__ == "__main__":
    unittest.main(verbosity=2)
