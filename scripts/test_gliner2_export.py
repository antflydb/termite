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

from export_gliner2_onnx import GLiNER2ONNXWrapper


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


class MockSpanRepLayer(nn.Module):
    """Mock span representation layer."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, start_reps, end_reps):
        combined = torch.cat([start_reps, end_reps], dim=-1)
        return self.proj(combined)


class MockEntityClassifier(nn.Module):
    """Mock entity classifier."""

    def __init__(self, hidden_size=768, num_labels=5):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, span_reps):
        return self.classifier(span_reps)


class MockGLiNER2:
    """Mock GLiNER2 model for testing."""

    def __init__(self, hidden_size=768, num_labels=5):
        self.model = type(
            "Model",
            (),
            {
                "encoder": MockEncoder(hidden_size=hidden_size),
                "span_rep_layer": MockSpanRepLayer(hidden_size=hidden_size),
                "entity_classifier": MockEntityClassifier(
                    hidden_size=hidden_size, num_labels=num_labels
                ),
            },
        )()


class TestGLiNER2ONNXWrapper(unittest.TestCase):
    """Test cases for the GLiNER2 ONNX wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.max_width = 12
        self.num_labels = 5
        self.hidden_size = 768

        self.mock_model = MockGLiNER2(
            hidden_size=self.hidden_size, num_labels=self.num_labels
        )
        self.wrapper = GLiNER2ONNXWrapper(self.mock_model, max_width=self.max_width)
        self.wrapper.eval()

    def _create_test_inputs(self, batch_size=1, seq_len=64, num_tokens=10):
        """Create test inputs for the wrapper."""
        num_spans = num_tokens * self.max_width

        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        words_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        text_lengths = torch.tensor([[num_tokens]], dtype=torch.long)
        span_idx = torch.zeros(batch_size, num_spans, 2, dtype=torch.long)
        span_mask = torch.ones(batch_size, num_spans, dtype=torch.bool)

        # Build valid span indices
        for t in range(num_tokens):
            for w in range(self.max_width):
                idx = t * self.max_width + w
                span_idx[0, idx, 0] = t
                span_idx[0, idx, 1] = min(t + w, num_tokens - 1)

        return input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 1
        seq_len = 64
        num_tokens = 10

        inputs = self._create_test_inputs(batch_size, seq_len, num_tokens)

        with torch.no_grad():
            output = self.wrapper(*inputs)

        expected_shape = (batch_size, num_tokens, self.max_width, self.num_labels)
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {output.shape}",
        )

    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 2, 4]:
            with self.subTest(batch_size=batch_size):
                inputs = self._create_test_inputs(batch_size=batch_size, num_tokens=10)

                with torch.no_grad():
                    output = self.wrapper(*inputs)

                self.assertEqual(output.shape[0], batch_size)

    def test_forward_pass_different_num_tokens(self):
        """Test forward pass with different numbers of tokens."""
        for num_tokens in [5, 10, 20]:
            with self.subTest(num_tokens=num_tokens):
                inputs = self._create_test_inputs(num_tokens=num_tokens)

                with torch.no_grad():
                    output = self.wrapper(*inputs)

                self.assertEqual(output.shape[1], num_tokens)
                self.assertEqual(output.shape[2], self.max_width)

    def test_onnx_export(self):
        """Test that model can be exported to ONNX format."""
        inputs = self._create_test_inputs()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")

            dynamic_axes = {
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "words_mask": {0: "batch", 1: "seq_len"},
                "text_lengths": {0: "batch"},
                "span_idx": {0: "batch", 1: "num_spans"},
                "span_mask": {0: "batch", 1: "num_spans"},
                "logits": {0: "batch", 1: "num_tokens", 2: "max_width", 3: "num_labels"},
            }

            torch.onnx.export(
                self.wrapper,
                inputs,
                onnx_path,
                input_names=[
                    "input_ids",
                    "attention_mask",
                    "words_mask",
                    "text_lengths",
                    "span_idx",
                    "span_mask",
                ],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=14,
            )

            self.assertTrue(os.path.exists(onnx_path))
            self.assertGreater(os.path.getsize(onnx_path), 0)

    def test_onnx_runtime_inference(self):
        """Test that exported ONNX model works with ONNX Runtime."""
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")

        inputs = self._create_test_inputs()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")

            dynamic_axes = {
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "words_mask": {0: "batch", 1: "seq_len"},
                "text_lengths": {0: "batch"},
                "span_idx": {0: "batch", 1: "num_spans"},
                "span_mask": {0: "batch", 1: "num_spans"},
                "logits": {0: "batch", 1: "num_tokens", 2: "max_width", 3: "num_labels"},
            }

            torch.onnx.export(
                self.wrapper,
                inputs,
                onnx_path,
                input_names=[
                    "input_ids",
                    "attention_mask",
                    "words_mask",
                    "text_lengths",
                    "span_idx",
                    "span_mask",
                ],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=14,
            )

            # Load with ONNX Runtime
            session = ort.InferenceSession(onnx_path)

            # Run inference
            ort_inputs = {
                "input_ids": inputs[0].numpy().astype(np.int64),
                "attention_mask": inputs[1].numpy().astype(np.int64),
                "words_mask": inputs[2].numpy().astype(np.int64),
                "text_lengths": inputs[3].numpy().astype(np.int64),
                "span_idx": inputs[4].numpy().astype(np.int64),
                "span_mask": inputs[5].numpy().astype(np.bool_),
            }

            outputs = session.run(None, ort_inputs)

            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0].shape, (1, 10, 12, 5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
