#!/usr/bin/env -S uv run --with onnxruntime --with transformers --with numpy
"""Test the exported GLiNER2 ONNX model against expected results."""

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = "/tmp/gliner2-export/models/recognizers/fastino/gliner2-base-v1"

# Special token IDs
E_TOKEN_ID = 128005
P_TOKEN_ID = 128003
SEP_TEXT_ID = 128002

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def build_inputs(tokenizer, text, labels, max_width=8):
    """Build inputs matching the structured schema format."""
    # Build schema tokens: ( [P] entities ( [E] label1 [E] label2 ... ) ) [SEP_TEXT]
    schema_parts = ["(", "[P]", "entities", "("]
    for label in labels:
        schema_parts.extend(["[E]", label])
    schema_parts.extend([")", ")"])
    schema_parts.append("[SEP_TEXT]")

    # Lowercase text and split into words
    words = text.lower().split()

    # Tokenize each part individually (like Python GLiNER2 does)
    all_subwords = []
    word_boundaries = []  # (start_subword_idx, end_subword_idx) for each text word

    # Schema subwords
    for part in schema_parts:
        subs = tokenizer.tokenize(part)
        all_subwords.extend(subs)

    text_start_subword = len(all_subwords)

    # Text subwords with word tracking
    for word in words:
        subs = tokenizer.tokenize(word)
        start = len(all_subwords)
        all_subwords.extend(subs)
        end = len(all_subwords)
        word_boundaries.append((start, end))

    input_ids = tokenizer.convert_tokens_to_ids(all_subwords)
    seq_len = len(input_ids)

    # Build attention mask
    attention_mask = np.ones(seq_len, dtype=np.int64)

    # Build words_mask: >0 for text tokens (word index, 1-indexed)
    words_mask = np.zeros(seq_len, dtype=np.int64)
    for word_idx, (start, end) in enumerate(word_boundaries):
        for pos in range(start, end):
            words_mask[pos] = word_idx + 1

    num_words = len(words)
    num_spans = num_words * max_width

    # Build word-level span indices
    span_idx = np.zeros((num_spans, 2), dtype=np.int64)
    span_mask = np.zeros(num_spans, dtype=np.bool_)
    for w in range(num_words):
        for wi in range(max_width):
            si = w * max_width + wi
            end_word = w + wi
            if end_word < num_words:
                span_idx[si, 0] = w
                span_idx[si, 1] = end_word
                span_mask[si] = True

    return {
        "input_ids": input_ids[np.newaxis, :].astype(np.int64),
        "attention_mask": attention_mask[np.newaxis, :].astype(np.int64),
        "words_mask": words_mask[np.newaxis, :].astype(np.int64),
        "span_idx": span_idx[np.newaxis, :].astype(np.int64),
        "span_mask": span_mask[np.newaxis, :].astype(np.bool_),
    }, words, num_words


def extract_entities(logits, words, labels, max_width, threshold=0.5, text=None):
    """Extract entities from model output logits."""
    # logits shape: [1, num_words, max_width, num_labels]
    logits = logits[0]  # [num_words, max_width, num_labels]
    scores = sigmoid(logits)
    num_words = len(words)

    entities = []
    for w in range(num_words):
        for wi in range(max_width):
            end_word = w + wi
            if end_word >= num_words:
                continue
            for li, label in enumerate(labels):
                score = scores[w, wi, li]
                if score >= threshold:
                    entity_text = " ".join(words[w:end_word + 1])
                    entities.append({
                        "text": entity_text,
                        "label": label,
                        "word_start": w,
                        "word_end": end_word,
                        "score": float(score),
                    })

    # Sort by score descending for flat NER
    entities.sort(key=lambda e: e["score"], reverse=True)

    # Flat NER: remove overlapping entities (keep highest score)
    kept = []
    for ent in entities:
        overlap = False
        for existing in kept:
            if ent["word_start"] <= existing["word_end"] and ent["word_end"] >= existing["word_start"]:
                overlap = True
                break
        if not overlap:
            kept.append(ent)

    # Sort by position
    kept.sort(key=lambda e: e["word_start"])
    return kept


# Load model
print("Loading ONNX model...")
session = ort.InferenceSession(
    f"{MODEL_DIR}/model.onnx", providers=["CPUExecutionProvider"]
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print("Model inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")
print("Model outputs:")
for out in session.get_outputs():
    print(f"  {out.name}: {out.shape} ({out.type})")
print()

# Test 1: Standard NER
texts = [
    "John Smith works at Google in New York.",
    "Apple Inc. was founded by Steve Jobs.",
]
labels = ["person", "organization", "location"]

print("=== Standard NER ===")
print(f"Labels: {labels}")
print()

for text in texts:
    inputs, words, num_words = build_inputs(tokenizer, text, labels)
    print(f"Text: {text!r}")
    print(f"Words: {words}")
    print(f"Input IDs: {inputs['input_ids'][0].tolist()}")
    print(f"Decoded: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())}")

    outputs = session.run(None, inputs)
    logits = outputs[0]
    print(f"Logits shape: {logits.shape}")

    entities = extract_entities(logits, words, labels, max_width=8, text=text)
    print(f"Entities ({len(entities)}):")
    for ent in entities:
        print(f"  - {ent['text']!r} ({ent['label']}) score={ent['score']:.3f}")
    print()

# Test 2: Custom labels
custom_texts = [
    "The iPhone 15 Pro is a great smartphone released in September 2023.",
    "Tesla Model Y is an electric vehicle manufactured by Tesla Inc.",
]
custom_labels = ["product", "company", "date", "vehicle"]

print("=== Custom Labels ===")
print(f"Labels: {custom_labels}")
print()

for text in custom_texts:
    inputs, words, num_words = build_inputs(tokenizer, text, custom_labels)
    outputs = session.run(None, inputs)
    entities = extract_entities(outputs[0], words, custom_labels, max_width=8, text=text)
    print(f"Text: {text!r}")
    print(f"Entities ({len(entities)}):")
    for ent in entities:
        print(f"  - {ent['text']!r} ({ent['label']}) score={ent['score']:.3f}")
    print()
