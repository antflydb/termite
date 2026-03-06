#!/usr/bin/env -S uv run --with gliner2 --with torch --with transformers --with urllib3 --with requests
"""Test GLiNER2 model directly in Python to get reference outputs."""

from gliner2 import GLiNER2

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

texts = [
    "John Smith works at Google in New York.",
    "Apple Inc. was founded by Steve Jobs.",
]

labels = ["person", "organization", "location"]

print("=== GLiNER2 Python Reference Output ===")
print(f"Labels: {labels}")
print()

for i, text in enumerate(texts):
    print(f"Text {i}: {text!r}")
    result = model.extract_entities(text, labels, threshold=0.5)
    print(f"  Result type: {type(result)}")
    print(f"  Result: {result}")
    print()

# Also test custom labels
custom_labels = ["product", "company", "date", "vehicle"]
custom_texts = [
    "The iPhone 15 Pro is a great smartphone released in September 2023.",
    "Tesla Model Y is an electric vehicle manufactured by Tesla Inc.",
]

print("=== Custom Labels ===")
print(f"Labels: {custom_labels}")
print()

for i, text in enumerate(custom_texts):
    print(f"Text {i}: {text!r}")
    entities = model.extract_entities(text, custom_labels, threshold=0.5)
    print(f"  Entities ({len(entities)}):")
    for ent in entities:
        if isinstance(ent, dict):
            print(f"    - {ent['text']!r} ({ent['label']}) score={ent['score']:.3f} span=[{ent['start']}:{ent['end']}]")
        elif hasattr(ent, 'text'):
            print(f"    - {ent.text!r} ({ent.label}) score={ent.score:.3f} span=[{ent.start}:{ent.end}]")
        else:
            print(f"    - {ent!r}")
    print()

# Debug: show tokenization details
print("=== Tokenization Debug ===")
tokenizer = model.processor.tokenizer if hasattr(model, 'processor') else model.tokenizer
print(f"Tokenizer type: {type(tokenizer).__name__}")

# Check special token IDs
for tok_name in ['[E]', '[C]', '[R]', '[SEP_TEXT]']:
    tok_id = tokenizer.convert_tokens_to_ids(tok_name)
    print(f"  {tok_name} -> {tok_id}")

# Show how the model tokenizes a prompt + text
text = "John Smith works at Google in New York."
label = "person"
prompt = f"[E]{label}[SEP_TEXT]"
print(f"\nPrompt: {prompt!r}")
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
print(f"Prompt token IDs: {prompt_tokens}")
prompt_decoded = tokenizer.convert_ids_to_tokens(prompt_tokens)
print(f"Prompt tokens decoded: {prompt_decoded}")

# Full sequence
full = f"[E]{label}[SEP_TEXT]{text}"
full_tokens = tokenizer.encode(full, add_special_tokens=True)
print(f"\nFull sequence tokens ({len(full_tokens)}): {full_tokens}")
full_decoded = tokenizer.convert_ids_to_tokens(full_tokens)
print(f"Full decoded: {full_decoded}")
