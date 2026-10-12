#!/usr/bin/env python3
"""
Inject OpenJev decision metadata and systemone Jinja chat template into a Qwen3.5 GGUF model.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

import gguf

DEFAULT_INPUT = Path(r"C:\Users\pverg\.cache\huggingface\hub\models--alibiserikbay--JevK5-GGUF\snapshots\ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a\jevk5-4b-v0.3-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path(r"C:\Users\pverg\.cache\huggingface\hub\models--alibiserikbay--JevK5-GGUF\snapshots\ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a\jevk5-4b-openjev-Q4_K_M.gguf")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
TEMPERATURE = 0.85
TEMPERATURE_NOUL = 1.829074

def build_systemone_template() -> str:
    description = "{{ o.description if o.description is string else o.description | tojson }}"
    state_expr = "{{ state if state is string else state | tojson }}"
    inst_expr = "{{ instructions if instructions is string else instructions | tojson }}"

    option = (
        "{% if type != 'noul' %}{{ o.key }}: {% if o.description %}" + description + "{% endif %}"
        "{% elif o.key == 'true' %}yes: {% if o.description %}" + description + "{% else %}The statement is true.{% endif %}"
        "{% else %}no: {% if o.description %}" + description + "{% else %}The statement is false.{% endif %}{% endif %}"
    )
    images = (
        "{% for image in images %}{{ image }}{% endfor %}"
        "{% if images %}{{ 'The screenshot shows the current screen.\\n' }}{% endif %}"
    )
    tmpl = (
        "{% set letters = '" + LETTERS + "' %}"
        "<|im_start|>user\n" + images + "State:\n" + state_expr + "\n\nQuestion: " + inst_expr
        + "{% if type == 'score' %} Rate along the ordered levels below (lowest first).{% endif %}"
        "{{ '\\nOptions:\\n' }}"
        "{% for o in options %}[{{ letters[loop.index0] }}] " + option + "{{ '\\n' }}{% endfor %}"
        "{{ '\\nAnswer with the letter of the best option only.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
    )
    return tmpl


def inject_metadata(input_file: Path, output_file: Path) -> None:
    print(f"Reading input model: {input_file}")
    reader = gguf.GGUFReader(str(input_file))

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field else "qwen35"
    print(f"Model architecture: {arch}")

    print(f"Preparing output model: {output_file}")
    writer = gguf.GGUFWriter(str(output_file), arch=arch)

    # Copy existing fields
    orig_chat_template = None
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name == gguf.Keys.Tokenizer.CHAT_TEMPLATE:
            orig_chat_template = field.contents()
            continue
        if field.name.startswith(f"{arch}.decision."):
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    # Inject decision metadata
    print("Injecting OpenJev decision parameters...")
    writer.add_decision_type("openjev")
    writer.add_decision_temperature("choice", TEMPERATURE)
    writer.add_decision_temperature("score", TEMPERATURE)
    writer.add_decision_temperature("noul", TEMPERATURE * TEMPERATURE_NOUL)

    # Inject systemone chat template
    print("Injecting tokenizer.chat_template.systemone...")
    sys1_tmpl = build_systemone_template()
    templates = [{"name": "systemone", "template": sys1_tmpl}]
    if orig_chat_template:
        templates.insert(0, {"name": "default", "template": orig_chat_template})
    writer.add_chat_template(templates)

    # Register tensors
    total_bytes = sum(t.n_bytes for t in reader.tensors)
    print(f"Copying {len(reader.tensors)} tensors ({total_bytes / (1024**3):.2f} GB)...")
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    with tqdm(total=total_bytes, unit="byte", unit_scale=True, desc="Writing tensors") as pbar:
        for tensor in reader.tensors:
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
            pbar.update(tensor.n_bytes)

    writer.close()
    print("Done! Metadata successfully injected.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject OpenJev decision metadata into Qwen3.5 GGUF")
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT, help="Source GGUF model")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Destination GGUF model")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input model not found: {args.input}")

    inject_metadata(args.input, args.output)


if __name__ == "__main__":
    main()
