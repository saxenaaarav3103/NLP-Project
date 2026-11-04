"""Check that the converted Hugging Face model computes exactly what spaCy's encoder computes.

Loads the original weights into the reference implementation (curated-transformers 0.1.1,
the library spaCy uses) and into the converted ``transformers`` RobertaModel, runs both
on the same padded batch and compares every layer's hidden states.

    pip install --no-deps curated-transformers==0.1.1
    python scripts/verify_conversion.py --wheel en_core_web_trf-3.8.0-py3-none-any.whl --hf models/roberta-base-spacy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, RobertaModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_spacy_roberta import PREFIX, read_spacy_transformer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--hf", type=Path, default=Path("models/roberta-base-spacy"))
    args = ap.parse_args()

    from curated_transformers.models.attention import AttentionMask
    from curated_transformers.models.roberta.config import RobertaConfig as CuratedConfig
    from curated_transformers.models.roberta.encoder import RobertaEncoder

    state, _, _ = read_spacy_transformer(args.wheel)
    ref = RobertaEncoder(CuratedConfig(hidden_width=768, intermediate_width=3072, num_attention_heads=12,
                                       num_hidden_layers=12, hidden_act="gelu"))
    ref.load_state_dict({k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)})
    ref.eval()

    tok = AutoTokenizer.from_pretrained(args.hf)
    hf = RobertaModel.from_pretrained(args.hf, add_pooling_layer=False).eval()
    batch = tok(["I got the job!! I'm so excited!", "Oh. My. God.", "What?"], padding=True, return_tensors="pt")
    with torch.no_grad():
        r = ref(batch["input_ids"], AttentionMask(batch["attention_mask"].bool())).all_outputs
        h = hf(**batch, output_hidden_states=True).hidden_states
    mask = batch["attention_mask"].bool()
    worst = 0.0
    for i, (a, b) in enumerate(zip(r, h)):
        diff = (a[mask] - b[mask]).abs().max().item()  # padding positions are irrelevant
        worst = max(worst, diff)
        print(f"layer {i:2d}: max |diff| = {diff:.2e}")
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst {worst:.2e})")


if __name__ == "__main__":
    main()
