"""Rebuild a Hugging Face ``roberta-base`` checkpoint from spaCy's ``en_core_web_trf``.

Why this exists: the cloud machine this project was developed on could not reach the
Hugging Face Hub, but it *could* download spaCy's English transformer pipeline from
GitHub. That package ships a full RoBERTa-base encoder (12 layers, 768 hidden, 50,265
BPE vocabulary). This script converts it into the standard ``transformers`` format so
the rest of the project can use ``AutoModel``/``AutoTokenizer`` as usual.

**If you can reach the Hub you do not need this**: just pass ``--base-model roberta-base``
to the training script.

Note: spaCy trained this encoder further on OntoNotes (tagging, parsing, NER), so it is
"roberta-base + extra syntactic training", not bit-identical to the original.

Usage:
    curl -L -o en_core_web_trf-3.8.0-py3-none-any.whl \
      https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl
    python scripts/convert_spacy_roberta.py --wheel en_core_web_trf-3.8.0-py3-none-any.whl --out models/roberta-base-spacy

Safety: the wheel is only *read* (never installed/imported); tensors are loaded with
``torch.load(weights_only=True)``, which cannot execute code.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import msgpack
import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

MEMBER = "en_core_web_trf/en_core_web_trf-3.8.0/transformer/model"
PREFIX = "curated_encoder."


def read_spacy_transformer(wheel: Path):
    """Return (state_dict, vocab, merges) from the serialized spaCy transformer component."""
    raw = zipfile.ZipFile(wheel).read(MEMBER)
    model = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    names = [node["name"] for node in model["nodes"]]
    torch_idx = names.index("pytorch")
    bpe_idx = names.index("byte_bpe_encoder")
    shim = msgpack.unpackb(model["shims"][torch_idx][0], raw=False, strict_map_key=False)
    state = torch.load(io.BytesIO(shim["state"]), map_location="cpu", weights_only=True)
    bpe = msgpack.unpackb(model["attrs"][bpe_idx]["byte_bpe_processor"], raw=False, strict_map_key=False)
    return state, bpe["vocab"], bpe["merges"]


def to_hf_state_dict(state: dict, n_layers: int) -> dict:
    """Rename curated-transformers parameters to Hugging Face RoBERTa names.

    The fused attention projection ``mha.input`` is ``cat([query, key, value])`` along
    dim 0 (see curated-transformers' ``_merge_qkv``), so we split it back into three.
    """
    s = {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}
    out = {
        "embeddings.word_embeddings.weight": s["embeddings.inner.word_embeddings.weight"],
        "embeddings.position_embeddings.weight": s["embeddings.inner.position_embeddings.weight"],
        "embeddings.token_type_embeddings.weight": s["embeddings.inner.token_type_embeddings.weight"],
        "embeddings.LayerNorm.weight": s["embeddings.inner.layer_norm.weight"],
        "embeddings.LayerNorm.bias": s["embeddings.inner.layer_norm.bias"],
    }
    for i in range(n_layers):
        src, dst = f"layers.{i}.", f"encoder.layer.{i}."
        for kind in ("weight", "bias"):
            q, k, v = s[f"{src}mha.input.{kind}"].chunk(3, dim=0)
            out[f"{dst}attention.self.query.{kind}"] = q
            out[f"{dst}attention.self.key.{kind}"] = k
            out[f"{dst}attention.self.value.{kind}"] = v
            out[f"{dst}attention.output.dense.{kind}"] = s[f"{src}mha.output.{kind}"]
            out[f"{dst}attention.output.LayerNorm.{kind}"] = s[f"{src}attn_output_layernorm.{kind}"]
            out[f"{dst}intermediate.dense.{kind}"] = s[f"{src}ffn.intermediate.{kind}"]
            out[f"{dst}output.dense.{kind}"] = s[f"{src}ffn.output.{kind}"]
            out[f"{dst}output.LayerNorm.{kind}"] = s[f"{src}ffn_output_layernorm.{kind}"]
    used = {k for k in s if k.startswith(("embeddings.", "layers."))}
    assert len(used) == len(s), f"unmapped parameters: {sorted(set(s) - used)}"
    return {k: v.contiguous() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/roberta-base-spacy"))
    args = ap.parse_args()

    state, vocab, merges = read_spacy_transformer(args.wheel)
    config = RobertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                           intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1,
                           attention_probs_dropout_prob=0.1, max_position_embeddings=514,
                           type_vocab_size=1, layer_norm_eps=1e-5, pad_token_id=1, bos_token_id=0,
                           eos_token_id=2)
    model = RobertaModel(config, add_pooling_layer=False)
    missing, unexpected = model.load_state_dict(to_hf_state_dict(state, 12), strict=False)
    missing = [m for m in missing if "position_ids" not in m and "token_type_ids" not in m]
    assert not missing and not unexpected, (missing, unexpected)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)

    # Tokenizer: GPT-2 style byte-level BPE (same files as roberta-base on the Hub).
    vocab_file, merges_file = args.out / "vocab.json", args.out / "merges.txt"
    vocab_file.write_text(json.dumps(vocab, ensure_ascii=False))
    merges_file.write_text("#version: 0.2\n" + "\n".join(f"{a} {b}" for a, b in merges) + "\n")
    tokenizer = RobertaTokenizer(vocab=str(vocab_file), merges=str(merges_file))
    tokenizer.model_max_length = 512
    tokenizer.save_pretrained(args.out)

    ids = tokenizer("Hello world")["input_ids"]
    print(f"Saved to {args.out}. 'Hello world' -> {ids} (roberta-base gives [0, 31414, 232, 2])")


if __name__ == "__main__":
    main()
