#!/usr/bin/env python3
"""Convert trained Elpis router (nanoGPT) to GGUF v3 using llama.cpp's GGUFWriter."""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "/home/joe/llama.cpp/gguf-py")
import gguf


def _ensure_np(x):
    """Convert torch.Tensor or numpy array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return x


def convert(ckpt_path: str, out_path: str, outtype: str = "f32"):
    print(f"[1/4] Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, weights_only=False)
    state = ckpt["model_state"]
    cfg = ckpt["model_config"]

    n_embd = cfg["n_embd"]
    n_layer = cfg["n_layer"]
    n_head = cfg["n_head"]
    vocab_size = cfg["vocab_size"]
    block_size = cfg["block_size"]
    d_ff = cfg["d_ff"]

    print(f"  Config: n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}, "
          f"vocab={vocab_size}, block_size={block_size}, d_ff={d_ff}")

    if outtype == "f16":
        wtype = gguf.GGMLQuantizationType.F16
    elif outtype == "q4_k_m":
        wtype = gguf.GGMLQuantizationType.Q4_K_M
    else:
        wtype = gguf.GGMLQuantizationType.F32

    print("[2/4] Building GGUF writer...")
    writer = gguf.GGUFWriter(out_path, "elpis_router", endianess=gguf.GGUFEndian.LITTLE)

    print("  Adding tensors...")

    def add(name, tensor, dtype=None):
        writer.add_tensor(name, _ensure_np(tensor), raw_dtype=dtype or wtype)

    add("token_emb.weight", state["transformer.wte.weight"])
    add("block.pos_emb.weight", state["transformer.wpe.weight"],
        gguf.GGMLQuantizationType.F32)

    for i in range(n_layer):
        add(f"block.{i}.attn_norm.weight",
            state[f"transformer.h.{i}.ln1.weight"],
            gguf.GGMLQuantizationType.F32)
        add(f"block.{i}.ffn_norm.weight",
            state[f"transformer.h.{i}.ln2.weight"],
            gguf.GGMLQuantizationType.F32)

        head_keys = sorted([k for k in state.keys()
                           if k.startswith(f"transformer.h.{i}.attn.heads.")])
        q = np.concatenate([_ensure_np(state[k]) for k in head_keys if k.endswith(".query.weight")], axis=0)
        k = np.concatenate([_ensure_np(state[k]) for k in head_keys if k.endswith(".key.weight")], axis=0)
        v = np.concatenate([_ensure_np(state[k]) for k in head_keys if k.endswith(".value.weight")], axis=0)
        add(f"block.{i}.attn_qkv.weight", np.concatenate([q, k, v], axis=0))
        add(f"block.{i}.attn_output.weight",
            state[f"transformer.h.{i}.attn.out_proj.weight"])
        add(f"block.{i}.ffn_gate.weight",
            state[f"transformer.h.{i}.mlp.net.0.weight"])
        add(f"block.{i}.ffn_gate.bias",
            state[f"transformer.h.{i}.mlp.net.0.bias"],
            gguf.GGMLQuantizationType.F32)
        add(f"block.{i}.ffn_down.weight",
            state[f"transformer.h.{i}.mlp.net.2.weight"])
        add(f"block.{i}.ffn_down.bias",
            state[f"transformer.h.{i}.mlp.net.2.bias"],
            gguf.GGMLQuantizationType.F32)

    add("norm.weight", state["transformer.ln_f.weight"],
        gguf.GGMLQuantizationType.F32)
    add("norm.bias", state["transformer.ln_f.bias"],
        gguf.GGMLQuantizationType.F32)
    add("output.weight", state["lm_head.weight"])

    print("  Adding KV metadata...")
    writer.add_uint32("general.alignment", 32)
    writer.add_float32("elpis.embedding_length", n_embd)
    writer.add_uint32("elpis.block_count", n_layer)
    writer.add_uint32("elpis.attention.head_count", n_head)
    writer.add_uint32("elpis.attention.head_count_kv", n_head)
    writer.add_float32("elpis.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_uint32("elpis.feed_forward_length", d_ff)
    writer.add_uint32("elpis.context_length", block_size)
    writer.add_uint32("elpis.token_count", vocab_size)

    # ── Add llama-compatible tokenizer metadata ────────────────
    # The model uses ASCII char-level tokenizer (95 chars + newline = 96 tokens)
    # Map: 0→pad, 1→' ', 2→'!', ..., 94→'~', 95→newline, 96→EOF
    print("  Adding tokenizer metadata...")

    # Build the vocab: 95 printable ASCII + newline + pad
    # GGUF llama tokenizer uses token type + text
    vocab_entries = []
    token_types = []  # 0=normal, 1=unknown, 2=control, 3=user_defined, 4=unused
    for i in range(95):
        vocab_entries.append(chr(i + 32).encode("utf-8"))
        token_types.append(0)  # normal
    # token 95 = newline
    vocab_entries.append(b"\n")
    token_types.append(0)  # normal
    # token 96 = pad/unknown
    vocab_entries.append(b"<pad>")
    token_types.append(2)  # control

    # Write tokenizer vocab to GGUF as a bytes list
    writer.add_bytes_list("tokenizer.ggml.tokens", vocab_entries)
    # Token types: 0=normal for printable chars and newline, 2=control for <pad>
    writer.add_i32_array("tokenizer.ggml.token_type", token_types)
    # Merges (empty for char-level tokenizer)
    writer.add_bytes_list("tokenizer.ggml.merges", [])
    # BOS token (pad = 0)
    writer.add_uint32("tokenizer.ggml.bos_token_id", 0)
    # EOS token (pad = 0, same as BOS for this model)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 0)
    # Unknown token = pad = 0
    writer.add_uint32("tokenizer.ggml.unknown_token_id", 0)
    # Add general llama KV entries for compatibility
    writer.add_string("general.architecture", "llama")
    writer.add_string("general.name", "elpis_router")
    writer.add_string("general.description", "Custom GPT model trained on PR-BC with Muon optimizer")
    # rope settings
    writer.add_float32("llama.rope.freq_base", 0.0)
    # File type: 1=f16
    writer.add_uint32("general.file_type", 1)

    print("  Writing header...")
    writer.write_header_to_file(path=Path(out_path))

    print("  Writing KV data...")
    writer.write_kv_data_to_file()

    print("  Writing tensor data...")
    writer.write_tensors_to_file()
    writer.close()

    size = Path(out_path).stat().st_size
    print(f"[3/4] GGUF saved: {size/1024/1024:.1f} MB")
    return out_path


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/elpis_router.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "models/elpis_router.gguf"
    outtype = sys.argv[3] if len(sys.argv) > 3 else "f32"
    convert(ckpt, out, outtype)
