#!/usr/bin/env python3
"""
Convert a trained PyTorch custom GPT model (elpis_router.pt) to GGUF format.

Architecture:
  vocab_size=4096, d_model=256, n_layers=4, n_heads=8, d_ff=1024, max_seq_len=256
  RoPE, GELU, Pre-LayerNorm (RMSNorm), per-head attention with out_proj.
"""

import os
import sys
import numpy as np
import torch

# Suppress numpy deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

# Allow loading numpy scalars in torch checkpoints
torch.serialization.add_safe_globals([np._core.multiarray.scalar])

import gguf
from gguf import GGUFWriter, GGUFEndian


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def concat_heads(state_dict, layer, attr, n_heads):
    """Concatenate per-head tensors along axis 1 (the head-dim axis).

    PyTorch stores each head as (d_model // n_heads, d_model).
    Concatenating along axis 1 gives (d_model // n_heads, d_model * n_heads)
    which is NOT what llama.cpp expects.

    Actually, looking at the shapes:
        each head is (32, 256) = (d_model//n_heads, d_model)
    llama.cpp key/query/value expect (n_embd, n_embd/n_head) = (256, 32).

    So we concatenate along axis 0 to get (n_heads * 32, 256) = (256, 256).
    """
    pieces = []
    for h in range(n_heads):
        key = f"transformer.h.{layer}.attn.heads.{h}.{attr}.weight"
        pieces.append(state_dict[key].numpy())
    # Concatenate along axis 0 to get (d_model, d_model)
    result = np.concatenate(pieces, axis=0)
    # Transpose to get (d_model, d_model) -> (d_model, d_model)
    # The llama convention for key/query/value is (n_embd, n_embd/n_head)
    # but our per-head shape is already (d_model//n_heads, d_model).
    # Concat along axis 0 gives (n_heads * d_model//n_heads, d_model) = (256, 256).
    # This is the correct shape for llama.cpp attn.key/query/value.
    return result


def convert_state_dict(state_dict, n_layers, n_heads, d_model):
    """Map PyTorch tensor names -> GGUF names, concatenating per-head tensors."""
    mapped = {}

    # Token embeddings
    mapped["token_embd.weight"] = state_dict["transformer.wte.weight"].numpy()

    # Positional encodings  -- llama.cpp uses this name for RoPE base embeddings
    mapped["blk.0.attn.positional_encodings"] = state_dict["transformer.wpe.weight"].numpy()

    for i in range(n_layers):
        # --- Pre-attention norm ---
        mapped[f"blk.{i}.attn_norm.weight"] = state_dict[f"transformer.h.{i}.ln1.weight"].numpy()
        mapped[f"blk.{i}.attn_norm.bias"] = state_dict[f"transformer.h.{i}.ln1.bias"].numpy()

        # --- Attention: per-head -> concatenated ---
        key_tensor = concat_heads(state_dict, i, "key", n_heads)
        mapped[f"blk.{i}.attn.key.weight"] = key_tensor

        query_tensor = concat_heads(state_dict, i, "query", n_heads)
        mapped[f"blk.{i}.attn.query.weight"] = query_tensor

        value_tensor = concat_heads(state_dict, i, "value", n_heads)
        mapped[f"blk.{i}.attn.value.weight"] = value_tensor

        # --- Attention output projection ---
        mapped[f"blk.{i}.attn.output.weight"] = state_dict[f"transformer.h.{i}.attn.out_proj.weight"].numpy()
        mapped[f"blk.{i}.attn.output.bias"] = state_dict[f"transformer.h.{i}.attn.out_proj.bias"].numpy()

        # --- Post-attention norm ---
        mapped[f"blk.{i}.ffn_norm.weight"] = state_dict[f"transformer.h.{i}.ln2.weight"].numpy()
        mapped[f"blk.{i}.ffn_norm.bias"] = state_dict[f"transformer.h.{i}.ln2.bias"].numpy()

        # --- MLP (FFN) ---
        mapped[f"blk.{i}.ffn_up.weight"] = state_dict[f"transformer.h.{i}.mlp.net.0.weight"].numpy()
        mapped[f"blk.{i}.ffn_up.bias"] = state_dict[f"transformer.h.{i}.mlp.net.0.bias"].numpy()
        mapped[f"blk.{i}.ffn_down.weight"] = state_dict[f"transformer.h.{i}.mlp.net.2.weight"].numpy()
        mapped[f"blk.{i}.ffn_down.bias"] = state_dict[f"transformer.h.{i}.mlp.net.2.bias"].numpy()

    # --- Final norm ---
    mapped["output_norm.weight"] = state_dict["transformer.ln_f.weight"].numpy()
    mapped["output_norm.bias"] = state_dict["transformer.ln_f.bias"].numpy()

    # --- LM head / output projection ---
    mapped["output.weight"] = state_dict["lm_head.weight"].numpy()

    return mapped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checkpoint_path = "/mnt/primesauce/Elpis/models/elpis_router.pt"
    output_path = "/mnt/primesauce/Elpis/models/elpis_router.gguf"

    print(f"Loading checkpoint from {checkpoint_path} ...")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract the state dict (our checkpoint wraps it in a dict)
    if isinstance(state, dict) and "model_state" in state:
        sd = state["model_state"]
    else:
        sd = state

    # Extract model config
    model_config = state.get("model_config", {})
    vocab_size = model_config.get("vocab_size", 4096)
    n_embd = model_config.get("n_embd", 256)
    n_layers = model_config.get("n_layer", 4)
    n_heads = model_config.get("n_head", 8)
    n_ff = model_config.get("d_ff", 1024)
    max_seq_len = model_config.get("block_size", 256)

    print(f"  vocab_size={vocab_size}, n_embd={n_embd}, n_layers={n_layers}, "
          f"n_heads={n_heads}, n_ff={n_ff}, max_seq_len={max_seq_len}")

    # Convert tensor names
    print("Converting tensor names ...")
    mapped_tensors = convert_state_dict(sd, n_layers, n_heads, n_embd)
    print(f"  Converted {len(mapped_tensors)} tensors")

    # Open GGUF writer -- f16 precision
    print(f"Writing GGUF to {output_path} ...")
    w = GGUFWriter(output_path, arch="llama", endianess=GGUFEndian.LITTLE)

    # --- Metadata ---
    w.add_block_count(n_layers)
    w.add_context_length(max_seq_len)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_heads)
    w.add_vocab_size(vocab_size)
    w.add_rope_dimension_count(n_embd // n_heads)  # RoPE head dim
    w.add_rope_freq_base(0.0)  # RoPE base frequency (unknown)
    w.add_layer_norm_rms_eps(1e-5)  # Required for llama.cpp quantizer
    w.add_description("Custom GPT model trained on PR-BC with AdamW optimizer")
    w.add_name("Elpis-R0.1")
    w.add_file_type(1)  # f16

    # --- Add tensors (buffers tensor info) ---
    converted_count = 0
    for gguf_name, np_array in mapped_tensors.items():
        # Ensure f16
        if np_array.dtype != np.float16:
            np_array = np_array.astype(np.float16)
        w.add_tensor(gguf_name, np_array)
        converted_count += 1
        print(f"  {gguf_name}: shape={np_array.shape}, dtype={np_array.dtype}")

    # --- Write everything to disk ---
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # Report
    file_size = os.path.getsize(output_path)
    print(f"\nConversion complete!")
    print(f"  Tensors converted: {converted_count}")
    print(f"  Output file: {output_path}")
    print(f"  File size: {file_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
