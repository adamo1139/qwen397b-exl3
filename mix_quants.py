#!/usr/bin/env python3
"""
Mix multiple quantized models into one based on a YAML recipe.

This script combines tensors from different quantized versions of the same model
(e.g., 2bpw, 3bpw, 4bpw, 5bpw EXL3 quants) according to pattern-matching rules.

The recipe YAML specifies:
- sources: List of model directories with their IDs
- settings: Configuration for output (shard size, default source, etc.)
- overrides: Pattern-based rules to select which source to use for each tensor

Example usage:
    python mix_quants.py --recipe hermes-override-mixed-3.5bpw.yaml --out /path/to/output
"""
import argparse
import fnmatch
import json
import re
import shutil
import struct
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from safetensors import safe_open
from safetensors.torch import save_file


_RANGE_RE = re.compile(r"(\d+)-(\d+)")


def expand_ranges(pattern: str) -> list[str]:
    """
    Expand integer-range tokens in an fnmatch pattern into a list of patterns.

    fnmatch has no native support for `<lo>-<hi>` numeric ranges (it treats
    `0-1` as the literal string "0-1"). This walks the pattern left-to-right
    and, for each `\\d+-\\d+` token, recursively expands it into one pattern
    per integer in [lo, hi] inclusive.

    Examples:
        "layers.0-1.self_attn.*"  -> ["layers.0.self_attn.*", "layers.1.self_attn.*"]
        "layers.7-10.mlp.*"       -> ["layers.7.mlp.*", "layers.8.mlp.*",
                                      "layers.9.mlp.*", "layers.10.mlp.*"]
        "layers.5.self_attn.*"    -> ["layers.5.self_attn.*"]   (no range)

    A pattern with multiple ranges (rare in practice) gets the cartesian
    product expanded.
    """
    m = _RANGE_RE.search(pattern)
    if not m:
        return [pattern]
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        raise ValueError(f"Invalid range in pattern {pattern!r}: {lo}-{hi}")
    head = pattern[:m.start()]
    tail = pattern[m.end():]
    out = []
    for n in range(lo, hi + 1):
        out.extend(expand_ranges(f"{head}{n}{tail}"))
    return out


def load_recipe(path: Path):
    """
    Load and parse the YAML recipe file.

    Reads a YAML file containing:
    - sources: List of {id, model_dir} pairs defining available quantized models
    - settings: Optional configuration (default_source, template_source, shard_size_mb,
                bits_label, regen_quant_config)
    - overrides: List of {key, source} pattern rules for tensor assignment

    Args:
        path: Path to the YAML recipe file

    Returns:
        Tuple of (sources dict, overrides list, settings dict)
        - sources: {id: Path} mapping source IDs to resolved model directories
        - overrides: List of pattern rules from the recipe
        - settings: Dict with default_source, template_source, shard_size_mb,
                   bits_label, regen_quant_config
    """
    with path.open("r") as f:
        recipe = yaml.safe_load(f)

    sources = {
        int(item["id"]): Path(item["model_dir"]).expanduser().resolve()
        for item in recipe["sources"]
    }

    settings = recipe.get("settings", {})
    return sources, recipe.get("overrides", []), {
        "default_source": settings.get("default_source", 3),
        "template_source": settings.get("template_source", 3),
        "shard_size_mb": settings.get("shard_size_mb", 4096),
        "bits_label": settings.get("bits_label"),
        "regen_quant_config": settings.get("regen_quant_config", True),
        "num_workers": settings.get("num_workers", 8),
    }


def build_index(model_dir: Path) -> dict[str, Path]:
    """
    Build an index mapping tensor keys to their file locations.

    Scans the model directory to create a lookup table for all tensors.
    First checks for model.safetensors.index.json (standard HF format).
    If no index exists, scans all .safetensors files directly.

    Args:
        model_dir: Path to the model directory containing safetensors files

    Returns:
        Dict mapping tensor key names (e.g., "model.layers.0.self_attn.q_proj.weight")
        to the Path of the file containing that tensor
    """
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with index_file.open("r") as f:
            data = json.load(f)
        return {k: model_dir / v for k, v in data["weight_map"].items()}

    key_to_file = {}
    for file in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as st:
            for key in st.keys():
                key_to_file[key] = file
    return key_to_file


def get_tensor(index: dict[str, Path], key: str):
    """
    Load a single tensor from an indexed model directory.

    Opens the safetensors file containing the specified key and extracts
    the tensor. The tensor is returned on CPU in PyTorch format.

    Args:
        index: Tensor index dict from build_index()
        key: Tensor key name to retrieve

    Returns:
        PyTorch tensor loaded from the safetensors file
    """
    with safe_open(str(index[key]), framework="pt", device="cpu") as st:
        return st.get_tensor(key)


def compile_overrides(overrides: list) -> list[tuple[list[str], int]]:
    """
    Pre-expand override rules into (patterns, source_id) tuples so we don't
    re-expand numeric ranges on every tensor lookup.

    Returns a list mirroring `overrides`, where each rule's pattern is
    replaced by a list of expanded patterns.
    """
    return [(expand_ranges(rule["key"]), int(rule["source"])) for rule in overrides]


def match_source(key: str, default: int, compiled: list[tuple[list[str], int]]) -> int:
    """
    Determine which source model to use for a given tensor key.

    Applies pattern-matching rules in order; later matching rules override
    earlier ones (last match wins). Uses Unix shell-style wildcards
    (`*`, `?`) plus numeric range expansion (see `expand_ranges`).

    Args:
        key: Tensor key name (e.g., "model.language_model.layers.5.mlp.up_proj.weight")
        default: Default source ID to use if no patterns match
        compiled: Output of `compile_overrides(...)`

    Returns:
        Source ID (int) to use for this tensor.

    Example patterns:
        "model.language_model.layers.*.self_attn.q_proj.*" - all Q projections
        "model.language_model.layers.5.*"      - all tensors in layer 5
        "model.language_model.layers.0-1.*"    - layers 0 and 1 (range expansion)
        "model.language_model.layers.7-10.*"   - layers 7-10 (range expansion)
    """
    source = default
    for patterns, sid in compiled:
        for pat in patterns:
            if fnmatch.fnmatchcase(key, pat):
                source = sid
                break
    return source


def copy_config_files(template_dir: Path, out_dir: Path):
    """
    Copy non-tensor configuration files from template model to output.

    Copies tokenizer, config, and other metadata files while skipping:
    - Tensor files (.safetensors, .bin, .pt, .pth, .ckpt)
    - Index files (model.safetensors.index.json - regenerated)
    - Quant config (quantization_config.json - regenerated if enabled)

    Args:
        template_dir: Source model directory to copy from
        out_dir: Destination directory for the mixed model
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = {"model.safetensors.index.json", "quantization_config.json"}
    skip_ext = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}

    for item in template_dir.iterdir():
        if item.is_file() and item.name not in skip and not any(item.name.endswith(e) for e in skip_ext):
            shutil.copy2(item, out_dir / item.name)


def _read_safetensors_header(path: str) -> dict:
    """Read just the JSON header of a safetensors file (no tensor data loaded)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len))


def _write_shard_worker(out_path: str, key_to_path: dict[str, str]) -> str:
    """
    Worker process: read tensors from their source files and save one shard.

    Groups keys by source path so each source file is opened once per shard.
    Returns the shard filename (basename) for the parent to record in the index.
    """
    by_path: dict[str, list[str]] = defaultdict(list)
    for k, p in key_to_path.items():
        by_path[p].append(k)

    tensors = {}
    for path, keys in by_path.items():
        with safe_open(path, framework="pt", device="cpu") as st:
            for k in keys:
                tensors[k] = st.get_tensor(k).contiguous()

    save_file(tensors, out_path)
    return Path(out_path).name


def write_shards(out_dir: Path, assignments: dict, indexes: dict, shard_bytes: int, num_workers: int = 8):
    """
    Write tensors to sharded safetensors files using parallel worker processes.

    Phases:
        1. Read safetensors headers (fast, no tensor data) to get exact byte sizes.
        2. Plan shard composition deterministically (sorted key order, fill to shard_bytes).
        3. Dispatch each shard to a worker process; workers read+save independently.

    Each tensor is read exactly once. Final shard count is known up front, so
    filenames are written with the correct total — no rename pass needed.

    Args:
        out_dir: Output directory for shard files
        assignments: Dict mapping tensor keys to source IDs
        indexes: Dict mapping source IDs to their tensor indexes
        shard_bytes: Maximum bytes per shard file
        num_workers: Number of worker processes (each writes one shard at a time)

    Returns:
        Total size in bytes of all tensors written
    """
    keys_sorted = sorted(assignments.keys())

    file_to_keys: dict[str, list[str]] = defaultdict(list)
    for k in keys_sorted:
        file_to_keys[str(indexes[assignments[k]][k])].append(k)

    sizes: dict[str, int] = {}
    for path, keys in file_to_keys.items():
        header = _read_safetensors_header(path)
        for k in keys:
            start, end = header[k]["data_offsets"]
            sizes[k] = end - start

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for k in keys_sorted:
        nbytes = sizes[k]
        if current and current_bytes + nbytes > shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(k)
        current_bytes += nbytes
    if current:
        shards.append(current)

    total_shards = len(shards)
    total_size = sum(sizes.values())
    weight_map: dict[str, str] = {}

    print(f"  Planned {total_shards} shards; writing with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = []
        for i, shard_keys in enumerate(shards, 1):
            filename = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            out_path = str(out_dir / filename)
            key_to_path = {k: str(indexes[assignments[k]][k]) for k in shard_keys}
            fut = pool.submit(_write_shard_worker, out_path, key_to_path)
            futures.append((shard_keys, filename, fut))

        for shard_keys, filename, fut in futures:
            fut.result()
            for k in shard_keys:
                weight_map[k] = filename

    with (out_dir / "model.safetensors.index.json").open("w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

    return total_size


def update_config(out_dir: Path, bits_label: float | None):
    """
    Update config.json with EXL3 quantization metadata.

    Modifies the model's config.json to indicate it uses exl3 quantization.
    Optionally sets the bits field to the specified bitrate label.
    Preserves existing quantization_config fields while updating quant_method.

    Args:
        out_dir: Output directory containing config.json
        bits_label: Optional bitrate value (e.g., 3.5) to write in config
    """
    config_path = out_dir / "config.json"
    if not config_path.exists():
        return

    with config_path.open("r") as f:
        config = json.load(f)

    config["quantization_config"] = {
        "quant_method": "exl3",
        **({"bits": bits_label} if bits_label is not None else {}),
        **config.get("quantization_config", {}),
    }

    with config_path.open("w") as f:
        json.dump(config, f, indent=2)


def regen_quant_config(out_dir: Path):
    """
    Regenerate exllamav3-specific quantization_config.json.

    Uses exllamav3's create_quantization_config_json function to generate
    a detailed quantization config based on the actual tensors in the output.
    This is more accurate than the simple update_config() but requires
    exllamav3 to be installed.

    Args:
        out_dir: Output directory to generate quantization_config.json in
    """
    try:
        from exllamav3.conversion.quant_config import create_quantization_config_json
        create_quantization_config_json(str(out_dir))
        print("Regenerated quantization_config.json")
    except Exception as e:
        print(f"Warning: Could not regenerate quant config: {e}")


def main():
    """
    Main entry point for the model mixing script.

    Orchestrates the full mixing workflow:
    1. Load recipe YAML and validate source paths
    2. Build tensor indexes for all source models
    3. Assign each tensor to a source based on pattern rules
    4. Copy config files from template source
    5. Write mixed tensors to sharded safetensors files
    6. Update config.json with quantization metadata
    7. Optionally regenerate exllamav3 quant config
    """
    parser = argparse.ArgumentParser(description="Mix quantized models from YAML recipe")
    parser.add_argument("--recipe", required=True, type=Path, help="YAML recipe file")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    sources, overrides, settings = load_recipe(args.recipe)

    for sid, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"Source {sid} not found: {path}")

    print("Building tensor indexes...")
    indexes = {sid: build_index(path) for sid, path in sources.items()}

    base_keys = indexes[settings["default_source"]]
    print(f"Default source {settings['default_source']}: {len(base_keys):,} tensors")

    compiled = compile_overrides(overrides)
    assignments = {}
    counts = {}
    for key in base_keys:
        sid = match_source(key, settings["default_source"], compiled)
        assignments[key] = sid
        counts[sid] = counts.get(sid, 0) + 1

    print("Tensor distribution:")
    for sid in sorted(counts):
        print(f"  Source {sid}: {counts[sid]:,}")

    # Sanity check: if (almost) everything went to the default source the
    # recipe's overrides probably aren't matching the real key names. Common
    # causes: numeric-range syntax (`0-1`) without expansion, or wrong key
    # prefix (e.g. `model.language_model.layers.*` against a non-multimodal
    # model whose keys are just `model.layers.*`).
    default_sid = settings["default_source"]
    default_share = counts.get(default_sid, 0) / max(1, len(base_keys))
    if overrides and default_share > 0.95:
        print(
            f"\n  WARNING: {default_share*100:.1f}% of tensors went to the "
            f"default source ({default_sid}). The override rules likely "
            f"aren't matching real tensor keys. Sample real keys:"
        )
        for k in list(base_keys)[:5]:
            print(f"    {k}")
        print(
            "  Compare against your recipe patterns — common pitfalls: wrong "
            "prefix (`language_model.` etc.), or `0-1` range syntax in a "
            "field other than the layer index."
        )

    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f"Output directory not empty: {args.out}")

    print("Copying config files...")
    copy_config_files(sources[settings["template_source"]], args.out)

    print("Writing mixed shards...")
    total = write_shards(args.out, assignments, indexes, settings["shard_size_mb"] * 1024 * 1024, settings["num_workers"])
    print(f"Total: {total:,} bytes")

    update_config(args.out, settings["bits_label"])

    if settings["regen_quant_config"]:
        regen_quant_config(args.out)

    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
