#!/usr/bin/env python3
"""
Optimize Qwen3.5 MoE expert projection bitrates via greedy sort-and-assign.

Strategy: sort all 171 projections (57 layers x gate/up/down) by |KLD| descending.
The optimal assignment has clean bands — highest-KLD get 5bpw, then 4bpw, 3bpw, 2bpw.

Sweeps n5 to meet the bit budget, picks split with minimum total error.

Fixed components (attention, norms, embeddings, etc.) hardcoded at 5bpw.

QUADRATIC PENALTY: Uses KLD^1.5 to account for non-linear quantization error.
Layers that are hard to quant (high KLD) benefit more from higher bitrates.
"""
import json
from pathlib import Path
from collections import Counter


# Global KLD measurements vs fp16:
#   5bpw=0.0079, 4bpw=0.0203, 3bpw=0.0674, 2bpw=0.516
# Quadratic penalty: raise KLD ratios to power 1.5 to model non-linear quant difficulty
KLD = {5: 0.0079, 4: 0.0203, 3: 0.0674, 2: 0.516}
PENALTY = {b: (kld / KLD[5]) ** 1.5 for b, kld in KLD.items()}
BITRATES = [2, 3, 4, 5]


def parse_kld_table(md_path: Path) -> dict[int, dict[str, float]]:
    kld = {}
    with md_path.open() as f:
        content = f.read()
    in_table = False
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("|") and "gate" in line and "up" in line and "down" in line:
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                continue
            try:
                kld[int(parts[1])] = {
                    "gate": float(parts[2]),
                    "up": float(parts[3]),
                    "down": float(parts[4]),
                }
            except (ValueError, IndexError):
                pass
        elif in_table and not line.startswith("|"):
            break
    return kld


def build_candidates(kld: dict, layer_range: list[int]) -> list[tuple[int, str, float]]:
    """Flatten all projections into (layer, projection, |KLD|) list sorted descending."""
    candidates = []
    for layer in layer_range:
        lk = kld.get(layer)
        if lk is None:
            continue
        for proj in ("gate", "up", "down"):
            candidates.append((layer, proj, abs(lk[proj])))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def optimize(target_avg: float, candidates: list[tuple[int, str, float]]):
    """
    Find optimal four-band split: 5 | 4 | 3 | 2 bpw.

    Sweep n5, then solve for n4, n3, n2 from remaining budget.
    Assignments: top n5→5bpw, next n4→4bpw, next n3→3bpw, rest n2→2bpw.
    """
    N = len(candidates)
    target_bits = target_avg * N

    best_cost = float("inf")
    best_split = None
    best_assignments = None

    max_n5 = min(N, int(target_bits / 5))
    for n5 in range(0, max_n5 + 1):
        # remaining for {4,3,2} bands
        R = N - n5
        Rb = int(round(target_bits - 5 * n5))
        if Rb < 2 * R or Rb > 4 * R:
            continue

        # 4·n4 + 3·n3 + 2·n2 = Rb,  n4 + n3 + n2 = R
        # → n2 = R - n4 - n3
        # → 4n4 + 3n3 + 2(R - n4 - n3) = Rb
        # → 2n4 + n3 = Rb - 2R
        rhs = Rb - 2 * R

        # Sweep n4 (feasible range so n3, n2 stay >= 0)
        # n3 = rhs - 2*n4 >= 0 → n4 <= rhs/2
        # n2 = R - n4 - n3 = R - n4 - (rhs - 2*n4) = R - rhs + n4 >= 0 → n4 >= rhs - R
        n4_min = max(0, int(rhs - R))
        n4_max = min(R, int(rhs // 2))
        if n4_min > n4_max:
            continue
        for n4 in range(n4_min, n4_max + 1):
            n3 = rhs - 2 * n4
            if n3 < 0:
                continue
            n2 = R - n4 - n3
            if n2 < 0:
                continue

            actual = 5 * n5 + 4 * n4 + 3 * n3 + 2 * n2
            if abs(actual / N - target_avg) > 0.005:
                continue

            # Assign: top n5→5, n4→4, n3→3, n2→2
            cost = 0.0
            assignments = {}
            i = 0
            for (layer, proj, sens) in candidates:
                if i < n5:
                    bits = 5
                elif i < n5 + n4:
                    bits = 4
                elif i < n5 + n4 + n3:
                    bits = 3
                else:
                    bits = 2
                cost += sens * PENALTY[bits]
                if layer not in assignments:
                    assignments[layer] = {}
                assignments[layer][proj] = bits
                i += 1

            if cost < best_cost:
                best_cost = cost
                best_split = (n5, n4, n3, n2)
                best_assignments = assignments

    return best_split, best_cost, best_assignments


def generate_yaml(assignments: dict[int, dict[str, int]], fixed_layers: dict, layer_range: list[int]) -> str:
    """Generate CPRAL YAML overrides."""
    lines = [
        "# Optimized using KLD sensitivity (greedy sort-and-assign, 5bpw max)",
        "# Model: Qwen3.5-397B-A17B MoE",
        "",
        "sources:",
        "  - id: 2",
        "    model_dir: .../Nex-N2-Pro-EXL3-2bpw",
        "  - id: 3",
        "    model_dir: .../Nex-N2-Pro-EXL3-3bpw",
        "  - id: 4",
        "    model_dir: .../Nex-N2-Pro-EXL3-4bpw",
        "  - id: 5",
        "    model_dir: .../Nex-N2-Pro-EXL3-5bpw",
        "  - id: 16",
        "    model_dir: .../Nex-N2-Pro",
        "overrides: # last to match applies",
        "",
    ]

    proj_modes = {}
    for p in ("down", "up", "gate"):
        bits_list = [assignments[l][p] for l in layer_range]
        proj_modes[p] = max(set(bits_list), key=bits_list.count)

    for p in ("down", "up", "gate"):
        lines.append(f'  - key: "model.language_model.layers.*.mlp.experts.*.{p}_proj.*"')
        lines.append(f"    source: {proj_modes[p]}")
        lines.append("")

    lines.append("  # Fixed early layers (no KLD data)")
    for layer in sorted(fixed_layers):
        ass = fixed_layers[layer]
        bits_set = set(ass.values())
        if len(bits_set) == 1:
            lines.append(f'  - key: "model.language_model.layers.{layer}.*"')
            lines.append(f"    source: {next(iter(bits_set))}")
        else:
            for p in ("gate", "up", "down"):
                if ass[p] != proj_modes[p]:
                    lines.append(f'  - key: "model.language_model.layers.{layer}.mlp.experts.*.{p}_proj.*"')
                    lines.append(f"    source: {ass[p]}")
        lines.append("")

    lines.append("  # Optimized expert projection overrides (differing from mode)")
    for p in ("down", "up", "gate"):
        mode = proj_modes[p]
        diff_layers = sorted(l for l in layer_range if assignments[l][p] != mode)
        if not diff_layers:
            continue

        ranges = []
        start = diff_layers[0]
        prev_layer = start
        prev_bits = assignments[start][p]
        for l in diff_layers[1:]:
            if l == prev_layer + 1 and assignments[l][p] == prev_bits:
                pass
            else:
                ranges.append((start, prev_layer, prev_bits))
                start = l
                prev_bits = assignments[l][p]
            prev_layer = l
        ranges.append((start, prev_layer, prev_bits))

        for s, e, bits in ranges:
            if s == e:
                lines.append(f'  - key: "model.language_model.layers.{s}.mlp.experts.*.{p}_proj.*"')
            else:
                lines.append(f'  - key: "model.language_model.layers.{s}-{e}.mlp.experts.*.{p}_proj.*"')
            lines.append(f"    source: {bits}")
            lines.append("")

    # Fixed components at 5bpw
    fixed_5 = [
        ("Linear attention", "model.language_model.layers.*.linear_attn.*"),
        ("Full attention", "model.language_model.layers.*.self_attn.*"),
        ("MoE router", "model.language_model.layers.*.mlp.shared_expert_gate.*"),
        ("Shared expert", "model.language_model.layers.*.mlp.shared_expert.*"),
        ("Input LayerNorm", "model.language_model.layers.*.input_layernorm.*"),
        ("Post-Attn LayerNorm", "model.language_model.layers.*.post_attention_layernorm.*"),
        ("Embed Tokens", "model.language_model.embed_tokens.*"),
        ("Final Norm", "model.language_model.norm.*"),
        ("lm_head", "lm_head.*"),
    ]
    for label, key in fixed_5:
        lines.append(f"  # {label}")
        lines.append(f'  - key: "{key}"')
        lines.append("    source: 5")
        lines.append("")

    return "\n".join(lines)


def compute_stats(assignments, fixed_layers, layer_range):
    all_bits = []
    for layer in layer_range:
        a = assignments.get(layer) or {"gate": 5, "up": 5, "down": 5}
        all_bits.extend(a.values())
    for layer in fixed_layers:
        all_bits.extend(fixed_layers[layer].values())

    avg = sum(all_bits) / len(all_bits)
    dist = Counter(all_bits)

    print(f"\n  Distribution ({len(all_bits)} expert projection assignments):")
    for b in sorted(dist):
        print(f"    {b}bpw: {dist[b]:3d} ({dist[b] / len(all_bits) * 100:5.1f}%)")
    print(f"  Expert projection average: {avg:.3f}bpw")

    for p in ("gate", "up", "down"):
        pb = []
        for layer in layer_range:
            a = assignments.get(layer) or {"gate": 5, "up": 5, "down": 5}
            pb.append(a[p])
        for layer in fixed_layers:
            pb.append(fixed_layers[layer][p])
        pavg = sum(pb) / len(pb)
        pd = Counter(pb)
        print(f"    {p}: avg={pavg:.3f}, dist={dict(sorted(pd.items()))}")

    return avg, dist


def main():
    base_dir = Path("/home")

    kld = parse_kld_table(base_dir / "estimated_kld.md")
    print(f"Loaded KLD for {len(kld)} layers ({min(kld)}–{max(kld)})")

    with (base_dir / "qwen_config.json").open() as f:
        config = json.load(f)

    fixed_layers = {
        0: {"gate": 3, "up": 3, "down": 3},
        1: {"gate": 3, "up": 3, "down": 3},
        2: {"gate": 3, "up": 3, "down": 3},
    }
    layer_range = list(range(3, 60))

    candidates = build_candidates(kld, layer_range)
    print(f"Candidates: {len(candidates)} projections from {len(layer_range)} layers")
    print(f"KLD range: [{candidates[-1][2]:.4f}, {candidates[0][2]:.4f}]")
    print(f"PENALTY (quadratic, power=1.5): 2={PENALTY[2]:.4f}, 3={PENALTY[3]:.4f}, 4={PENALTY[4]:.4f}, 5={PENALTY[5]:.4f}")

    targets = [2.2, 2.5, 2.8, 3.0, 3.1, 3.3, 3.35, 3.37, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.5]

    for target in targets:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target}bpw")
        print(f"{'=' * 70}")

        split, cost, assignments = optimize(target, candidates)
        n5, n4, n3, n2 = split
        print(f"  Optimal split: {n5} @ 5bpw, {n4} @ 4bpw, {n3} @ 3bpw, {n2} @ 2bpw")
        print(f"  Total error (KLD-weighted): {cost:.4f}")

        # Show which projections got 5bpw and 2bpw
        if n5 > 0:
            print(f"\n  Projections assigned 5bpw:")
            for layer in sorted(assignments):
                for proj in ("gate", "up", "down"):
                    if assignments[layer][proj] == 5:
                        lk = kld.get(layer, {})
                        print(f"    L{layer:2d} {proj}: KLD={abs(lk.get(proj, 0)):.4f}")
        if n2 > 0:
            print(f"\n  Projections assigned 2bpw (lowest KLD):")
            for layer in sorted(assignments):
                for proj in ("gate", "up", "down"):
                    if assignments[layer][proj] == 2:
                        lk = kld.get(layer, {})
                        print(f"    L{layer:2d} {proj}: KLD={abs(lk.get(proj, 0)):.4f}")

        avg, dist = compute_stats(assignments, fixed_layers, layer_range)

        yaml_out = generate_yaml(assignments, fixed_layers, layer_range)
        print(f"\n{'=' * 70}")
        print("YAML OVERRIDES:")
        print(f"{'=' * 70}\n")
        print(yaml_out)

        yaml_path = base_dir / f"greedy_optimization_nex_{target}bpw_quadratic.yaml"
        yaml_path.write_text(yaml_out)
        print(f"\nSaved: {yaml_path}")

        json_path = base_dir / f"greedy_optimization_nex_{target}bpw_quadratic.json"
        json_data = {
            "assignments": {str(k): v for k, v in assignments.items()},
            "fixed_layers": fixed_layers,
            "distribution": {str(k): v for k, v in dist.items()},
            "avg_bitrate": avg,
            "target_bitrate": target,
            "total_error": cost,
            "split": {"n5": n5, "n4": n4, "n3": n3, "n2": n2},
        }
        json_path.write_text(json.dumps(json_data, indent=2))
        print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
