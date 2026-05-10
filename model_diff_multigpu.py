import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force NCCL onto TCP socket transport — required while the geohot P2P patch
# isn't live (nvidia-smi topo -p2p r shows CNS). Remove these once the patched
# driver is reloaded and P2P is back. Must be set before NCCL initializes;
# spawned workers re-import this module so they pick the env up too.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")

import argparse
from exllamav3.util.file import disk_lru_cache
from exllamav3.util.progress import ProgressBar
from exllamav3.util.memory import free_mem
from exllamav3.util.measures import cosine_error, sqnr
from exllamav3 import Config, Model, Tokenizer
from exllamav3.loader import SafetensorsCollection, VariantSafetensorsCollection
from datasets import load_dataset
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import math
import threading
import yaml


# Tunables — edit here, not via CLI
LOGITS_CHUNK = 256
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"
# States stay on GPU between layers (lower latency, no CPU↔GPU traffic).
# Set to True to offload between-layer hidden states to host RAM if you
# need GPU headroom — costs one D2H + one H2D copy per batch per layer.
STATE_OFFLOAD_CPU = True
# Prefetch next layer's weights on a background thread while the current
# layer is computing. Disk/compute overlap.
PREFETCH_OVERLAP = True


@disk_lru_cache("get_dataset_text")
def get_dataset_text(spec: dict):
    assert spec["dataset"] == "wiki2", "Only wiki2 implemented atm"
    return "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])


def get_test_tokens(tokenizer, rows, eval_len=2048, eval_stride=512):
    with ProgressBar("Tokenizing", rows) as pb:
        eval_tokens = tokenizer.encode(get_dataset_text({"dataset": "wiki2"}))
        num_tokens = eval_tokens.shape[-1]
        seqs = []
        for a in range(0, num_tokens - eval_len, eval_stride):
            seqs.append(eval_tokens[:, a:a + eval_len])
            pb.update(len(seqs))
            if len(seqs) >= rows:
                break
    return torch.cat(seqs, dim=0)


def ppl_partial(input_ids, logits):
    chunksize = 10240
    logprob_sum, logprob_count = 0.0, 0
    b_ = 0
    while b_ < logits.shape[0]:
        a_ = b_
        b_ = min(b_ + chunksize, logits.shape[0])
        logits_f = logits[a_:b_, :].float() + 1e-10
        target_ids = input_ids[a_ + 1:b_ + 1].to(logits.device)
        log_probs = F.log_softmax(logits_f, dim=-1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        logprob_sum += token_log_probs.sum().item()
        logprob_count += target_ids.numel()
    return logprob_sum, logprob_count


def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def broadcast_tokens(rank, all_eval_ids_or_none, device):
    if rank == 0:
        shape_t = torch.tensor(list(all_eval_ids_or_none.shape), dtype=torch.long, device=device)
    else:
        shape_t = torch.zeros(2, dtype=torch.long, device=device)
    dist.broadcast(shape_t, src=0)
    if rank == 0:
        gpu_tokens = all_eval_ids_or_none.to(device)
    else:
        gpu_tokens = torch.empty(shape_t[0].item(), shape_t[1].item(),
                                 dtype=torch.long, device=device)
    dist.broadcast(gpu_tokens, src=0)
    return gpu_tokens.cpu()


class _Prefetcher:
    # Loads module N+1 on a background thread while module N is computing.
    # Assumes module.unload() does not touch config.stc; the main thread waits
    # for any pending prefetch before its next stc-touching call.
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.thread = None
        self.error = None

    def start(self, idx):
        if idx >= len(self.model.modules):
            self.thread = None
            return
        module = self.model.modules[idx]
        prefer_cpu = bool(module.caps.get("prefer_cpu"))
        target = self.device if not prefer_cpu else "cpu"
        self.error = None

        def _load():
            try:
                self.config.stc.begin_deferred_load()
                module.load(target)
                self.config.stc.end_deferred_load()
            except BaseException as e:
                self.error = e

        self.thread = threading.Thread(target=_load, daemon=True)
        self.thread.start()

    def wait(self):
        if self.thread is None:
            return
        self.thread.join()
        self.thread = None
        if self.error is not None:
            err = self.error
            self.error = None
            raise err


@torch.inference_mode()
def worker(rank, world_size, args):
    setup_distributed(rank, world_size)

    pair_id = rank // 2
    role = rank % 2  # 0 = model A holder, 1 = model B holder
    partner_rank = rank + 1 if role == 0 else rank - 1
    n_pairs = world_size // 2
    device = torch.device("cuda", rank)

    # Each rank loads ONLY its half (model A or model B). Per-process host
    # RAM is half what it would be if every rank loaded both models.
    model_dir = args.model_a if role == 0 else args.model_b
    config = Config.from_directory(model_dir)
    config.override_dynamic_seq_len(2048)
    tokenizer = Tokenizer.from_config(config) if rank == 0 else None
    model = Model.from_config(config)

    # Tensor overrides only meaningful for model A
    if args.override and role == 0:
        with open(args.override, "r") as f:
            comp = yaml.safe_load(f)
        sources = {s["id"]: s["model_dir"] for s in comp["sources"]}
        overrides = {o["key"]: sources[o["source"]] for o in comp["overrides"]}
        collections = {}
        for o_key, o_dir in overrides.items():
            collections.setdefault(o_dir, []).append(o_key)
        if collections:
            vstc = VariantSafetensorsCollection(config.stc)
            for o_dir, o_keys in collections.items():
                if rank == 0:
                    print(f" -- Overriding from: {o_dir}:")
                    for o_key in o_keys:
                        print(f"      {o_key}")
                vstc.add_stc(o_keys, SafetensorsCollection(o_dir))
            config.stc = vstc

    # Tokenize once on rank 0, broadcast to all
    all_eval_ids = get_test_tokens(tokenizer, args.rows) if rank == 0 else None
    all_eval_ids = broadcast_tokens(rank, all_eval_ids, device)

    # Each pair handles its own slice of batches
    all_batches = list(all_eval_ids.split(args.batch_size))
    local_batches = all_batches[pair_id::n_pairs]
    n_local = len(local_batches)
    n_total_rows = sum(b.shape[0] for b in all_batches)

    states = [b.clone() for b in local_batches]
    topk_max = args.topk_max

    prefetch = _Prefetcher(model, config, device)

    for idx, module in enumerate(model.modules):
        is_logits_layer = (module is model.modules[-1])
        is_prefer_cpu = bool(module.caps.get("prefer_cpu"))

        # Ensure current module is loaded — sync for the first one, otherwise
        # whatever the previous iteration kicked off is still in flight.
        if PREFETCH_OVERLAP:
            if idx == 0:
                config.stc.begin_deferred_load()
                module.load(device if not is_prefer_cpu else "cpu")
                config.stc.end_deferred_load()
            else:
                prefetch.wait()
            # Kick off next module's load in the background.
            prefetch.start(idx + 1)
        else:
            # Synchronous load every iteration.
            config.stc.begin_deferred_load()
            module.load(device if not is_prefer_cpu else "cpu")
            config.stc.end_deferred_load()

        # Decoder accumulators (role 0 only meaningfully populates)
        max_diff = 0.0
        rfn_error_sum = 0.0
        cos_error_sum = 0.0
        sqnr_sum = 0.0

        # Logits accumulators (role 0 has KL + agree + own ppl/topk; role 1 has own ppl/topk)
        logprob_sum_local = 0.0
        logprob_count_local = 0
        kl_token_sum_ab = 0.0
        kl_token_sum_ba = 0.0
        kl_token_count = 0
        topk_hits_sum_local = [0] * topk_max
        topk_hits_count_local = [0] * topk_max
        topk_agreement_sum = [0] * topk_max
        topk_agreement_count = [0] * topk_max

        for b in range(n_local):
            eval_ids = local_batches[b]
            state = states[b]

            params = {}
            state = module.prepare_for_device(state, params)
            state = module.forward(state, params)

            # Force state onto GPU after forward. Otherwise prefer_cpu modules
            # (e.g. embed_tokens) leave us with one CPU output per batch
            # accumulating in `states[b]` — that's the source of the 60 GB
            # host-RAM spike during iter 0 if not handled.
            if not state.is_cuda:
                state = state.to(device)

            if not is_logits_layer:
                # Decoder layer: ship state across pair, role 0 computes metrics
                state_gpu = state.contiguous()

                if role == 0:
                    state_b_recv = torch.empty_like(state_gpu)
                    dist.recv(state_b_recv, src=partner_rank)

                    # Match upstream: keep_b override happens before metrics so
                    # the metric prints zero error for forced-equal layers.
                    if idx < args.keep_b:
                        state_gpu = state_b_recv
                        state = state_b_recv

                    rows = state_gpu.shape[0]
                    for j in range(rows):
                        # CRITICAL: use Python `float` (= torch.float64), NOT
                        # torch.float (= torch.float32). On a fp32 source,
                        # .to(torch.float) returns a view; the in-place
                        # sa -= sb / sa.abs_() below would then corrupt
                        # state_gpu (which IS states[b]). .to(float) goes to
                        # fp64 and forces an out-of-place copy.
                        sa = state_gpu[j].to(float)
                        sb = state_b_recv[j].to(float)
                        cos_error_sum += cosine_error(sa, sb)
                        sqnr_sum += sqnr(sa, sb)
                        sa -= sb
                        denom = torch.linalg.norm(sb, 'fro').mean()
                        rfn_error_sum += (torch.linalg.norm(sa, 'fro') / denom).item()
                        sa.abs_()
                        md = (sa.max().item() / denom).item()
                        max_diff = max(max_diff, md)
                        del sa, sb
                else:
                    dist.send(state_gpu, dst=partner_rank)

                if STATE_OFFLOAD_CPU:
                    states[b] = state.cpu()
                else:
                    states[b] = state

            else:
                # Logits layer: per-row work, chunked KL stream
                rows = state.shape[0]
                seq_len = state.shape[1]

                for j in range(rows):
                    logits_row = state[j]  # [seq, vocab] on GPU
                    input_ids = eval_ids[j]

                    # Local PPL on logits[:-1]
                    lps, lpc = ppl_partial(input_ids, logits_row[:-1, :])
                    logprob_sum_local += lps
                    logprob_count_local += lpc

                    # Local top-K hits (label appears in own top-K)
                    _, top_index = torch.topk(logits_row, topk_max, dim=-1)
                    top_index_aligned = top_index[:-1, :].cpu()
                    targets = input_ids[1:].view(-1, 1)
                    for t in range(topk_max):
                        ts = top_index_aligned[:, :t + 1]
                        hits = torch.eq(targets, ts).any(dim=1)
                        topk_hits_sum_local[t] += hits.sum().item()
                        topk_hits_count_local[t] += ts.shape[0]

                    # Top-K agreement: role 1 -> role 0
                    top_index_xfer = top_index.contiguous()
                    if role == 1:
                        dist.send(top_index_xfer, dst=partner_rank)
                    else:
                        partner_top = torch.empty_like(top_index_xfer)
                        dist.recv(partner_top, src=partner_rank)
                        a_aligned = top_index_xfer[:-1, :].cpu()
                        b_aligned = partner_top[:-1, :].cpu()
                        for t in range(topk_max):
                            sa_ = a_aligned[:, :t + 1]
                            sb_ = b_aligned[:, :t + 1]
                            row_hits = torch.eq(sa_, sb_).all(dim=1)
                            topk_agreement_sum[t] += row_hits.sum().item()
                            topk_agreement_count[t] += sa_.shape[0]
                        del partner_top
                    del top_index, top_index_aligned, top_index_xfer

                    # KL via chunked seq stream — bf16 logits over the wire,
                    # both sides log_softmax in fp32 locally to keep precision.
                    for cs in range(0, seq_len, LOGITS_CHUNK):
                        ce = min(cs + LOGITS_CHUNK, seq_len)
                        local_chunk = logits_row[cs:ce, :].contiguous()
                        if local_chunk.dtype != torch.bfloat16:
                            local_chunk = local_chunk.to(torch.bfloat16)

                        if role == 1:
                            dist.send(local_chunk, dst=partner_rank)
                        else:
                            recv_chunk = torch.empty_like(local_chunk)
                            dist.recv(recv_chunk, src=partner_rank)
                            log_p_a = F.log_softmax(local_chunk.float(), dim=-1)
                            log_p_b = F.log_softmax(recv_chunk.float(), dim=-1)
                            p_a = log_p_a.exp()
                            p_b = log_p_b.exp()
                            # Match upstream variable convention:
                            #   kl_div_ab uses log(p_a) and p_b
                            #     = sum p_b (log p_b - log p_a) = KL(B || A)
                            kl_ab = (p_b * (log_p_b - log_p_a)).sum(dim=-1)
                            kl_ba = (p_a * (log_p_a - log_p_b)).sum(dim=-1)
                            kl_token_sum_ab += kl_ab.sum().item()
                            kl_token_sum_ba += kl_ba.sum().item()
                            kl_token_count += kl_ab.numel()
                            del log_p_a, log_p_b, p_a, p_b, kl_ab, kl_ba, recv_chunk
                        del local_chunk

                states[b] = None

        # Reduce decoder metrics across world (role 1 contributes zeros)
        if not is_logits_layer:
            sums = torch.tensor(
                [rfn_error_sum, cos_error_sum, sqnr_sum] if role == 0 else [0.0, 0.0, 0.0],
                dtype=torch.float64, device=device,
            )
            mx = torch.tensor(max_diff if role == 0 else float("-inf"),
                              dtype=torch.float64, device=device)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(mx, op=dist.ReduceOp.MAX)

            if rank == 0:
                rfn_e = (sums[0] / n_total_rows).item()
                cos_e = (sums[1] / n_total_rows).item()
                sqnr_v = (sums[2] / n_total_rows).item()
                print(
                    f" -- {module.key:40}"
                    f"   rfn_err: {rfn_e:.6f}"
                    f"   max_diff/norm: {mx.item():.6f}"
                    f"   sqnr: {sqnr_v:9.6f}"
                    f"   cos_err: {cos_e:.6f}"
                )

        module.unload()
        if not PREFETCH_OVERLAP:
            config.stc.close()
        free_mem()

    # Make sure the trailing prefetch (if any) is fully drained before close
    prefetch.wait()
    config.stc.close()

    # Logits-layer aggregation: pack everything and reduce once
    K = topk_max
    pack = torch.zeros(7 + 6 * K, dtype=torch.float64, device=device)
    if role == 0:
        pack[0] = logprob_sum_local
        pack[1] = logprob_count_local
        pack[4] = kl_token_sum_ab
        pack[5] = kl_token_sum_ba
        pack[6] = kl_token_count
        for t in range(K):
            pack[7 + t] = topk_hits_sum_local[t]
            pack[7 + K + t] = topk_hits_count_local[t]
            pack[7 + 4 * K + t] = topk_agreement_sum[t]
            pack[7 + 5 * K + t] = topk_agreement_count[t]
    else:
        pack[2] = logprob_sum_local
        pack[3] = logprob_count_local
        for t in range(K):
            pack[7 + 2 * K + t] = topk_hits_sum_local[t]
            pack[7 + 3 * K + t] = topk_hits_count_local[t]

    dist.all_reduce(pack, op=dist.ReduceOp.SUM)

    if rank == 0:
        ppl_A = math.exp(-pack[0].item() / pack[1].item())
        ppl_B = math.exp(-pack[2].item() / pack[3].item())
        kl_ab = pack[4].item() / pack[6].item()
        kl_ba = pack[5].item() / pack[6].item()

        print(f" -- A perplexity: {ppl_A:11.8f}")
        print(f" -- B perplexity: {ppl_B:11.8f}")
        print(f" -- A label in top-K:")
        for t in range(K):
            s, c = pack[7 + t].item(), pack[7 + K + t].item()
            print(f"      K = {t+1}: {s/c:6.4f}")
        print(f" -- B label in top-K:")
        for t in range(K):
            s, c = pack[7 + 2 * K + t].item(), pack[7 + 3 * K + t].item()
            print(f"      K = {t+1}: {s/c:6.4f}")
        print(f" -- Top-K agreement, A vs B:")
        for t in range(K):
            s, c = pack[7 + 4 * K + t].item(), pack[7 + 5 * K + t].item()
            print(f"      K = {t+1}: {s/c:6.4f}")
        print(f" -- KL divergence (A, B): {kl_ab:11.8f}")
        print(f" -- KL divergence (B, A): {kl_ba:11.8f}")

    dist.barrier()
    dist.destroy_process_group()


def main(args):
    avail = torch.cuda.device_count()
    world_size = args.world_size if args.world_size > 0 else avail
    assert world_size >= 2 and world_size % 2 == 0, \
        f"Need an even number of GPUs >= 2 for pair-DP, got {world_size}"
    assert world_size <= avail, f"--world_size {world_size} > {avail} visible GPUs"
    print(f" -- Launching pair-DP={world_size // 2} on {world_size}/{avail} GPUs")
    mp.spawn(worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ma",  "--model_a", type=str, required=True, help="Model A directory")
    parser.add_argument("-mb",  "--model_b", type=str, required=True, help="Model B directory")
    parser.add_argument("-r",   "--rows", type=int, default=100)
    parser.add_argument("-kb",  "--keep_b", type=int, default=0,
                        help="Force model A to use model B state for first N modules")
    parser.add_argument("-tkm", "--topk_max", type=int, default=5)
    parser.add_argument("-or",  "--override", type=str, default=None,
                        help="Model A tensor override spec (YAML)")
    parser.add_argument("-bsz", "--batch_size", type=int, default=1)
    parser.add_argument("-ws",  "--world_size", type=int, default=0,
                        help="Number of GPUs/ranks (0 = use all visible). Must be even for pair-DP.")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    main(args)
