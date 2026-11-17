# GPU expert tier for MoE streaming

With `--moe-stream`, decode batches compute the streamed experts on the CPU from the RAM cache, which leaves most
of the VRAM that prompt processing needed idle. Two changes put it to use:

- **Workspace switching.** Batches of up to `LLAMA_DECODE_WORKSPACE` tokens (32) run with a scheduler reserved
  for that size; a larger batch restores the full `-ub` workspace. The old scheduler is freed before the new one
  is reserved.
- **GPU expert tier.** In the small workspace, each GPU gets banks of the most used streamed experts, sized from
  its free VRAM minus a headroom. Their routes are computed on the GPU that holds them and masked out of the CPU
  path; the other routes keep the RAM/SSD stream. The contents follow the usage scores of the stream: every 32
  processed tokens, experts used clearly more than the least used held ones replace them, within a transfer budget.

Opt-in: `LLAMA_MOE_HOT=1` with `--moe-stream`, `-np 1` and a causal decoder model; the sizing is automatic. Without
it, nothing changes.

## Settings

| Variable | Default | |
| --- | --- | --- |
| `LLAMA_MOE_HOT` | 0 | 1 = GPU expert tier (and workspace switching, if not set) |
| `LLAMA_DECODE_WORKSPACE` | 32 with the tier, else 0 | small workspace size in tokens, 0 = no switching (and no tier); usable without the tier |
| `LLAMA_MOE_HOT_HEADROOM_MIB` | 256 | VRAM left free on each GPU after the banks |
| `LLAMA_MOE_HOT_UPDATE` | 32 | processed tokens between replacement checks, 0 = fixed contents for the phase |
| `LLAMA_MOE_HOT_UPDATE_MIB` | 256 | GPU upload + download per check |
| `LLAMA_MOE_STREAM_PROMPT_WEIGHT` | 0 | weight of a prompt token in the usage scores, relative to a decode token (e.g. 0.05) |

The free VRAM is measured after the small workspace is reserved, so the compute buffers are already accounted for;
the headroom covers what is allocated later (the reservation with the GPU branches, the CUDA pools, CUDA graphs,
other contexts). A CUDA allocation failure during decode aborts. Check the lowest free VRAM during a long decode
(`nvidia-smi`) before lowering it; raise it if another process uses the GPUs.

A small workspace above the largest verification batch (draft max + 1) keeps short follow-up prompts in the small
phase: a larger batch releases and rebuilds the tier.

## How it works

**Banks.** An expert is the gate/up/down (or gate_up/down) planes of one layer. Experts are grouped by layout
(role, type, shape, stride): one bank per tensor of the layout and per GPU, `[ne0, ne1, slots + 2]` with a zero
expert and a padding plane after the slots, all banks of a GPU in one buffer. Layers with the same quantization
share slots, so a hot expert of any of them can take a slot from a cold one; layers with different quantizations
(mixed quants such as per-layer `down` types) get banks of their own. The number of slots of each layout is
fixed when the tier is built, from the usage of the experts at that time. No weight is requantized.

**Usage.** The tier ranks experts by the usage scores of the stream: their selections by decode batches, decayed
with `LLAMA_MOE_STREAM_HALFLIFE` (also the lfu order of the RAM cache, and saved with the cached experts in the
stream state file next to the model, with the experts of the tier, so a restart caches them). The experts of the
tier keep being scored. By default only decode batches count: after a long prompt on another topic, the RAM
cache and the tier keep following the previous one until decode catches up. `LLAMA_MOE_STREAM_PROMPT_WEIGHT`
makes prompt tokens count too (0.05: a 9k-token prompt weighs like ~450 decode tokens). A candidate replaces a held expert only if it is used 25% and 2 selections more, which keeps routing
noise from churning the tier.

**Placement.** At phase entry, the most used experts are placed first, from RAM or read from the model file if
they are not cached (only experts with usage, or in RAM, are candidates), each on the GPU with the most budget
left after it, whatever GPU its layer is on: an expert has one owner, two GPUs can hold different experts of a
layer. Live replacement takes the coldest compatible slot on either GPU.

**Graph.** Per layer and GPU with experts of it, a branch computes all routes of the layer as one-expert rows
(`[n_embd, 1, k*n]`): its own experts from their slots, the others from the zero expert. It is added to the CPU
part, in which the stream gives the tier's experts zero rows, before the router weighting. Scales, biases and the
activation are the same graph code as the CPU path, with the original expert ids. Unsupported ops make the tier
fall back to the streamed experts.

**RAM.** A GPU hit skips the RAM cache and its prefetches. The RAM copies of the tier's experts are the first
victims of the eviction policy, so that decode misses take their slots and VRAM adds to the RAM working set,
without extra reads. A replaced GPU expert is first admitted to RAM (downloaded, preferably into such a copy), and
is kept on the GPU if no RAM slot can take it. When the tier is released (a large batch, or a GPU suspend), its
experts are admitted to RAM the same way, keeping the copies of the others.
The GGUF stays the backing store of every expert: an expert is never only on a GPU.

**Timing.** Replacements happen between synchronized batches, never while a graph runs. A check costs a scan of
the expert usage and, if anything is replaced, the transfers; the graph is only reserved
again if a layer gains or loses its branch on a GPU.

**Phases.** A batch larger than `LLAMA_DECODE_WORKSPACE` (a new prompt) releases the tier and restores the full
workspace; the next small batch rebuilds the tier from RAM. Each new prompt therefore costs a download of the
tier to RAM and an upload of the new one.

**Suspend.** `--can-suspend` frees the tier before the CUDA buffers are suspended (`llama_moe_hot_release`), so
that it needs no host backup and no VRAM on restore; it is rebuilt from the free VRAM at the next decode.

## Logs

- `workspace_switch:` the compute buffer and free memory of each backend at a switch.
- `moe_hot: CUDA0: N experts in L layouts, X MiB (Y MiB zero/padding)` at phase entry.
- `moe_hot: T tokens: GPU a% of routes (...), RAM b%, GPU+RAM c%; ...` every `LLAMA_MOE_STREAM_INTERVAL` decode
  tokens: the share of all routes of the streamed layers computed by the GPUs and found in RAM, and the
  cumulative replacements. Tokens include rejected draft tokens.
- `moe_hot: released: ...` at phase exit.

## Limitations

- One sequence (`-np 1`), causal decoder models, no LoRA, no rotated (Hadamard) experts, not with `--moe-cache-mib`.
- The GPU branch computes one row per route, misses included: for one token, a GPU reads as many expert planes as
  the router selects. This runs in parallel with the CPU part and is normally shorter, but it is not a hit-only
  kernel.
- The headroom is a heuristic, not a VRAM ceiling: the CUDA pools and other contexts can still allocate.
- The per-layout slot counts are fixed for a phase; an expert can only replace one of its own layout.
- Only the main context has a tier; an MTP draft context switches its workspace on its own.
