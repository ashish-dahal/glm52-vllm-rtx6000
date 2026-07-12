# GLM-5.2-NVFP4 on 8x RTX PRO 6000 Blackwell (patched vLLM v0.25.0)

Serving setup for `nvidia/GLM-5.2-NVFP4` (744B MoE, 40B active, DeepSeek
Sparse Attention, ~465 GB weights) on a single PCIe-only node (no NVLink)
with 8x RTX PRO 6000 Blackwell (SM120, 96 GB).

Stock `vllm/vllm-openai:v0.25.0` cannot serve this model correctly on this
hardware; this repo packages the minimal set of post-tag upstream fixes (plus
one community DCP patch) into a reproducible Docker image and a tuned
`docker-compose.yml`.

## Quick start

```bash
# Model weights (~465 GB) downloaded somewhere local, e.g. /models/glm52
export MODEL_DIR=/models/glm52

docker compose up -d --build
# First start takes several minutes (weight load + CUDA graph capture).
curl http://localhost:8000/v1/models
```

The OpenAI-compatible API is served on port 8000 with model name
`GLM-5.2-NVFP4`.

## Files

- [`Dockerfile`](Dockerfile): official `vllm/vllm-openai:v0.25.0` base,
  applies `fix.patch` to the installed package, and pins FlashInfer 0.6.14
  with the prebuilt cu130 jit-cache.
- [`docker-compose.yml`](docker-compose.yml): GPU wiring, PCIe-safe NCCL
  environment, and the recommended serve command.
- [`fix.patch`](fix.patch): 7 git-format commits, all Python-only, so the
  image's precompiled CUDA extensions keep working.

## What fix.patch contains, and attribution

Six exact upstream vLLM backports that landed after the v0.25.0 tag, needed
for correct and fast MTP speculative decoding with this model family. All
credit for these goes to their upstream authors; the commits are applied
verbatim (author, sign-off, and message preserved):

| Upstream PR | Commit | Author | Why it's needed |
|---|---|---|---|
| [#47381](https://github.com/vllm-project/vllm/pull/47381): [Bugfix][Model Runner V2] Order uniform decodes first | `85b3a72` | Woosuk Kwon, Nick Hill | Spec-decode batches could be misclassified as prefills; correctness and perf for MTP |
| [#48085](https://github.com/vllm-project/vllm/pull/48085): [Bugfix] Fix race condition in KVBlockZeroer | `1cd75b3` | Benjamin Chislett (NVIDIA), Wentao Ye | KV-cache zeroing race |
| [#48046](https://github.com/vllm-project/vllm/pull/48046): [Bugfix] Use int8 workspace for FlashInfer MLA decode | `95d6d6f` | Nick Hill | Blackwell-compatible FlashInfer MLA workspace |
| [#42642](https://github.com/vllm-project/vllm/pull/42642): Fix FlashAttention MLA prefill V unpadding | `c2ecd0f` | Martin Vit, Matthew Bonanni (Red Hat) | Long-context MLA prefill correctness |
| [#47911](https://github.com/vllm-project/vllm/pull/47911): fix: hash speculative draft model config | `7cc2e8e` | Ace Eldeib (CoreWeave) | Draft-config changes otherwise miss the compile cache |
| [#47914](https://github.com/vllm-project/vllm/pull/47914): [Spec Decode] Support hybrid (SWA + full attention) DFlash drafters | `0d12618` | Michael Goin | Spec-decode plumbing the above fixes build on |

Plus one feature patch:

| Source | Commit | Author | Why it's needed |
|---|---|---|---|
| [vllm-project/vllm PR #47779](https://github.com/vllm-project/vllm/pull/47779): [Bugfix][SM120][MLA] Enable DCP for FlashInfer sparse MLA decode (open, not yet merged) | `0f5a46a` | [Sebastiaan van Duijn](https://github.com/sebastiaanvduijn) (Codex-assisted) | Unlocks `--decode-context-parallel-size` on this GPU; needed only for the long-context DCP variant below |

The DCP commit is taken from the PR's source repo,
[`sebastiaanvduijn/vllm`](https://github.com/sebastiaanvduijn/vllm), branch
[`codex/sm120-sparse-mla-dcp`](https://github.com/sebastiaanvduijn/vllm/tree/codex/sm120-sparse-mla-dcp).
Many thanks to Sebastiaan van Duijn for this work.

### Provenance notes

- The six backports are verbatim upstream vllm-project commits (merged; PR
  links above). If you only need the standard 131k-context lane, the DCP
  commit can be dropped and the patch set becomes pure upstream backports.
- Upstream's DCP support (the [#46076](https://github.com/vllm-project/vllm/pull/46076)
  lineage, tracked in [issue #37113](https://github.com/vllm-project/vllm/issues/37113))
  covers other MLA backends but not `FLASHINFER_MLA_SPARSE_SM120`, the
  backend this GPU requires; PR #47779 extends the same mechanism to it.
  Related upstream PRs ([#46514](https://github.com/vllm-project/vllm/pull/46514),
  [#39635](https://github.com/vllm-project/vllm/pull/39635)) target
  `FLASHMLA_SPARSE`, whose kernels do not exist for SM120.

## Why these vLLM configs

- **`-tp 8 --enable-expert-parallel`**: the model needs all 8 GPUs
  (~57 GiB weights per GPU). EP keeps each expert whole on one GPU and routes
  tokens to it (all-to-all), instead of TP-slicing every expert, which would
  add per-layer all-reduce traffic over PCIe, the scarce resource on a
  no-NVLink node.
- **MTP3** (`--speculative-config {"method":"mtp","num_speculative_tokens":3}`):
  GLM-5.2 ships a built-in multi-token-prediction drafting head; with 3
  draft tokens per step it roughly doubles single-user decode throughput on
  PCIe-bound TP8, because each expensive all-reduce round is amortized over
  multiple tokens. Speculative decoding is lossless: every draft token is
  verified by the target model. Prefer 3 draft tokens; 1 gives less depth
  for the same per-step overhead.
- **`--kv-cache-dtype fp8_e4m3`**: halves KV-cache memory, directly doubling
  context capacity.
- **`--gpu-memory-utilization 0.9448`**: sized for a dedicated node; lower it
  if the GPUs are shared with other workloads.
- **`--disable-custom-all-reduce`** plus the NCCL env in compose: mandatory on
  this hardware. These GPUs have no NVLink and no P2P atomics: NCCL P2P
  deadlocks at init (hence `NCCL_P2P_DISABLE=1`), and vLLM's custom
  allreduce paths are dramatically slower than plain NCCL ring over PCIe.
- **`--attention-backend FLASHINFER_MLA_SPARSE_SM120`**: the sparse-MLA
  backend for DeepSeek Sparse Attention on SM120. It requires FlashInfer
  0.6.14 with the **prebuilt cu130 jit-cache** wheel: the AOT kernel cache
  avoids FlashInfer's JIT path, which does not compile on this CUDA 13
  toolchain. `FLASHINFER_DISABLE_VERSION_CHECK=1` is set because the
  companion cubin package version lags behind.
- **`--max-model-len 131072 --max-num-seqs 16 --max-num-batched-tokens 8192`**:
  a balanced interactive/coding/RAG profile. With fp8 KV at util 0.9448 the
  KV pool is roughly 560k tokens, i.e. about six concurrent requests at ~90k
  context each when all are at full depth simultaneously; prefix caching
  stretches this in practice.
- **`--reasoning-parser glm45 --tool-call-parser glm47`**: GLM-5.2's chat
  format. Reasoning is returned in the `reasoning` field, tool calls parse
  with the glm47 grammar.

## Long-context variant (up to 256k per request)

DCP (decode context parallelism) shards the KV cache across ranks: more
context capacity, paid for with extra inter-GPU communication (roughly
20-45% slower decode and about half the prefill rate on PCIe). Enable it only
when the workload actually needs it, e.g. several concurrent users at the
model's full 262,144-token context:

```
--decode-context-parallel-size=4 --dcp-comm-backend=ag_rs --max-model-len=262144
```

DCP=4 roughly quadruples the KV pool (~2.1M tokens, about six users at 256k
each). Avoid DCP=8: its extra capacity exceeds what the model's context limit
can use, and the communication cost keeps growing. Do not raise
`max-model-len` beyond 262,144; that is the model's trained
positional-embedding ceiling.

## Acknowledgements

- The [vLLM project](https://github.com/vllm-project/vllm) and the authors of
  the backported PRs listed above.
- [Sebastiaan van Duijn](https://github.com/sebastiaanvduijn) for the SM120
  sparse-MLA DCP patch ([PR #47779](https://github.com/vllm-project/vllm/pull/47779)).
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for the SM120
  sparse-MLA kernels and prebuilt cu130 jit-cache.

This repo redistributes vLLM patches under vLLM's
[Apache-2.0 license](https://github.com/vllm-project/vllm/blob/main/LICENSE);
original authorship is preserved in the git-format headers inside
`fix.patch`.
