# Future latent rollout training

Two independent axes, with confusingly similar names. Get them straight first:

```text
FUTURE_PREDICTOR_MODE   WHERE the future positions live
  resampler   learned queries cross-attend to Qwen's hidden states, outside the backbone
  direct      the positions are packed into Qwen's own sequence and pass all 32 layers

PREDICTION_MODE         HOW the horizons are produced
  one_shot        every horizon emitted in a single pass (the original behaviour)
  autoregressive  one shared transition applied per chunk, fed its own output
```

`one_shot` is the default on both axes and is unchanged, byte for byte -- `test_rollout.py`
asserts it against a hand-built reference.

## What the rollout actually computes

Qwen runs **once**. The rollout is a small transition applied per chunk:

```text
Z_hat_k = Z_{k-1} + g(cond_k, Z_{k-1})
```

`direct` (recommended): `cond_k` is Qwen's own hidden state at chunk k's future positions,
`future_hidden[:, k]`. One forward already produced all ten chunks, so the rollout costs
one backbone pass, not ten. No resampler and no cross-attention -- chunk k's conditioning
already went through 32 layers and already carries its own time identity from
`future_embeddings.time_embed`, so the step only folds in the previous latent. 8.62 M
trainable outside the LoRA.

`resampler` (ablation): `cond_k` is the projected Qwen context, identical at every step,
and the transition is 4 blocks of cross-attention. 73.55 M. Kept so that "does the
conditioning source matter" is answerable, not because it is the better arm -- the direct
one-shot baseline scores better than the resampler one-shot baseline (val 0.6729 vs
0.6819), and the rollout inherits that.

Re-running Qwen per chunk is deliberately not built: a single forward is 43.4 s/it
measured, so ten sequential grad-carrying ones is ~44 days for an 8800-step run.

**What this costs.** The feedback enters *after* the backbone, so Qwen never sees a
predicted latent: `cond_k` is computed from the learned future positions only. The
recurrence is therefore shallow -- a transition on top of a fixed per-chunk conditioning,
not reasoning about the predicted future inside the 32 layers. That is the price of one
forward instead of ten, and it is the thing an in-Qwen rollout would buy. Worth naming
because "autoregressive" in the experiment plan implies the deeper version.

### Residual parameterization

`RESIDUAL_FEEDBACK=1` (default) predicts a delta on the previous chunk and initialises the
head's output weight small (`LatentHead.RESIDUAL_INIT_STD`, 1e-3 -> delta std ~0.03). The
transition is then very nearly the identity before training, so **the rollout starts out
being the persistence baseline** rather than the dataset mean. On trunk0 that is the
difference between a starting loss of ~0.51 and ~1.17, and it is the whole reason feeding
the observed chunk in is worth anything: without it there is no path from input to output
at initialisation, and the model has to learn to copy from scratch.

Small rather than exactly zero on purpose. Measured on real latents, a delta of std 0.03
moves the starting loss by 0.0006 (0.4403 -> 0.4409), so zero buys nothing -- and zero
would make the gradient w.r.t. everything upstream of the head exactly zero on the first
step, since it arrives multiplied by those weights.

## Data

Rebuild the splits so rows carry the observed chunk:

```bash
python <this-dir>/dataset.py --cache-root "$VPDATA_LATENT_ROOT" --build
```

The manifest's `first_half` becomes `context_latent_path`; `last_half` stays the target.
Existing JSONL can instead have an absolute `context_latent_path` added. It must hold
observed latents from the same encoder, statistics and spatial grid as the target. The
template loads only the final observed chunk. Missing context is a hard error in AR mode.
Cache tensors must be `[T, H*W, D]`, optionally under a `latent` key.

## Training

```bash
FUTURE_PREDICTOR_MODE=direct PREDICTION_MODE=autoregressive \
  bash <this-dir>/sft.sh --output_dir output/vrae_direct_rollout
```

Run the `sft.sh` that sits next to the `model.py` you mean to train: the plugins are
loaded from the script's own directory. (A copy of this tree once kept loading the
original `examples/` plugins, which have no `prediction_mode` at all -- so
`PREDICTION_MODE=autoregressive` was accepted, ignored, and the run trained `one_shot`
without a word.)

| knob | meaning |
|---|---|
| `ROLLOUT_WEIGHT` | 0 = teacher forcing only, 1 = predicted feedback only, in between = `(1-w)*L_teacher + w*L_rollout` with both branches run (two predictor passes, one Qwen pass). Loss mixing, **not** scheduled sampling. |
| `BPTT_DEPTH` | how far gradients travel back along the feedback chain. `d >= chunks` is full BPTT; `d=1` treats each feedback as a constant. **This is the knob for the plan's A2/A3/A4** (`d=2/5/10`): all horizons stay scored, so the arms remain comparable to each other and to one-shot. |
| `ROLLOUT_STEPS` | how many horizons exist at all. Shortening it makes an *easier problem* and the loss is then **not comparable** to a full-horizon run; the model logs a warning. Debug only -- leave at `LATENT_CHUNKS`. |
| `RESIDUAL_FEEDBACK` | see above. Set 0 for the non-residual arm. |
| `AR_TIME_EMBED` | adds `e_t(t_k)` to the resampler's queries. Without it the resampler transition is time-invariant and cannot tell step 1 from step 9 on a near-static clip. No effect in direct mode, which gets time from `cond_k`. |

Evaluation always rolls out with predicted feedback and never sees future targets,
whatever `ROLLOUT_WEIGHT` was trained with, so the reported `eval_loss` is a full rollout.

Logged: `loss_cos`, `loss_mse`, `loss_temp`, `loss_rollout` / `loss_teacher` for the active
branches, `horizon_N_loss` per horizon, and `rollout_weight`.

`--modules_to_save` is chosen by `sft.sh` from `FUTURE_PREDICTOR_MODE`
(`future_embeddings latent_head` for direct, `resampler latent_head` otherwise). The
feedback projection is attached to whichever of those the mode keeps, so it is saved with
it -- parked anywhere else it would train and silently never be written.

One-shot uses T query time embeddings and AR uses one, so switching an existing one-shot
adapter to AR is not a strict-load-compatible resume: start a new adapter. For reload,
resume and evaluation, export the same geometry, `FUTURE_PREDICTOR_MODE`,
`PREDICTION_MODE`, `ROLLOUT_STEPS`, `BPTT_DEPTH`, `RESIDUAL_FEEDBACK`, `AR_TIME_EMBED` and
`LATENT_STATS_PATH` as training. Environment settings override the saved config.

## Validation

```bash
python <this-dir>/test_rollout.py        # 13 CPU tests, no GPU/Qwen/dataset needed
```

Real predictor/resampler/autograd against a tiny stub backbone, Swift registration
stubbed. Covers: final-horizon gradients reaching the first prediction; GT feedback under
teacher forcing; no future leakage at eval; mixed loss; **one-shot equivalence**; residual
init reproducing persistence exactly and its gradients staying alive; `BPTT_DEPTH` cutting
the chain; the direct rollout using one backbone pass, carrying no resampler, and
conditioning each step on its own chunk; `AR_TIME_EMBED` being resampler-only; shape and
config errors; state round-trip; template encode/collate/hook transport.

It does not validate real Qwen, PEFT/DDP, video processing or decoder quality. Before a
long run, in order:

1. `verify.py --check-checkpoint` with the run's own mode env exported.
2. **The init-loss gate.** One step of real training: the first `loss` must be ~0.51, not
   ~1.17. If it is ~1.17 the residual path is not wired up -- stop there, nothing after it
   is worth running.
3. The log must show `prediction_mode autoregressive` and a `feedback_proj` in the param
   count. If it says `one_shot`, the plugins came from the wrong directory.
4. Overfit gate (`overfit.sh`): loss < 0.1.
5. `eval_latent.py --split train --limit 300` and `--split val`, then `decode_eval.py`,
   against the persistence baseline (0.5144 train / 0.5410 val on trunk0) and the existing
   decodes in `outputs/` on the same samples.
