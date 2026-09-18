"""CPU smoke tests: real predictor/resampler/autograd, stubbed Swift registration only.
Run: python examples/vrae_future_pred/test_rollout.py
No Qwen weights, dataset, GPU, or Swift optional dependencies required.
"""
import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import PretrainedConfig


def load_model():
    # Isolate plugin registration from the numerical implementation under test.
    names = ['swift', 'swift.model', 'swift.model.models', 'swift.model.models.qwen', 'swift.utils']
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules[name] = types.ModuleType(name)
    registry = sys.modules['swift.model']
    for name in ('Model', 'ModelGroup', 'ModelMeta', 'MultiModelKeys'):
        setattr(registry, name, lambda *a, **k: None)
    registry.register_model = registry.register_model_arch = lambda *a, **k: None
    sys.modules['swift.model.models.qwen'].Qwen3_5Loader = object
    sys.modules['swift.utils'].get_logger = lambda: logging.getLogger('test')
    sys.modules['swift.utils'].get_env_args = lambda name, kind, default: default
    try:
        spec = importlib.util.spec_from_file_location('rollout_model_test', Path(__file__).with_name('model.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


m = load_model()


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig(hidden_size=8)
        self.embedding = nn.Embedding(16, 8)

    def forward(self, input_ids, **kwargs):
        assert 'context_latent' not in kwargs
        return types.SimpleNamespace(hidden_states=(self.embedding(input_ids),))


class DirectBackbone(nn.Module):
    """Stands in for Qwen on the direct path: echoes inputs_embeds as the hidden state.

    `scale` is there so a test can check gradients reach the backbone, and `calls` so a
    test can prove the rollout runs the backbone once rather than once per chunk.
    """

    def __init__(self, dim=8):
        super().__init__()
        self.config = PretrainedConfig(hidden_size=dim)
        self.embedding = nn.Embedding(16, dim)
        self.scale = nn.Linear(dim, dim, bias=False)
        self.calls = 0
        self.last_seq_len = None

    def forward(self, inputs_embeds=None, input_ids=None, **kwargs):
        self.calls += 1
        assert 'context_latent' not in kwargs
        assert 'video_token_mask' not in kwargs, 'must not reach the backbone'
        hidden = self.scale(inputs_embeds) if inputs_embeds is not None else self.embedding(input_ids)
        self.last_seq_len = hidden.shape[1]
        return types.SimpleNamespace(hidden_states=(hidden,))


def predictor(mode='autoregressive', weight=1., steps=3, predictor_mode='resampler',
              backbone=None, **extra):
    return m.QwenVRAEFuturePredictor(backbone or Backbone(), m.QwenVRAEFuturePredictorConfig(
        latent_chunks=3, latent_height=1, latent_width=2, latent_dim=4,
        resampler_dim=8, resampler_depth=1, resampler_heads=2,
        predictor_mode=predictor_mode,
        prediction_mode=mode, rollout_weight=weight, rollout_steps=steps, **extra))


class RolloutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.inputs = dict(input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
                           attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
                           context_latent=torch.randn(2, 2, 2, 4), target_latent=torch.randn(2, 3, 2, 4))

    def test_final_horizon_backpropagates_to_first_prediction(self):
        model = predictor()
        chunks = []
        def hook(module, args, output):
            output.retain_grad()
            chunks.append(output)
        model.latent_head.register_forward_hook(hook)
        out = model(**self.inputs)
        model.latent_losses(out.z_pred[:, -1:], self.inputs['target_latent'][:, -1:])['loss'].backward()
        self.assertGreater(chunks[0].grad.abs().sum().item(), 0)
        for param in (model.resampler.feedback_proj.weight, model.qwen.embedding.weight):
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(), 0)

    def test_teacher_forcing_feedback_and_eval_no_target_leakage(self):
        model = predictor(weight=0)
        feedback = []
        model.resampler.feedback_proj.register_forward_pre_hook(lambda mod, args: feedback.append(args[0].detach()))
        model(**self.inputs)
        torch.testing.assert_close(feedback[0], self.inputs['context_latent'][:, -1])
        torch.testing.assert_close(feedback[1], self.inputs['target_latent'][:, 0])
        model.eval()
        original = model(**self.inputs).z_pred
        changed = model(**{**self.inputs, 'target_latent': torch.randn_like(self.inputs['target_latent'])}).z_pred
        torch.testing.assert_close(original, changed)
        torch.testing.assert_close(original, model(**{k:v for k,v in self.inputs.items() if k!='target_latent'}).z_pred)

    def test_mixed_objective(self):
        model = predictor(weight=.3)
        out = model(**self.inputs)
        torch.testing.assert_close(out.loss, .3*out.rollout_metrics['loss_rollout'] + .7*out.rollout_metrics['loss_teacher'])
        out.loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_one_shot_unchanged(self):
        model = predictor(mode='one_shot')
        out = model(**self.inputs)
        hidden = model.qwen.embedding(self.inputs['input_ids'])
        reference = model.latent_head(model.resampler(hidden, self.inputs['attention_mask']==0)).view(2,3,2,4)
        torch.testing.assert_close(out.z_pred, reference)
        torch.testing.assert_close(out.loss, model.latent_losses(reference, self.inputs['target_latent'])['loss'])
        self.assertFalse(hasattr(model.resampler, 'feedback_proj'))

    def test_direct_one_shot_unchanged(self):
        # Guards the merge: direct one-shot is the strongest baseline measured and a run
        # is training against it, so its arithmetic must stay exactly what it was before
        # the rollout was folded in -- gather the future positions, flatten (t,h,w) in
        # that order, project, reshape.
        backbone = DirectBackbone()
        model = predictor(mode='one_shot', predictor_mode='direct', backbone=backbone).eval()
        inputs = dict(self.inputs)
        inputs['inputs_embeds'] = torch.randn(2, 3, 8)
        inputs.pop('input_ids')
        with torch.no_grad():
            out = model(**inputs)
            packed, mask, future_mask = model._append_future_positions(
                inputs['inputs_embeds'], inputs['attention_mask'])
            hidden = backbone.scale(packed)
            reference = model.latent_head(
                hidden[future_mask].view(2, -1, 8)).view(2, 3, 2, 4)
        torch.testing.assert_close(out.z_pred, reference)
        self.assertFalse(hasattr(model.latent_head, 'feedback_proj'))

    def slice_inputs(self, chunks=3, slices_per_chunk=2, hq=1, wq=1, dim=8):
        # latent grid 2x2 (=4 tokens) from 1x1 merged cells at merge_size 2
        n_video = chunks * slices_per_chunk * hq * wq
        seq = n_video + 2                       # two text tokens for good measure
        mask = torch.zeros(2, seq, dtype=torch.bool)
        mask[:, 1:1 + n_video] = True
        return dict(inputs_embeds=torch.randn(2, seq, dim),
                    attention_mask=torch.ones(2, seq, dtype=torch.long),
                    video_token_mask=mask,
                    context_latent=torch.randn(2, 2, 4, 4),
                    target_latent=torch.randn(2, chunks, 4, 4))

    def slice_predictor(self, mode='autoregressive', **extra):
        backbone = DirectBackbone()
        backbone.config.vision_config = PretrainedConfig(spatial_merge_size=2)
        model = m.QwenVRAEFuturePredictor(backbone, m.QwenVRAEFuturePredictorConfig(
            latent_chunks=3, latent_height=2, latent_width=2, latent_dim=4,
            resampler_dim=8, resampler_depth=1, resampler_heads=2,
            predictor_mode='slice', prediction_mode=mode, context_slices=6, **extra))
        return model, backbone

    def paired_inputs(self):
        """paired_residual aligns chunk k with chunk k, so context carries all 3 chunks."""
        inputs = dict(self.inputs)
        inputs['context_latent'] = torch.randn(2, 3, 2, 4)
        return inputs

    def test_paired_residual_starts_as_the_identity_not_the_dataset_mean(self):
        # The gate for an edit model: with the head's output weight at zero it must emit
        # the source unchanged, chunk for chunk -- so training starts on the persistence
        # baseline it has to beat, instead of spending its budget relearning identity.
        for predictor_mode in ('resampler', 'direct'):
            with self.subTest(predictor_mode=predictor_mode):
                backbone = DirectBackbone() if predictor_mode == 'direct' else None
                model = predictor(mode='paired_residual', weight=0., predictor_mode=predictor_mode,
                                  backbone=backbone).eval()
                torch.nn.init.zeros_(model.latent_head.proj.weight)
                inputs = self.paired_inputs()
                if predictor_mode == 'direct':
                    inputs['inputs_embeds'] = torch.randn(2, 3, 8)
                    inputs.pop('input_ids')
                with torch.no_grad():
                    out = model(**inputs)
                torch.testing.assert_close(out.z_pred, inputs['context_latent'])
                # ... and the loss equals the chunkwise-persistence loss on those rows
                torch.testing.assert_close(
                    out.loss,
                    model.latent_losses(inputs['context_latent'], inputs['target_latent'])['loss'])

    def test_paired_residual_does_not_chain_and_runs_one_backbone_pass(self):
        # Chunk k conditions on chunk k only: no feedback, no second forward. An edit is
        # not an extrapolation, so a chained rollout would be the wrong inductive bias.
        backbone = DirectBackbone()
        model = predictor(mode='paired_residual', weight=0., predictor_mode='direct',
                          backbone=backbone).eval()
        with torch.no_grad():
            inputs = self.paired_inputs()
            inputs['inputs_embeds'] = torch.randn(2, 3, 8)
            inputs.pop('input_ids')
            model(**inputs)
        self.assertEqual(backbone.calls, 1)
        self.assertFalse(hasattr(model.latent_head, 'feedback_proj'))

    def test_paired_residual_rejects_a_misaligned_context(self):
        # A 2-chunk context against a 3-chunk target must stop the run rather than
        # broadcast: silently misaligned pairs would train against the wrong frames.
        model = predictor(mode='paired_residual', weight=0.)
        with self.assertRaises(ValueError):
            model(**self.inputs)          # context_latent is [2, 2, 2, 4]
        with self.assertRaises(ValueError):
            model(**{**self.paired_inputs(), 'context_latent': None})

    def test_paired_residual_without_residual_feedback_ignores_the_source(self):
        model = predictor(mode='paired_residual', weight=0., residual_feedback=False)
        torch.nn.init.zeros_(model.latent_head.proj.weight)
        inputs = self.paired_inputs()
        self.assertFalse(torch.allclose(model(**inputs).z_pred, inputs['context_latent']))

    def test_paired_residual_refuses_a_rollout_weight(self):
        with self.assertRaises(ValueError):
            predictor(mode='paired_residual', weight=1.)

    def test_slice_adds_no_tokens_and_uses_one_backbone_pass(self):
        model, backbone = self.slice_predictor()
        self.assertIsNone(model.resampler)
        self.assertIsNone(model.future_embeddings)
        self.assertIsNotNone(model.slice_expand)
        self.assertTrue(hasattr(model.slice_expand, 'feedback_proj'))
        inputs = self.slice_inputs()
        seq_in = inputs['inputs_embeds'].shape[1]
        out = model(**inputs)
        self.assertEqual(out.z_pred.shape, (2, 3, 4, 4))
        self.assertEqual(backbone.calls, 1)
        # the whole point: Qwen's input length is untouched
        self.assertEqual(backbone.last_seq_len, seq_in)
        out.loss.backward()
        self.assertGreater(backbone.scale.weight.grad.abs().sum().item(), 0)

    def test_slice_conditions_each_step_on_its_own_context_slices(self):
        model, _ = self.slice_predictor()
        model.eval()
        inputs = self.slice_inputs()
        with torch.no_grad():
            base = model(**inputs).z_pred
            bumped = dict(inputs)
            emb = inputs['inputs_embeds'].clone()
            # slices for chunk 2 are video positions 4,5 -> sequence offset 1+4, 1+5
            emb[:, 1 + 4:1 + 6] += 5.0
            bumped['inputs_embeds'] = emb
            moved = model(**bumped).z_pred
        torch.testing.assert_close(base[:, 0], moved[:, 0])
        self.assertFalse(torch.allclose(base[:, 2], moved[:, 2]))

    def test_slice_geometry_mismatch_is_loud(self):
        model, _ = self.slice_predictor()
        inputs = self.slice_inputs()
        wrong = dict(inputs)
        mask = inputs['video_token_mask'].clone()
        mask[:, -1] = True                      # one video token too many
        wrong['video_token_mask'] = mask
        with self.assertRaises(ValueError):
            model(**wrong)
        with self.assertRaises(ValueError):     # mask must cover the sequence
            model(**{**inputs, 'video_token_mask': torch.ones(2, 3, dtype=torch.bool)})

    def test_slice_expansion_is_row_major_on_the_latent_grid(self):
        # cell (h, w) must land on the 2x2 block at (2h, 2w) in the 448-token raster
        exp = m.ContextSliceExpansion(context_dim=1, dim=1, slices_per_chunk=1, merge_size=2)
        with torch.no_grad():
            exp.norm.weight.fill_(1.0); exp.norm.bias.zero_()
            exp.proj.weight.zero_(); exp.proj.bias.copy_(torch.tensor([0., 1., 2., 3.]))
        out = exp(torch.zeros(1, 1, 2, 3, 1))   # H_q=2, W_q=3 -> 4x6 grid
        grid = out.view(1, 4, 6)
        # cell (0,0) occupies rows 0-1, cols 0-1 with the 0,1,2,3 pattern
        torch.testing.assert_close(grid[0, 0, :2], torch.tensor([0., 1.]))
        torch.testing.assert_close(grid[0, 1, :2], torch.tensor([2., 3.]))
        # and the same pattern repeats in the neighbouring cell, i.e. cols 2-3
        torch.testing.assert_close(grid[0, 0, 2:4], torch.tensor([0., 1.]))

    def test_slice_one_shot_runs_without_feedback(self):
        model, _ = self.slice_predictor(mode='one_shot')
        self.assertFalse(hasattr(model.slice_expand, 'feedback_proj'))
        out = model(**{k: v for k, v in self.slice_inputs().items() if k != 'context_latent'})
        self.assertEqual(out.z_pred.shape, (2, 3, 4, 4))

    def interleave_predictor(self, **extra):
        backbone = DirectBackbone()
        model = m.QwenVRAEFuturePredictor(backbone, m.QwenVRAEFuturePredictorConfig(
            latent_chunks=3, latent_height=2, latent_width=2, latent_dim=4,
            resampler_dim=8, resampler_depth=1, resampler_heads=2,
            predictor_mode='interleave', prediction_mode='autoregressive',
            latent_seq_h=1, latent_seq_w=2, **extra))
        return model, backbone

    def interleave_inputs(self, chunks=3):
        return dict(inputs_embeds=torch.randn(2, 3, 8),
                    attention_mask=torch.ones(2, 3, dtype=torch.long),
                    context_latent=torch.randn(2, 2, 4, 4),
                    target_latent=torch.randn(2, chunks, 4, 4))

    def test_interleave_teacher_forcing_is_one_forward(self):
        # The cost claim: teacher forcing puts every chunk's positions in one causal
        # sequence, so training is a single pass rather than one per chunk.
        model, backbone = self.interleave_predictor(rollout_weight=0.)
        out = model(**self.interleave_inputs())
        self.assertEqual(out.z_pred.shape, (2, 3, 4, 4))
        self.assertEqual(backbone.calls, 1)
        # 3 chunks x (1x2 grid) = 6 latent positions appended to a 3-token context
        self.assertEqual(backbone.last_seq_len, 3 + 6)
        out.loss.backward()
        for name in ('interleave.in_proj.weight', 'interleave.expand.proj.weight',
                     'qwen.scale.weight'):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum().item(), 0, name)

    def test_interleave_predicted_feedback_is_sequential(self):
        # and predicted feedback genuinely cannot be: one forward per step, each one
        # longer than the last.
        model, backbone = self.interleave_predictor(rollout_weight=1.)
        out = model(**self.interleave_inputs())
        self.assertEqual(out.z_pred.shape, (2, 3, 4, 4))
        self.assertEqual(backbone.calls, 3)
        self.assertEqual(backbone.last_seq_len, 3 + 6)

    def test_interleave_feeds_the_previous_chunk_not_a_learned_constant(self):
        # Feeding the observed chunk in must actually change the prediction -- otherwise
        # the latent positions are just `direct` with a coarser grid.
        model, _ = self.interleave_predictor(rollout_weight=0.)
        model.eval()
        inputs = self.interleave_inputs()
        with torch.no_grad():
            base = model(**inputs).z_pred
            moved = model(**{**inputs, 'context_latent': inputs['context_latent'] + 3.0}).z_pred
        self.assertFalse(torch.allclose(base, moved))

    def test_interleave_residual_starts_as_persistence(self):
        model, _ = self.interleave_predictor(rollout_weight=1.)
        torch.nn.init.zeros_(model.latent_head.proj.weight)
        observed = self.interleave_inputs()['context_latent'][:, -1]
        inputs = self.interleave_inputs()
        z = model(**inputs).z_pred
        for step in range(z.shape[1]):
            torch.testing.assert_close(z[:, step], inputs['context_latent'][:, -1])
        self.assertEqual(observed.shape, (2, 4, 4))

    def test_interleave_pool_and_expand_round_trip_shapes(self):
        model, _ = self.interleave_predictor()
        mod = model.interleave
        self.assertEqual(mod.tokens_per_chunk, 2)
        self.assertEqual(mod.pool, (2, 1))            # 2x2 latent grid -> 1x2 seq grid
        pooled = mod.pool_latent(torch.randn(2, 4, 4))
        self.assertEqual(pooled.shape, (2, 2, 4))
        out = mod.read_out(torch.randn(2, 2, 8))
        self.assertEqual(out.shape, (2, 4, 8))

    def test_interleave_hidden_feedback_feeds_hidden_states(self):
        model, backbone = self.interleave_predictor(feedback_source='hidden')
        seen = []
        model.interleave.norm_feedback.register_forward_pre_hook(
            lambda mod, args: seen.append(args[0].detach()))
        out = model(**self.interleave_inputs())
        self.assertEqual(out.z_pred.shape, (2, 3, 4, 4))
        # one forward per chunk, and the previous block's hidden states are what came back
        self.assertEqual(backbone.calls, 3)
        self.assertEqual(len(seen), 2)              # steps 1 and 2; step 0 uses the query
        self.assertEqual(seen[0].shape, (2, 2, 8))  # [B, tokens_per_chunk, D_qwen]
        out.loss.backward()
        for name in ('interleave.initial_query', 'interleave.in_proj.weight',
                     'interleave.norm_feedback.weight', 'qwen.scale.weight'):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum().item(), 0, name)

    def test_interleave_hidden_step0_sees_the_observed_chunk(self):
        # The bug this guards: the residual is taken on Z_0, so a model whose step-0 input
        # is a learned constant would be predicting a delta on an unknown quantity.
        model, _ = self.interleave_predictor(feedback_source='hidden')
        model.eval()
        inputs = self.interleave_inputs()
        # A *non-constant* perturbation: `norm_in` is a LayerNorm, so a uniform offset is
        # normalized away and would leave the conditioning untouched -- only the residual
        # would move. That is a property of the normalization, not a missing input.
        torch.manual_seed(1)
        noise = torch.randn_like(inputs['context_latent'])
        with torch.no_grad():
            a = model(**inputs).z_pred[:, 0]
            b = model(**{**inputs,
                         'context_latent': inputs['context_latent'] + noise}).z_pred[:, 0]
        # Remove the residual contribution; what is left is the conditioning's effect.
        self.assertFalse(torch.allclose(b - noise[:, -1], a, atol=1e-5))

    def test_interleave_hidden_rejects_teacher_forcing(self):
        with self.assertRaises(ValueError):
            m.QwenVRAEFuturePredictorConfig(predictor_mode='interleave',
                                            prediction_mode='autoregressive',
                                            feedback_source='hidden', rollout_weight=0.)

    def test_interleave_bptt_depth_detaches_the_hidden_chain(self):
        model, _ = self.interleave_predictor(feedback_source='hidden', bptt_depth=1)
        seen = []
        def keep(mod, args, out):
            out.retain_grad(); seen.append(out)
        model.latent_head.register_forward_hook(keep)
        out = model(**self.interleave_inputs())
        model.latent_losses(out.z_pred[:, -1:],
                            self.interleave_inputs()['target_latent'][:, -1:])['loss'].backward()
        grad = seen[0].grad
        self.assertTrue(grad is None or grad.abs().sum().item() == 0)

    def test_interleave_requires_autoregressive(self):
        with self.assertRaises(ValueError):
            m.QwenVRAEFuturePredictorConfig(predictor_mode='interleave', prediction_mode='one_shot')

    def test_shapes_and_config(self):
        model = predictor(steps=2)
        self.assertEqual(model(**self.inputs).z_pred.shape, (2,2,2,4))
        for key, value in [('context_latent', None), ('target_latent', torch.randn(1,3,2,4))]:
            with self.assertRaises(ValueError):
                model(**{**self.inputs, key: value})
        for kwargs in ({'rollout_steps': 0}, {'rollout_steps': 11}, {'rollout_weight': 1.1}, {'prediction_mode': 'C'}):
            with self.assertRaises(ValueError):
                m.QwenVRAEFuturePredictorConfig(**kwargs)

    def test_residual_rollout_starts_as_persistence(self):
        # The point of the residual parameterization: with the head's output weight at
        # zero the transition is exactly the identity, so the rollout reproduces the last
        # observed chunk at every horizon -- the persistence baseline -- before training.
        model = predictor()
        torch.nn.init.zeros_(model.latent_head.proj.weight)
        observed = self.inputs['context_latent'][:, -1]
        z = model(**self.inputs).z_pred
        for step in range(z.shape[1]):
            torch.testing.assert_close(z[:, step], observed)
        # and without it the prediction has nothing to do with the observed chunk
        plain = predictor(residual_feedback=False)
        torch.nn.init.zeros_(plain.latent_head.proj.weight)
        self.assertFalse(torch.allclose(plain(**self.inputs).z_pred[:, 0], observed))

    def test_residual_head_init_is_small_but_not_dead(self):
        # Small init keeps the starting point at persistence *and* the gradient to
        # upstream modules alive on the very first step; exact zero would kill the latter.
        model = predictor()
        self.assertLess(model.latent_head.proj.weight.std().item(), 0.01)
        model(**self.inputs).loss.backward()
        for name in ('resampler.feedback_proj.weight', 'qwen.embedding.weight'):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad)
            self.assertGreater(grad.abs().sum().item(), 0, name)

    def test_bptt_depth_truncates_the_feedback_chain(self):
        def first_step_grad(depth):
            model = predictor(bptt_depth=depth)
            seen = []
            # Must return None: a forward hook's return value replaces the output.
            def keep(mod, args, out):
                out.retain_grad()
                seen.append(out)
            model.latent_head.register_forward_hook(keep)
            out = model(**self.inputs)
            model.latent_losses(out.z_pred[:, -1:], self.inputs['target_latent'][:, -1:])['loss'].backward()
            grad = seen[0].grad
            return 0.0 if grad is None else grad.abs().sum().item()

        # depth 1 treats each feedback as a constant, so the last horizon's loss cannot
        # reach the first transition; full depth can.
        self.assertEqual(first_step_grad(1), 0.0)
        self.assertGreater(first_step_grad(3), 0.0)

    def test_direct_rollout_uses_one_backbone_pass_and_no_resampler(self):
        backbone = DirectBackbone()
        model = predictor(predictor_mode='direct', backbone=backbone)
        # No resampler at all on this path; the feedback projection rides with the head,
        # which is what modules_to_save keeps in direct mode.
        self.assertIsNone(model.resampler)
        self.assertIsNotNone(model.future_embeddings)
        self.assertTrue(hasattr(model.latent_head, 'feedback_proj'))
        inputs = dict(self.inputs)
        inputs['inputs_embeds'] = torch.randn(2, 3, 8)
        inputs.pop('input_ids')
        out = model(**inputs)
        self.assertEqual(out.z_pred.shape, (2, 3, 2, 4))
        # One forward for the whole rollout, not one per chunk.
        self.assertEqual(backbone.calls, 1)
        out.loss.backward()
        self.assertGreater(backbone.scale.weight.grad.abs().sum().item(), 0)

    def test_direct_rollout_conditions_each_step_on_its_own_chunk(self):
        # The reason the direct path needs no step embedding: chunk k is conditioned on
        # Qwen's hidden state at chunk k's own future positions, so perturbing only the
        # last chunk's conditioning must change only the last prediction.
        backbone = DirectBackbone()
        model = predictor(predictor_mode='direct', backbone=backbone).eval()
        inputs = dict(self.inputs)
        inputs['inputs_embeds'] = torch.randn(2, 3, 8)
        inputs.pop('input_ids')
        with torch.no_grad():
            base = model(**inputs).z_pred
            with torch.no_grad():
                model.future_embeddings.time_embed[-1] += 1.0
            moved = model(**inputs).z_pred
        torch.testing.assert_close(base[:, 0], moved[:, 0])
        self.assertFalse(torch.allclose(base[:, -1], moved[:, -1]))

    def test_ar_time_embed_is_resampler_only(self):
        model = predictor(ar_time_embed=True)
        self.assertEqual(tuple(model.resampler.step_embed.shape), (3, 8))
        queries = [model.resampler.build_queries(2, 'cpu', torch.float32) + model.resampler.step_embed[k]
                   for k in range(3)]
        self.assertFalse(torch.allclose(queries[0], queries[1]))
        # direct carries its time identity in the future positions instead
        direct = predictor(predictor_mode='direct', backbone=DirectBackbone(), ar_time_embed=True)
        self.assertFalse(hasattr(direct.latent_head, 'step_embed'))

    def test_state_roundtrip(self):
        model = predictor().eval()
        clone = predictor().eval()
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(**self.inputs).z_pred, clone(**self.inputs).z_pred)


class TemplateTests(unittest.TestCase):
    def test_latents_survive_encode_collate_and_hook(self):
        import ast
        import os
        import tempfile
        from typing import Any, Dict, List, Optional
        tree = ast.parse(Path(__file__).with_name('template.py').read_text())
        # Module-level helpers come along: `_encode` calls `_paired_residual()`, so
        # compiling the class alone would raise NameError instead of testing anything.
        body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        class Base:
            def _encode(self, inputs): return {'labels': [1]}
            def _data_collator(self, batch, **kwargs): return {'labels': torch.ones(len(batch), 1)}
            def _post_encode(self, model, inputs): return {}
        scope = dict(torch=torch, os=os, Any=Any, Dict=Dict, List=List, Optional=Optional,
                     Qwen3_5Template=Base, StdTemplateInputs=object)
        exec(compile(ast.Module(body=body, type_ignores=[]), 'template.py', 'exec'), scope)
        tpl = scope['Qwen35VRAETemplate']()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'latent.pt')
            latent = torch.randn(3,2,4)
            torch.save({'latent': latent}, path)
            encoded = tpl._encode(types.SimpleNamespace(extra_kwargs={
                'context_latent_path': path, 'target_latent_path': path, 'grid_hw': [1,2]}))
            torch.testing.assert_close(encoded['context_latent'], latent[-1:])
            batch = tpl._data_collator([encoded, encoded])
            self.assertEqual(batch['context_latent'].shape, (2,1,2,4))
            restored = tpl._post_encode(None, batch)
            self.assertIs(restored['context_latent'], batch['context_latent'])
            with self.assertRaises(ValueError):
                tpl._data_collator([encoded, {'labels': [1]}])
            # paired_residual needs every context chunk, not just the last observed one
            import unittest.mock
            with unittest.mock.patch.dict(os.environ, {'PREDICTION_MODE': 'paired_residual'}):
                paired = tpl._encode(types.SimpleNamespace(extra_kwargs={
                    'context_latent_path': path, 'target_latent_path': path, 'grid_hw': [1, 2]}))
            torch.testing.assert_close(paired['context_latent'], latent)
            self.assertEqual(tpl._data_collator([paired, paired])['context_latent'].shape,
                             (2, 3, 2, 4))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
