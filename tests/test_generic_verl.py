"""BUG-27: native heads must preserve SFT masks and gradients in real verl."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import need
from test_multimodel_training import tiny_model


def check_native_backend(family, *, conditional=False):
    need("verl")
    torch = need("torch")
    hf = need("transformers")
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.workers.engine.utils import prepare_micro_batches
    from verl.workers.utils.losses import sft_loss

    model = tiny_model(family)
    if conditional:
        config = model.config.to_dict()
        config.update(num_hidden_layers=4, num_kv_shared_layers=2,
                      layer_types=["sliding_attention", "full_attention"] * 2,
                      use_double_wide_mlp=True)
        config = hf.Gemma4Config(text_config=hf.Gemma4TextConfig(**config),
                                vision_config=None, audio_config=None)
        config._attn_implementation = "eager"
        model = hf.Gemma4ForConditionalGeneration(config)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    samples = []
    for length in (31, 17):
        ids = torch.randint(3, 64, (length,))
        mask = torch.ones(length, dtype=torch.long)
        mask[:4], mask[10:13] = 0, 0
        samples.append(dict(input_ids=ids, loss_mask=mask, position_ids=torch.arange(length)))
    denominator = sum(s["loss_mask"].sum().item() for s in samples)
    baseline = torch.tensor(0.0)
    for sample in samples:
        labels = sample["input_ids"].clone()
        labels[sample["loss_mask"] == 0] = -100
        loss = model(input_ids=sample["input_ids"][None], labels=labels[None], use_cache=False).loss
        weighted = loss * sample["loss_mask"].sum() / denominator
        baseline += weighted.detach()
        weighted.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    batch = tu.get_tensordict(SFTTensorCollator("no_padding")(samples), non_tensor_dict={
        "use_dynamic_bsz": False, "micro_batch_size_per_gpu": 1,
        "max_token_len_per_gpu": 64, "use_remove_padding": False,
        "use_fused_kernels": False, "pad_mode": "no_padding", "pad_token_id": 0,
        "temperature": 1.0, "dp_size": 1, "batch_num_tokens": denominator,
    })
    micros, _ = prepare_micro_batches(batch)
    assert len(micros) == 2
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.pad_to_length = False
    engine.use_ulysses_sp = False
    total = 0.0
    for micro in micros:
        inputs, args = engine.prepare_model_inputs(micro)
        output = model(**inputs, use_cache=False)
        prepared = engine.prepare_model_outputs(output, args, micro, logits_processor_func=None)
        loss, _ = sft_loss(None, prepared, micro)
        total += loss.item()
        loss.backward()
    torch.testing.assert_close(torch.tensor(total), baseline, atol=2e-5, rtol=2e-5)
    for name, parameter in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(parameter.grad, grads[name], atol=2e-5, rtol=2e-4, msg=name)


def test_smollm_real_verl_sft_loss_and_gradients():
    check_native_backend("SmolLM3")


def test_olmo_real_verl_sft_loss_and_gradients():
    check_native_backend("Olmo3")


def test_gemma_real_verl_sft_loss_and_gradients():
    check_native_backend("Gemma4")


def test_gemma_wrapper_shared_kv_checkpointed_sft_loss_and_gradients():
    check_native_backend("Gemma4", conditional=True)


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
