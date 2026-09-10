"""BUG-27: native HF SFT losses and DPO gradients must agree for every new family."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import load_script, need, skip

DPO = load_script("dpo_common")


def tiny_model(family):
    torch = need("torch")
    hf = need("transformers")
    if not hasattr(hf, family + "ForCausalLM"):
        skip(f"installed Transformers has no {family}; use a version with this architecture")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    kwargs = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
              "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
              "max_position_embeddings": 64, "bos_token_id": 1, "eos_token_id": 2,
              "pad_token_id": 0, "tie_word_embeddings": True, "attention_dropout": 0.0}
    if family == "Phi3":
        kwargs.update(original_max_position_embeddings=64, resid_pdrop=0.0, embd_pdrop=0.0)
    if family == "Gemma4":
        kwargs.update(head_dim=8, global_head_dim=8, hidden_size_per_layer_input=8,
                      vocab_size_per_layer_input=64, final_logit_softcapping=0.2,
                      sliding_window=8, layer_types=["sliding_attention", "full_attention"])
    config = getattr(hf, family + ("TextConfig" if family == "Gemma4" else "Config"))(**kwargs)
    config._attn_implementation = "eager"
    return getattr(hf, family + "ForCausalLM")(config).eval()


def compare_native_loss_and_gradients(family):
    torch = need("torch")
    model = tiny_model(family)
    ids = [1, 12, 24, 13, 15, 19, 21, 2]
    mask = [0, 0, 1, 1, 0, 0, 1, 1]
    inputs = torch.tensor([ids])
    labels = inputs.clone()
    labels[torch.tensor([mask]) == 0] = -100
    native = model(input_ids=inputs, labels=labels, use_cache=False)
    native_sum = -native.loss.float() * sum(mask)
    native_sum.backward()
    expected_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    logp, count = DPO.masked_logprob_sum(model, ids, mask, chunk=2, grad=True)
    assert count == sum(mask)
    torch.testing.assert_close(logp.float(), native_sum.detach(), rtol=2e-5, atol=2e-5)
    logp.backward()
    for name, param in model.named_parameters():
        if name in expected_grads:
            assert param.grad is not None, name
            torch.testing.assert_close(param.grad, expected_grads[name], rtol=1e-4, atol=2e-5, msg=name)
    # A real preference update must have finite loss and move a trainable weight.
    model.zero_grad(set_to_none=True)
    ref_c, _ = DPO.masked_logprob_sum(model, ids, mask, chunk=2)
    rejected = [1, 12, 25, 14, 15, 19, 22, 2]
    ref_r, _ = DPO.masked_logprob_sum(model, rejected, mask, chunk=2)
    policy_c, _ = DPO.masked_logprob_sum(model, ids, mask, chunk=2, grad=True)
    policy_r, _ = DPO.masked_logprob_sum(model, rejected, mask, chunk=2, grad=True)
    loss, _ = DPO.dpo_loss(policy_c, policy_r, ref_c, ref_r, beta=0.1)
    torch.testing.assert_close(loss, torch.tensor(2.0).double().log())
    loss.backward()
    param = model.get_output_embeddings().weight
    before = param.detach().clone()
    assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    torch.optim.SGD(model.parameters(), lr=1e-3).step()
    assert not torch.equal(before, param)


def test_llama_native_sft_and_dpo():
    compare_native_loss_and_gradients("Llama")


def test_phi_native_sft_and_dpo():
    compare_native_loss_and_gradients("Phi3")


def test_smollm_native_sft_and_dpo():
    compare_native_loss_and_gradients("SmolLM3")


def test_olmo_native_sft_and_dpo():
    compare_native_loss_and_gradients("Olmo3")


def test_gemma_native_softcap_sft_and_dpo():
    compare_native_loss_and_gradients("Gemma4")


def test_gemma_conditional_generation_loader_and_text_decoder_match():
    torch = need("torch")
    hf = need("transformers")
    text = tiny_model("Gemma4")
    config = hf.Gemma4Config(text_config=text.config, vision_config=None, audio_config=None)
    config._attn_implementation = "eager"
    model = hf.Gemma4ForConditionalGeneration(config).eval()
    ids = [1, 12, 13, 2]
    mask = [0, 0, 1, 1]
    native = model(input_ids=torch.tensor([ids]), labels=torch.tensor([[-100, -100, 13, 2]]))
    logp, n = DPO.masked_logprob_sum(model, ids, mask, chunk=1)
    torch.testing.assert_close(logp.float(), -native.loss * n, rtol=1e-5, atol=1e-5)
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        reloaded, auto_class = DPO.load_model(directory, dtype=torch.float32)
        assert auto_class in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM")
        actual, _ = DPO.masked_logprob_sum(reloaded.eval(), ids, mask, chunk=1)
        torch.testing.assert_close(actual, logp)


def check_dpo_reference_training_save_and_stale_dataset(family, profile):
    pd = need("pandas")
    need("pyarrow")
    need("torch")
    from test_model_tokenization import TOK, tokenizer

    reference = load_script("18_dpo_ref_logprobs")
    train = load_script("19_train_dpo")
    model = tiny_model(family)
    tok = tokenizer(profile)
    model.resize_token_embeddings(len(tok))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint, pairs, ref, out = [root / name for name in ("model", "pairs", "ref", "out")]
        checkpoint.mkdir()
        pairs.mkdir()
        model.save_pretrained(checkpoint)
        tok.save_pretrained(checkpoint)
        rows = []
        for i in range(2):
            rows.append({"pair_id": str(i), "task_group_id": "task" + str(i),
                         "chosen_input_ids": [1, 12, 13, 2], "chosen_loss_mask": [0, 0, 1, 1],
                         "rejected_input_ids": [1, 12, 14, 2], "rejected_loss_mask": [0, 0, 1, 1],
                         "chosen_n_tokens": 4, "rejected_n_tokens": 4,
                         "chosen_n_trained": 2, "rejected_n_trained": 2})
        path = pairs / "dpo_train.parquet"
        identity = TOK.tokenization_identity(tok, profile)
        TOK.write_tokenized_parquet(pd.DataFrame(rows), path, identity)
        base_args = ["--pairs", str(pairs), "--model-path", str(checkpoint),
                     "--max-seq-len", "64", "--logit-chunk", "2"]
        ref_args = ["reference", *base_args, "--out", str(ref), "--dtype", "fp32"]
        with patch.object(sys, "argv", ref_args):
            assert reference.main() == 0
        with patch.object(sys, "argv", ["train", *base_args, "--ref-logps", str(ref),
                                        "--out", str(out), "--max-steps", "1", "--grad-accum", "1"]):
            assert train.main() == 0
        assert (out / "hf/config.json").is_file() and (out / "hf/tokenizer.json").is_file()
        TOK.validate_training_data([path], out / "hf")
        assert json.loads((ref / "ref_logps_manifest.json").read_text())["pairs_sha256"]
        provenance = ref / "ref_logps_provenance.json"
        original = provenance.read_text()
        legacy = json.loads(original)
        legacy.pop("pairs_sha256")
        provenance.write_text(json.dumps(legacy))
        with patch.object(reference, "validate_pair_tokenization", return_value="qwen3_5"), \
                patch.object(sys, "argv", ref_args):
            assert reference.main() == 0
        assert json.loads(provenance.read_text())["pairs_sha256"] is None
        assert json.loads((ref / "ref_logps_manifest.json").read_text())["pairs_sha256"] is None
        provenance.write_text(original)
        # Keep lengths, masks, pair IDs and template fingerprint identical; alter
        # just one target token. Resuming old probabilities must now fail.
        rows[0]["chosen_input_ids"][2] = 15
        TOK.write_tokenized_parquet(pd.DataFrame(rows), path, identity)
        with patch.object(sys, "argv", ref_args):
            try:
                reference.main()
            except SystemExit as exc:
                assert "pairs_sha256" in str(exc), str(exc)
            else:
                raise AssertionError("stale reference logprobs were resumed")


def test_real_dpo_reference_training_save_and_stale_dataset_rejection():
    check_dpo_reference_training_save_and_stale_dataset("Llama", "llama3")


def test_smollm_dpo_reference_training_save_and_stale_dataset_rejection():
    check_dpo_reference_training_save_and_stale_dataset("SmolLM3", "smollm3")


def test_olmo_dpo_reference_training_save_and_stale_dataset_rejection():
    check_dpo_reference_training_save_and_stale_dataset("Olmo3", "olmo3")


def test_gemma_dpo_reference_training_save_and_stale_dataset_rejection():
    check_dpo_reference_training_save_and_stale_dataset("Gemma4", "gemma4")


def test_tied_embeddings_share_one_fsdp_group_with_both_forward_entry_points():
    torch = need("torch")
    train = load_script("19_train_dpo")
    model = tiny_model("Llama")
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    assert embedding.weight is head.weight
    calls = []
    with patch("torch.distributed.fsdp.fully_shard", side_effect=lambda module, **kw: calls.append((module, kw))):
        train.shard_model(model, world_size=2, param_dtype=torch.bfloat16)
    shared = [(module, kw) for module, kw in calls if isinstance(module, list)]
    assert len(shared) == 1 and shared[0][0] == [embedding, head]
    assert shared[0][1]["reshard_after_forward"] is False
    assert head not in [module for module, _ in calls if not isinstance(module, list)]


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
