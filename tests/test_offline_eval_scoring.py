"""The container-free eval must not be the reason a finished checkpoint is unmeasured.

The 4B SFT run completed 82/82 steps and still ended up recorded as "NO benchmark
produced a score, so this checkpoint is unmeasured" -- not because the sandbox was
blocked (it was, which is why `06b_eval_offline.py` exists at all) but because the
fallback itself crashed:

    File "scripts/06b_eval_offline.py", line 243, in score_rows
    RuntimeError: Expected all tensors to be on the same device, but got target is on
    cuda:7, different from other tensors on cuda:1

`load_model` uses `device_map="auto"`, and Qwen3.5 ties the output head to the input
embedding, which accelerate keeps on the FIRST device while `hidden` comes off the LAST
decoder block. So `head(hidden)` and `targets` end up on different cards by design, and
`cross_entropy` refuses to move either one. One `.to(logits.device)` fixes it.

That single-device-vs-sharded split cannot be reproduced on a machine with one GPU, let
alone on CPU, so it is pinned at the source level -- deliberately, with the traceback
above quoted so the next person editing this loop knows what the line is for.

The rest of the file is the arithmetic the loop is supposed to implement, checked on CPU
against an independent full-logits computation: position i is predicted from hidden i-1,
position 0 is never a target, and the chunking is only a memory strategy so it must not
change the number. A chunked reimplementation of cross-entropy is exactly the kind of
code that is off by one and still looks plausible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, load_script, need  # noqa: E402

SOURCE = (ROOT / "scripts" / "06b_eval_offline.py").read_text(encoding="utf-8")

HIDDEN = 16
VOCAB = 32
LENGTH = 40


def build_stub():
    """A model exposing only what score_rows uses: get_decoder + get_output_embeddings."""
    torch = need("torch")
    from torch import nn

    torch.manual_seed(7)

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(VOCAB, HIDDEN)
            self.mix = nn.Linear(HIDDEN, HIDDEN)

        def forward(self, input_ids):  # noqa: ANN001 - matches the HF call signature
            hidden = self.mix(self.embed(input_ids))

            class Out:
                last_hidden_state = hidden

            return Out()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = Decoder()
            self.head = nn.Linear(HIDDEN, VOCAB, bias=False)

        def get_decoder(self):
            return self.decoder

        def get_output_embeddings(self):
            return self.head

    model = Model().eval()
    return torch, model


def reference_score(torch, model, ids: list[int], mask: list[int]) -> tuple[float, int, int]:
    """Full-logits NLL over supervised positions -- the thing score_rows approximates.

    Written the obvious way (materialize every logit, shift by one) precisely because
    that is what the chunked loop must agree with.
    """
    with torch.inference_mode():
        hidden = model.get_decoder()(torch.tensor([ids]))
        logits = model.get_output_embeddings()(hidden.last_hidden_state[0]).float()
    targets = torch.tensor(ids)
    keep = [i for i, m in enumerate(mask) if m and i > 0]
    nll = 0.0
    correct = 0
    for i in keep:
        row = logits[i - 1]
        nll += torch.nn.functional.cross_entropy(
            row.unsqueeze(0), targets[i].view(1), reduction="sum").item()
        correct += int(row.argmax(-1).item() == ids[i])
    return nll, correct, len(keep)


def rows_for_test() -> list[tuple[list[int], list[int]]]:
    ids = [(i * 7 + 3) % VOCAB for i in range(LENGTH)]
    # Two supervised spans plus position 0 set, which must be ignored.
    mask = [0] * LENGTH
    mask[0] = 1
    for i in range(5, 12):
        mask[i] = 1
    for i in range(25, LENGTH):
        mask[i] = 1
    return [(ids, mask)]


def test_the_chunked_loop_agrees_with_a_full_logits_computation():
    torch, model = build_stub()
    module = load_script("06b_eval_offline")
    (ids, mask), = rows_for_test()
    want_nll, want_correct, want_n = reference_score(torch, model, ids, mask)

    out = module.score_rows(model, [(ids, mask)], chunk=4, progress_every=10**6)
    assert out["supervised_tokens"] == want_n, (
        f"scored {out['supervised_tokens']} positions, not {want_n}: position 0 has no "
        f"predecessor and must be dropped even when the mask sets it"
    )
    assert abs(out["loss"] - want_nll / want_n) < 1e-4, (
        f"chunked loss {out['loss']} vs full-logits {want_nll / want_n} -- an off-by-one "
        f"in `hidden[block - 1]` looks exactly like this"
    )
    assert abs(out["top1_accuracy"] - want_correct / want_n) < 1e-9


def test_the_chunk_size_is_a_memory_knob_and_not_a_result_knob():
    # --chunk exists only because a 32k x 150k logit tensor is ~10 GB. If lowering it to
    # survive an OOM also changed the number, two checkpoints scored on two machines
    # would not be comparable.
    torch, model = build_stub()
    module = load_script("06b_eval_offline")
    rows = rows_for_test()
    losses = {
        chunk: module.score_rows(model, rows, chunk=chunk, progress_every=10**6)["loss"]
        for chunk in (1, 3, 8, 4096)
    }
    assert len({round(v, 6) for v in losses.values()}) == 1, losses


def test_an_all_zero_mask_row_reports_no_tokens_rather_than_dividing_by_zero():
    torch, model = build_stub()
    module = load_script("06b_eval_offline")
    ids = [1, 2, 3, 4, 5]
    out = module.score_rows(model, [(ids, [0] * len(ids))], chunk=4, progress_every=10**6)
    assert out["supervised_tokens"] == 0
    assert out["per_row"][0]["loss"] is None and out["per_row"][0]["top1"] is None
    assert out["top1_accuracy"] is None


def test_the_gold_tensor_follows_the_logits_device():
    """The 4B crash. Unreproducible on one device, so asserted on the source.

    `device_map="auto"` + tied embeddings puts the head's output and `targets` on
    different cards; cross_entropy raises instead of moving anything.
    """
    assert 'device_map="auto"' in SOURCE, (
        "load_model no longer shards the model, so this guard may be unnecessary -- but "
        "check before deleting it: it costs chunk*8 bytes of H2D-free device copy"
    )
    assert "gold = targets[block].to(logits.device)" in SOURCE, (
        "gold is taken from `targets` without following the logits again. On a "
        "device_map='auto' load that is the RuntimeError that left the finished 4B "
        "checkpoint with no eval at all"
    )
    head = SOURCE.index("logits = head(hidden[block - 1])")
    gold = SOURCE.index("gold = targets[block]")
    assert head < gold, "gold must be placed after the logits it follows"


def test_a_missing_checkpoint_directory_says_so_instead_of_blaming_the_argument():
    """The 27B run's second-order confusion.

    Its training stage failed, so `out-hf-full` was never written, and the eval then
    printed three copies of

        OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
                 '/llm-align/liuchonghan/rst/out-hf-full'

    -- transformers falling through to the Hub resolver, once per auto class. Read
    literally it accuses the operator of passing a malformed name.
    """
    module = load_script("06b_eval_offline")
    missing = "/nonexistent-path-for-a-test/rst/out-hf-full"
    try:
        module.load_model(missing, "bfloat16")
    except SystemExit as exc:
        text = str(exc)
        assert missing in text and "does not exist" in text, text
        assert "Repo id" not in text
    else:
        raise AssertionError("a nonexistent checkpoint path was handed to transformers")


def test_the_argmax_comparison_is_not_a_second_cross_device_site():
    # `logits.argmax(-1) == gold` is the other place the two devices meet, and an
    # elementwise compare across devices raises the same way cross_entropy does. It is
    # safe only because `gold` was already moved; if someone reverts to a separately
    # built tensor there, this catches it.
    loop = SOURCE[SOURCE.index("for start in range(0, positions.numel(), chunk)"):
                  SOURCE.index("n = int(positions.numel())")]
    assert "logits.argmax(-1) == gold" in loop
    assert loop.count("targets[block]") == 1, (
        "targets is indexed more than once inside the chunk loop, so one of the uses is "
        "not going through the device-following `gold`"
    )


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
