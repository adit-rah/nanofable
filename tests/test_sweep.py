import os

import tinychat.sweep as sweep


def test_matrix_is_4x2x2():
    m = sweep.sweep_matrix()
    assert len(m) == 16
    assert len(set(m)) == 16
    tiers = {t for t, _, _ in m}
    assert tiers == {"tiny", "small", "medium", "large"}
    assert {p for _, p, _ in m} == {"fp16", "ternary"}
    assert {s for _, _, s in m} == {0, 1}


def test_run_sweep_skips_done(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sweep, "train_run",
                        lambda cfg, prec, seed, run_dir, *a, **k: calls.append(run_dir))

    # Pre-mark one combo as DONE; it must not be trained.
    done_dir = sweep.run_dir_for(str(tmp_path), "tiny", "fp16", 0)
    os.makedirs(done_dir)
    open(os.path.join(done_dir, "DONE"), "w").close()

    sweep.run_sweep(str(tmp_path), "train.bin", "val.bin")

    assert done_dir not in calls
    assert len(calls) == 15  # 16 - 1 skipped


def test_run_sweep_only_subset(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sweep, "train_run",
                        lambda cfg, prec, seed, run_dir, *a, **k: calls.append(run_dir))
    sweep.run_sweep(str(tmp_path), "t.bin", "v.bin", only=[("small", "ternary", 1)])
    assert len(calls) == 1
    assert calls[0].endswith("small_ternary_1")


def test_run_sweep_calls_eval_after_train(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "train_run", lambda *a, **k: None)
    evaled = []
    sweep.run_sweep(str(tmp_path), "t.bin", "v.bin",
                    only=[("tiny", "fp16", 0)], eval_fn=evaled.append)
    assert len(evaled) == 1 and evaled[0].endswith("tiny_fp16_0")
