import json

import pytest

from src.pipeline import RunConfig, run
from src.results_table import main as results_main


def _cfg(tmp_path, name="a", **kw):
    return RunConfig(
        synthetic=True,
        out_dir=tmp_path / name,
        model_path=tmp_path / name / "m.joblib",
        n_trials=2,
        n_boot=25,
        synthetic_rows=5000,
        synthetic_prev_rows=4000,
        **kw,
    )


@pytest.fixture(scope="module")
def run_a(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    return tmp, run(_cfg(tmp))


def test_artifacts_written_and_synthetic_is_stamped(run_a):
    tmp, m = run_a
    out = tmp / "a"
    for f in ["metrics.json", "MODEL_CARD.md", "funnel_summary.csv", "test_scores.parquet", "frontier_ml.csv", "m.joblib"]:
        assert (out / f).exists(), f
    for fig in ["roc", "calibration", "deciles", "frontier", "shap_importance", "fairness", "funnel_outcomes"]:
        assert (out / "figures" / f"{fig}.png").stat().st_size > 5_000
    assert m["meta"]["data_source"] == "synthetic" and m["meta"]["resume_safe"] is False
    assert "SYNTHETIC" in (out / "MODEL_CARD.md").read_text()
    assert json.loads((out / "metrics.json").read_text())["meta"]["model_version"] == m["meta"]["model_version"]


def test_split_is_leak_free_and_policy_thresholds_ordered(run_a):
    _, m = run_a
    meta = m["meta"]
    assert meta["n_train"] + meta["n_val"] + meta["n_test"] == meta["n_rows"]
    assert 0 <= m["policy"]["t1"] <= m["policy"]["t2"] <= 1.0
    assert abs(sum(m["decision_mix_test"].values()) - 1) < 1e-3


def test_tuned_rules_benchmark_is_never_looser_than_current(run_a):
    _, m = run_a
    c = m["comparison"]
    assert c["tuned_rules_only_same_target"]["approval_rate"] <= c["current_rules_only"]["approval_rate"] + 1e-9
    lo, hi = c["approval_uplift_vs_tuned_rules_pts"]["ci95"]
    assert lo <= c["approval_uplift_vs_tuned_rules_pts"]["point"] <= hi


def test_gender_is_audited_but_never_a_feature(run_a):
    _, m = run_a
    assert "CODE_GENDER" not in json.dumps(m["explainability"]["mean_abs_shap"])
    assert m["fairness"]["rules_plus_ml"]["gender"]["gap_pts"] is not None
    assert 0.4 < m["fairness"]["proxy_test"]["gender_from_features_auc"] < 0.75


def test_pipeline_is_deterministic(run_a, tmp_path):
    _, a = run_a
    b = run(_cfg(tmp_path, "b"))
    for k in ("discrimination_test", "comparison", "policy", "fairness", "decision_mix_test", "models"):
        assert a[k] == b[k], k


def test_results_table_refuses_to_publish_synthetic_numbers(run_a, tmp_path):
    out, _ = run_a
    readme = tmp_path / "README.md"
    readme.write_text("# x\n<!-- RESULTS:START -->\nSTALE-SENTINEL\n<!-- RESULTS:END -->\n")
    args = ["--metrics", str(out / "a" / "metrics.json"), "--readme", str(readme), "--write-readme"]
    assert results_main(args) == 1 and "STALE-SENTINEL" in readme.read_text()
    assert results_main(args + ["--allow-synthetic"]) == 0
    assert "SYNTHETIC DATA" in readme.read_text() and "STALE-SENTINEL" not in readme.read_text()


def test_real_data_path_reads_kaggle_schema_csvs(tmp_path):
    """The real-data branch (CSV + DuckDB file reads) exercised on CSVs written in the Kaggle schema."""
    from src import synthetic

    data = tmp_path / "data"
    data.mkdir()
    synthetic.make_application_train(n=5000, seed=5).assign(NAME_EXTRA="ignored").to_csv(
        data / "application_train.csv", index=False
    )
    synthetic.make_previous_application(n=6000, seed=6).to_csv(data / "previous_application.csv", index=False)
    m = run(RunConfig(data_dir=data, out_dir=tmp_path / "out", model_path=tmp_path / "m.joblib", n_trials=1, n_boot=15))
    assert m["meta"]["data_source"] == "home_credit" and m["meta"]["resume_safe"] is True
    assert m["funnel"]["total_rows"] == 6000
    assert "SYNTHETIC" not in (tmp_path / "out" / "MODEL_CARD.md").read_text()


def test_pipeline_without_previous_application_skips_funnel(tmp_path):
    from src import synthetic

    data = tmp_path / "data"
    data.mkdir()
    synthetic.make_application_train(n=4000, seed=5).to_csv(data / "application_train.csv", index=False)
    m = run(RunConfig(data_dir=data, out_dir=tmp_path / "out", model_path=tmp_path / "m.joblib", n_trials=1, n_boot=10))
    assert m["funnel"] is None


def test_cli_fails_clearly_when_data_is_missing(tmp_path):
    from src.pipeline import main

    with pytest.raises(SystemExit, match=r"application_train\.csv not found"):
        main(["--data-dir", str(tmp_path)])


def test_cli_quick_synthetic_run(tmp_path, capsys):
    from src.pipeline import main

    main(["--synthetic", "--quick", "--out-dir", str(tmp_path / "r"), "--model-path", str(tmp_path / "m.joblib")])
    assert "synthetic data" in capsys.readouterr().out and (tmp_path / "r" / "metrics.json").exists()


def test_cli_defaults_map_onto_runconfig_fields(monkeypatch, tmp_path):
    """Regression: RunConfig was once built positionally, so adding a field shifted n_trials to 300."""
    import src.pipeline as P

    seen = {}
    monkeypatch.setattr(P, "run", lambda cfg: seen.setdefault("cfg", cfg) and {"meta": {"resume_safe": False}})
    P.main(["--synthetic", "--out-dir", str(tmp_path)])
    cfg, defaults = seen["cfg"], P.RunConfig()
    assert (cfg.n_trials, cfg.n_boot, cfg.seed, cfg.refer_share, cfg.target_reduction) == (
        defaults.n_trials,
        defaults.n_boot,
        defaults.seed,
        defaults.refer_share,
        defaults.target_reduction,
    )
    assert cfg.synthetic is True and cfg.out_dir == tmp_path


def test_cli_quick_flag_overrides_effort(monkeypatch, tmp_path):
    import src.pipeline as P

    seen = {}
    monkeypatch.setattr(P, "run", lambda cfg: seen.setdefault("cfg", cfg) and {"meta": {"resume_safe": False}})
    P.main(["--synthetic", "--quick", "--refer-share", "0.2", "--out-dir", str(tmp_path)])
    assert (seen["cfg"].n_trials, seen["cfg"].n_boot, seen["cfg"].refer_share) == (3, 60, 0.2)
