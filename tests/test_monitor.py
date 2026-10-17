import numpy as np

from src import synthetic
from src.audit import AuditLog
from src.engine import DecisionEngine
from src.monitor import MIN_RECORDS, build_report
from src.policy import Policy


def _none_if_nan(v):
    return None if np.isnan(v) else float(v)


def _log(bundle, tmp_path, n, income_scale=1.0, seed=11):
    """Replay realistic traffic (same generator as training, new seed) through the engine into an audit log."""
    traffic = synthetic.make_application_train(n=n, seed=seed)
    audit, engine = AuditLog(tmp_path / f"a{income_scale}.jsonl"), DecisionEngine(bundle, Policy(0.05, 0.15))
    for i, r in enumerate(traffic.itertuples()):
        emp = None if r.DAYS_EMPLOYED > 0 else -r.DAYS_EMPLOYED / 365.25
        req = dict(
            pan_hash=f"P{i}",
            age_years=int(-r.DAYS_BIRTH / 365.25),
            annual_income=r.AMT_INCOME_TOTAL * income_scale,
            loan_amount=r.AMT_CREDIT,
            annuity=r.AMT_ANNUITY,
            goods_price=r.AMT_GOODS_PRICE,
            years_employed=emp,
            ext_source_1=_none_if_nan(r.EXT_SOURCE_1),
            ext_source_2=_none_if_nan(r.EXT_SOURCE_2),
            ext_source_3=_none_if_nan(r.EXT_SOURCE_3),
        )
        audit.write(f"r{i}", req, engine.decide(req), 1.0)
    return audit.read()


def test_insufficient_data_skips_drift(bundle, tmp_path):
    r = build_report(_log(bundle, tmp_path, 20), bundle.meta["reference"])
    assert not r["sufficient_data"] and r["psi"] == {} and r["n_decisions"] == 20


def test_stable_traffic_raises_no_alert(bundle, tmp_path):
    r = build_report(_log(bundle, tmp_path, MIN_RECORDS + 600), bundle.meta["reference"])
    assert r["sufficient_data"] and not r["alert"], r["psi"]
    assert set(r["psi"]) >= {"pd", "EXT_SOURCE_MEAN", "AMT_INCOME_TOTAL"}


def test_income_shift_is_detected(bundle, tmp_path):
    r = build_report(_log(bundle, tmp_path, MIN_RECORDS + 600, income_scale=0.25), bundle.meta["reference"])
    assert r["psi"]["AMT_INCOME_TOTAL"] > 0.25 and r["alert"]


def test_monitor_cli_exit_codes_and_json(bundle, tmp_path, capsys):
    import json

    from src.model import save_bundle
    from src.monitor import main

    save_bundle(bundle, tmp_path / "m.joblib")
    stable = _log(bundle, tmp_path, MIN_RECORDS + 600)  # noqa: F841 - written to tmp_path/a1.0.jsonl
    assert main(["--log", str(tmp_path / "a1.0.jsonl"), "--model", str(tmp_path / "m.joblib")]) == 0
    assert "PSI pd" in capsys.readouterr().out
    _log(bundle, tmp_path, MIN_RECORDS + 600, income_scale=0.25)
    assert main(["--log", str(tmp_path / "a0.25.jsonl"), "--model", str(tmp_path / "m.joblib"), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["alert"] is True


def test_monitor_cli_reports_when_data_is_insufficient(bundle, tmp_path, capsys):
    from src.model import save_bundle
    from src.monitor import main

    save_bundle(bundle, tmp_path / "m.joblib")
    _log(bundle, tmp_path, 10)
    assert main(["--log", str(tmp_path / "a1.0.jsonl"), "--model", str(tmp_path / "m.joblib")]) == 0
    assert "need >=" in capsys.readouterr().out
