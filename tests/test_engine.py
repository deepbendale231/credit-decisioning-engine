import numpy as np
import pandas as pd
from conftest import GOOD

from src.adapters import MockBureau, MockKYC, bureau_to_ext_source
from src.engine import DecisionEngine
from src.features import FEATURE_COLUMNS, build_features
from src.policy import Policy
from src.rules import RulesConfig


def test_rules_config_comes_from_the_model_bundle(bundle):
    assert DecisionEngine(bundle).rules_cfg == RulesConfig()


def test_train_serve_parity_features_match_build_features(bundle):
    """The engine must feed the model exactly what build_features produces for the same inputs."""
    engine = DecisionEngine(bundle, Policy(0.05, 0.15))
    out = engine.decide({**GOOD, "ext_source_2": 0.6})
    raw = pd.DataFrame(
        [
            dict(
                AMT_INCOME_TOTAL=600_000.0,
                AMT_CREDIT=300_000.0,
                AMT_ANNUITY=60_000.0,
                AMT_GOODS_PRICE=np.nan,
                DAYS_BIRTH=-35 * 365.25,
                DAYS_EMPLOYED=-8 * 365.25,
                EXT_SOURCE_1=np.nan,
                EXT_SOURCE_2=0.6,
                EXT_SOURCE_3=np.nan,
            )
        ]
    )
    expected = build_features(raw).iloc[0][FEATURE_COLUMNS]
    for k, v in expected.items():
        got = out["features"][k]
        assert (got is None and np.isnan(v)) or np.isclose(got, v), k


def test_bureau_score_fills_ext_source_2_only_when_absent(bundle):
    engine = DecisionEngine(bundle, Policy(0.05, 0.15), bureau=MockBureau({"GOODPAN1": 900}))
    assert engine.decide(GOOD)["features"]["EXT_SOURCE_2"] == bureau_to_ext_source(900) == 1.0
    assert engine.decide({**GOOD, "ext_source_2": 0.3})["features"]["EXT_SOURCE_2"] == 0.3


def test_better_bureau_score_never_raises_pd(bundle):
    pds = []
    for score in range(300, 901, 100):
        e = DecisionEngine(bundle, Policy(0.05, 0.15), bureau=MockBureau({"GOODPAN1": score}))
        pds.append(e.decide(GOOD)["pd"])
    assert all(b <= a + 1e-9 for a, b in zip(pds, pds[1:]))


def test_custom_kyc_adapter_is_used(bundle):
    e = DecisionEngine(bundle, Policy(1.1, 1.1), kyc=MockKYC(unverified={"GOODPAN1"}))
    assert e.decide(GOOD)["failed_rules"] == ["kyc_verified"]
