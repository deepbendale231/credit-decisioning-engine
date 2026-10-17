from src.adapters import BUREAU_MAX, BUREAU_MIN, MockBureau, MockKYC, bureau_to_ext_source


def test_bureau_is_deterministic_and_in_range():
    b = MockBureau()
    scores = [b.get_score(f"pan{i}") for i in range(500)]
    assert scores == [b.get_score(f"pan{i}") for i in range(500)]
    assert all(BUREAU_MIN <= s <= BUREAU_MAX for s in scores) and len(set(scores)) > 100


def test_bureau_override_and_mapping():
    assert MockBureau({"x": 750}).get_score("x") == 750
    assert bureau_to_ext_source(300) == 0.0 and bureau_to_ext_source(900) == 1.0
    assert bureau_to_ext_source(5000) == 1.0 and bureau_to_ext_source(0) == 0.0


def test_kyc_rules():
    k = MockKYC(unverified={"bad"})
    assert k.verify("ok") and not k.verify("bad") and not k.verify("KYCFAIL-1")
