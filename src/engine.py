"""Decision engine: adapters -> features -> Layer A rules -> Layer B model -> Layer C policy."""

from __future__ import annotations

import pandas as pd

from .adapters import BureauClient, KYCClient, MockBureau, MockKYC, bureau_to_ext_source
from .features import DAYS_PER_YEAR, FEATURE_COLUMNS, build_features
from .model import ModelBundle
from .policy import DECLINE, Policy, decide
from .rules import RULE_TEXT, RulesConfig, evaluate_rules


class DecisionEngine:
    def __init__(
        self,
        bundle: ModelBundle,
        policy: Policy | None = None,
        rules_cfg: RulesConfig | None = None,
        bureau: BureauClient | None = None,
        kyc: KYCClient | None = None,
    ):
        self.bundle = bundle
        self.policy = policy or Policy(bundle.meta["policy"]["t1"], bundle.meta["policy"]["t2"])
        self.rules_cfg = rules_cfg or RulesConfig.from_dict(bundle.meta.get("rules", {}))
        self.bureau = bureau or MockBureau()
        self.kyc = kyc or MockKYC()

    def decide(self, req: dict) -> dict:
        verified = self.kyc.verify(req["pan_hash"])
        bureau_score = self.bureau.get_score(req["pan_hash"])

        row = {
            "AMT_INCOME_TOTAL": req["annual_income"],
            "AMT_CREDIT": req["loan_amount"],
            "AMT_ANNUITY": req["annuity"],
            "AMT_GOODS_PRICE": req.get("goods_price"),
            "DAYS_BIRTH": -req["age_years"] * DAYS_PER_YEAR,
            "DAYS_EMPLOYED": None if req.get("years_employed") is None else -req["years_employed"] * DAYS_PER_YEAR,
            "EXT_SOURCE_1": req.get("ext_source_1"),
            # The bureau score stands in for the external score when none is supplied.
            "EXT_SOURCE_2": req.get("ext_source_2")
            if req.get("ext_source_2") is not None
            else bureau_to_ext_source(bureau_score),
            "EXT_SOURCE_3": req.get("ext_source_3"),
        }
        feats = build_features(pd.DataFrame([row]).astype(float))
        feats["KYC_VERIFIED"] = 1.0 if verified else 0.0

        results = evaluate_rules(feats.iloc[0].to_dict(), self.rules_cfg)
        rule_results = [{"rule": name, "passed": passed} for passed, name in results]
        failed = [name for passed, name in results if not passed]

        if failed:
            decision, pd_value = DECLINE, None
            reasons = [{"feature": name, "reason": RULE_TEXT[name]} for name in failed]
        else:
            pd_value = float(self.bundle.predict_pd(feats)[0])
            decision = decide(pd_value, self.policy)
            reasons = self.bundle.explain(feats)[0]

        return {
            "decision": decision,
            "pd": None if pd_value is None else round(pd_value, 4),
            "reasons": reasons,
            "rule_results": rule_results,
            "failed_rules": failed,
            "bureau_score": bureau_score,
            "kyc_verified": verified,
            "thresholds": {"t1": round(self.policy.t1, 4), "t2": round(self.policy.t2, 4)},
            "model_version": self.bundle.version,
            "features": {k: (None if pd.isna(v) else float(v)) for k, v in feats.iloc[0][FEATURE_COLUMNS].items()},
        }
