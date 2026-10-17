"""Vendor adapters. The engine depends on these interfaces only, so swapping a mock for
a real bureau / e-KYC vendor client does not touch the engine."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Protocol

BUREAU_MIN, BUREAU_MAX = 300, 900


class BureauClient(Protocol):
    def get_score(self, pan_hash: str) -> int: ...


class KYCClient(Protocol):
    def verify(self, pan_hash: str) -> bool: ...


class MockBureau:
    """Deterministic fake bureau: same pan_hash -> same score in [300, 900]."""

    def __init__(self, overrides: Mapping[str, int] | None = None):
        self.overrides = dict(overrides or {})

    def get_score(self, pan_hash: str) -> int:
        if pan_hash in self.overrides:
            return int(self.overrides[pan_hash])
        digest = int(hashlib.sha256(pan_hash.encode()).hexdigest(), 16)
        return BUREAU_MIN + digest % (BUREAU_MAX - BUREAU_MIN + 1)


class MockKYC:
    """Verified unless the pan_hash starts with 'KYCFAIL' (or is in `unverified`)."""

    def __init__(self, unverified: set[str] | None = None):
        self.unverified = set(unverified or ())

    def verify(self, pan_hash: str) -> bool:
        return not (pan_hash.startswith("KYCFAIL") or pan_hash in self.unverified)


def bureau_to_ext_source(score: int) -> float:
    """Stand-in mapping from a 300-900 bureau score to the 0-1 EXT_SOURCE scale."""
    return (min(max(score, BUREAU_MIN), BUREAU_MAX) - BUREAU_MIN) / (BUREAU_MAX - BUREAU_MIN)
