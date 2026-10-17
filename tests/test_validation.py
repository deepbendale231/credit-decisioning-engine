import pytest

from src import synthetic
from src.validation import DataValidationError, validate_training_frame


@pytest.fixture
def frame():
    return synthetic.make_application_train(n=3000, seed=1)


def test_valid_frame_produces_report(frame):
    r = validate_training_frame(frame)
    assert r["n_rows"] == 3000 and 0.05 < r["default_rate"] < 0.12
    assert r["range_violations"]["days_birth_not_negative"] == 0
    assert set(r["missing_rate"]) >= {"EXT_SOURCE_1", "AMT_ANNUITY"}


def test_missing_column_fails(frame):
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_training_frame(frame.drop(columns=["EXT_SOURCE_2"]))


def test_bad_target_fails(frame):
    frame.loc[0, "TARGET"] = 2
    with pytest.raises(DataValidationError, match="TARGET must be 0/1"):
        validate_training_frame(frame)


def test_too_few_rows_fails(frame):
    with pytest.raises(DataValidationError, match="at least 1,000"):
        validate_training_frame(frame.head(100))


def test_duplicate_ids_fail(frame):
    frame.loc[1, "SK_ID_CURR"] = frame.loc[0, "SK_ID_CURR"]
    with pytest.raises(DataValidationError, match="duplicated"):
        validate_training_frame(frame)


def test_positive_days_birth_fails(frame):
    frame["DAYS_BIRTH"] = -frame["DAYS_BIRTH"]
    with pytest.raises(DataValidationError, match="DAYS_BIRTH"):
        validate_training_frame(frame)
