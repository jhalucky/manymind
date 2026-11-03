from manymind.minds.kyc.data import load_seed
from manymind.minds.kyc.model import Guard, rule_flags


def test_guard_separates_seed_holdout(tmp_path):
    X, y = load_seed()
    model = Guard().fit(X.iloc[:300], y.iloc[:300])
    bait = model.explain("Urgent KYC update required. Click http://bit.ly/x or account blocked")
    ok = model.explain("HDFC Bank: 441920 is your OTP for login. Do not share it with anyone.")
    assert bait["prediction"] == "bait"
    assert ok["prediction"] == "ok"
    assert bait["bait_probability"] > ok["bait_probability"]
    path = tmp_path / "guard.joblib"
    model.save(path)
    restored = Guard().load(path)
    assert restored.explain(bait["text"])["prediction"] == "bait"


def test_rule_flags_catch_link_and_kyc():
    flags = rule_flags("Update Aadhaar KYC now https://evil.example/kyc")
    assert "contains_link" in flags
    assert "kyc_or_aadhaar" in flags
