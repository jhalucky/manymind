from manymind.minds.kyc.data import load_seed


def test_seed_has_bait_and_ok_text():
    text, y = load_seed()
    assert len(text) >= 400
    assert set(y.unique()) == {0, 1}
    assert (y == 1).sum() > 0
    assert (y == 0).sum() > 0
    assert text.str.len().min() > 10
    assert not text.str.contains("Aadhaar number is 1234", case=False).any()
