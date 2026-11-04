from manymind.minds.kyc.data import load_seed, load_spam


def test_seed_has_bait_and_ok_text():
    text, y = load_seed()
    assert len(text) >= 400
    assert set(y.unique()) == {0, 1}
    assert (y == 1).sum() > 0
    assert (y == 0).sum() > 0
    assert text.str.len().min() > 10
    assert not text.str.contains("Aadhaar number is 1234", case=False).any()


def test_spam_csv_is_text_and_label_only():
    from pathlib import Path

    path = Path("data/guard/spam.csv")
    if not path.is_file():
        import pytest

        pytest.skip("spam.csv is not present")
    text, y = load_spam(path)
    assert set(y.unique()) <= {0, 1}
    assert len(text) > 1000
    assert (y == 1).sum() > 0
    assert (y == 0).sum() > 0
