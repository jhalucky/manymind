
import argparse
from pathlib import Path

from manymind.minds.kyc.model import Guard

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = ROOT / "artifacts" / "kyc" / "model.joblib"


def load_model(path: str | Path | None = None) -> Guard:
    model_path = Path(path) if path is not None else ARTIFACT_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Guard model not found: {model_path}\n"
            "Train it with: python -m manymind.minds.kyc.train"
        )
    return Guard().load(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paste an SMS or WhatsApp. Guard scores KYC / bait risk."
    )
    parser.add_argument("text", nargs="+", help="The message to check.")
    args = parser.parse_args()
    message = " ".join(args.text)
    result = load_model().explain(message)
    print("Guard KYC check")
    print(f"prediction={result['prediction']}")
    print(f"bait_probability={result['bait_probability']:.3f}")
    print(f"flags={', '.join(result['flags']) or 'none'}")
    print(result["advice"])


if __name__ == "__main__":
    main()
