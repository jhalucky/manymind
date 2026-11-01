from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH=PROJECT_ROOT / "data" / "sentinel" / "creditcard"/ "creditcard.csv"
FEATURES_NAMES=[f"V{i}" for i in range(1,29)] + ["Amount"]

def load_creditcard(path=CSV_PATH):
    df=pd.read_csv(path)
    X=df[FEATURES_NAMES]
    y=df["Class"].astype(int)
    return X,y

# if __name__ == "__main__":
#     X,y=loan_creditcard()
#     print(X.shape)
#     print(y.shape)
#     print(X.head())
#     print(y.head())