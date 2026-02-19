import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

BASE_DIR = Path.cwd().parents[1]  # équivalent de ../..
TRAIN_PATH = BASE_DIR / "data" / "raw" / "loan_data" / "train_u6lujuX_CVtuZ9i.csv"
TEST_PATH  = BASE_DIR / "data" / "raw" / "loan_data" / "test_Y3wMUE5_7gLdaTN.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def cat_num_sep(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    return cat_cols, num_cols

def fit_preprocessors(train, cat_cols):
    imputer = SimpleImputer(strategy="most_frequent")
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    imputer.fit(train[cat_cols])
    train_cat_imputed = pd.DataFrame(
        imputer.transform(train[cat_cols]),
        columns=cat_cols,
        index=train.index
    )

    encoder.fit(train_cat_imputed)
    return imputer, encoder


def transform(df: pd.DataFrame, cat_cols, num_cols, imputer, encoder):
    df = df.copy()

    # imput cat then encode (uses fitted objects)
    df[cat_cols] = imputer.transform(df[cat_cols])
    df[cat_cols] = encoder.transform(df[cat_cols])

    # num: bfill (souvent mieux: median imputer, mais je garde ton choix)
    if num_cols:
        df[num_cols] = df[num_cols].bfill()

    # target if present
    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    return df

def save_csv(df: pd.DataFrame, name: str):
    df.to_csv(OUTPUT_DIR / name, index=False)

if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # colonnes définies sur train (référence)
    cat_cols, num_cols = cat_num_sep(train)

    # retire la target des features si elle est catégorielle
    if "Loan_Status" in cat_cols:
        cat_cols.remove("Loan_Status")

    imputer, encoder = fit_preprocessors(train, cat_cols)

    train_clean = transform(train, cat_cols, num_cols, imputer, encoder)
    test_clean  = transform(test,  cat_cols, num_cols, imputer, encoder)

    save_csv(train_clean, "train_processed.csv")
    save_csv(test_clean,  "test_processed.csv")
