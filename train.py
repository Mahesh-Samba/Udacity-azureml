from sklearn.linear_model import LogisticRegression
import argparse
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from azureml.core.run import Run


def clean_data(data):
    """
    Cleans and encodes the Bank Marketing dataset.
    Accepts a pandas DataFrame.
    """

    # Copy dataframe
    x_df = data.copy()

    # Remove missing values
    x_df = x_df.dropna()

    # -------------------------
    # Convert month to numeric
    # -------------------------

    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    x_df["month"] = x_df["month"].map(months)

    # -------------------------
    # Binary Encoding
    # -------------------------

    binary_columns = [
        "default",
        "housing",
        "loan",
    ]

    for column in binary_columns:
        x_df[column] = x_df[column].map({
            "yes": 1,
            "no": 0
        })

    # -------------------------
    # One-Hot Encoding
    # -------------------------

    categorical_columns = [
        "job",
        "marital",
        "education",
        "contact",
        "poutcome",
    ]

    x_df = pd.get_dummies(
        x_df,
        columns=categorical_columns,
        drop_first=True
    )

    # -------------------------
    # Target Variable
    # -------------------------

    y_df = x_df.pop("deposit").map({
        "yes": 1,
        "no": 0
    })

    return x_df, y_df

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse Regularization Strength"
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum Iterations"
    )

    args = parser.parse_args()

    run = Run.get_context()

    # Log Hyperparameters
    run.log("Regularization Strength", args.C)
    run.log("Max Iterations", args.max_iter)

    # -------------------------------------------------
    # Load Dataset (LOCAL CSV)
    # -------------------------------------------------

    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "bank.csv"
    )

    print(f"Loading dataset from: {dataset_path}")

    df = pd.read_csv(dataset_path)

    print("Dataset Shape:", df.shape)

    # Clean Dataset
    x, y = clean_data(df)

    print("Feature Shape:", x.shape)
    print("Target Shape :", y.shape)

    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train Model
    model = LogisticRegression(
    C=args.C,
    max_iter=max(args.max_iter, 200),
    solver="liblinear",
    random_state=42,
) 

    model.fit(
        x_train,
        y_train
    )

    accuracy = model.score(
        x_test,
        y_test
    )

    print("Accuracy:", accuracy)

    run.log("Accuracy", accuracy)

    # Save Model
    os.makedirs("outputs", exist_ok=True)

    joblib.dump(
        model,
        "outputs/model.pkl"
    )

    print("Model saved successfully.")


if __name__ == "__main__":
    main()