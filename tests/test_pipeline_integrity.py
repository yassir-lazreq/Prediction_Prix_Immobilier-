from pathlib import Path
import unittest

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class PipelineIntegrityTests(unittest.TestCase):
    def test_clean_data_and_model_artifacts_are_consistent(self):
        raw = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
        clean = pd.read_csv(
            ROOT / "data" / "processed" / "train_clean.csv",
            keep_default_na=False,
        )
        x_train = joblib.load(ROOT / "data" / "processed" / "X_train.pkl")
        x_val = joblib.load(ROOT / "data" / "processed" / "X_val.pkl")
        x_test = joblib.load(ROOT / "data" / "processed" / "X_test.pkl")
        y_train = joblib.load(ROOT / "data" / "processed" / "y_train.pkl")
        y_val = joblib.load(ROOT / "data" / "processed" / "y_val.pkl")
        y_test = joblib.load(ROOT / "data" / "processed" / "y_test.pkl")
        scaler = joblib.load(ROOT / "models" / "scaler.pkl")

        self.assertEqual(raw.shape, (1460, 81))
        self.assertEqual((raw["Id"].min(), raw["Id"].max()), (1, 1460))
        self.assertEqual(clean.shape, (1458, 81))
        self.assertEqual(int(clean.isna().sum().sum()), 0)
        self.assertEqual(x_train.shape, (1020, 209))
        self.assertEqual(x_val.shape, (219, 209))
        self.assertEqual(x_test.shape, (219, 209))
        self.assertEqual(y_train.shape, (1020,))
        self.assertEqual(y_val.shape, (219,))
        self.assertEqual(y_test.shape, (219,))
        self.assertEqual(scaler.n_features_in_, 209)


if __name__ == "__main__":
    unittest.main()
