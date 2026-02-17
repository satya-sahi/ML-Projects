"""
Data preprocessing module for house price prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class HousePricePreprocessor:
    """Preprocess house price data for ML model"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.categorical_columns = ["location", "property_type"]
        self.numeric_columns = []

    def fit_transform(self, df):
        """
        Fit preprocessor on training data and transform

        Args:
            df: DataFrame with all features including target

        Returns:
            X: Preprocessed features
            y: Target variable (price)
        """
        # Make a copy to avoid modifying original
        df = df.copy()

        # Separate features and target
        y = df["price"].copy()
        X = df.drop(
            ["price", "property_id", "listing_date", "sold_date"],
            axis=1,
            errors="ignore",
        ).copy()

        # Handle boolean columns (convert to int)
        bool_columns = X.select_dtypes(include=["bool"]).columns
        for col in bool_columns:
            X[col] = X[col].astype(int)

        # Handle missing values in numeric columns
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in self.numeric_columns:
            if X[col].isnull().sum() > 0:
                median_value = X[col].median()
                X[col] = X[col].fillna(median_value)
                print(f"Filled missing values in {col} with median: {median_value}")

        # Encode categorical variables
        for col in self.categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {list(le.classes_)}")

        # Store feature columns
        self.feature_columns = X.columns.tolist()

        # Scale numeric features
        X[self.numeric_columns] = self.scaler.fit_transform(X[self.numeric_columns])

        print("\n✅ Preprocessing complete!")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        return X, y

    def transform(self, df):
        """
        Transform new data using fitted preprocessor

        Args:
            df: DataFrame with features (no target needed)

        Returns:
            X: Preprocessed features
        """
        df = df.copy()

        # Remove target and metadata if present
        X = df.drop(
            ["price", "property_id", "listing_date", "sold_date"],
            axis=1,
            errors="ignore",
        ).copy()

        # Handle boolean columns
        bool_columns = X.select_dtypes(include=["bool"]).columns
        for col in bool_columns:
            X[col] = X[col].astype(int)

        # Handle missing values
        for col in self.numeric_columns:
            if col in X.columns and X[col].isnull().sum() > 0:
                median_value = X[col].median()
                X[col] = X[col].fillna(median_value)

        # Encode categorical variables
        for col in self.categorical_columns:
            if col in X.columns:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))

        # Ensure same columns as training
        X = X[self.feature_columns]

        # Scale numeric features
        X[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])

        return X

    def save(self, path):
        """Save preprocessor to disk"""
        joblib.dump(self, path)
        print(f"💾 Preprocessor saved to {path}")

    @staticmethod
    def load(path):
        """Load preprocessor from disk"""
        return joblib.load(path)


if __name__ == "__main__":
    # Test preprocessing
    df = pd.read_excel("data/houses.xlsx")

    preprocessor = HousePricePreprocessor()
    X, y = preprocessor.fit_transform(df)

    print("\n📊 Preprocessed Data Sample:")
    print(X.head())
    print("\nTarget (price) sample:")
    print(y.head())
