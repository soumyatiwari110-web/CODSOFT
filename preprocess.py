# ============================================================
# TITANIC SURVIVAL PREDICTION - PREPROCESSING
# Author: Soumya Tiwari
# Description: Clean and prepare data for model training
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath='dataset/titanic.csv'):
    """
    Complete preprocessing pipeline.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """

    # ----------------------------
    # STEP 1: LOAD DATA
    # ----------------------------
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ----------------------------
    # STEP 2: DROP USELESS COLUMNS
    # Why: PassengerId, Name, Ticket, Cabin don't help predictions.
    # Cabin has 77% missing values - too unreliable.
    # ----------------------------
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    print("✅ Dropped: PassengerId, Name, Ticket, Cabin")

    # ----------------------------
    # STEP 3: HANDLE MISSING VALUES
    # Age  -> fill with MEDIAN (robust to outliers)
    # Embarked -> fill with MODE (most common port)
    # Fare -> fill with MEDIAN
    # ----------------------------
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    print(f"✅ Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")

    # ----------------------------
    # STEP 4: FEATURE ENGINEERING
    # Create new meaningful features from existing ones
    # ----------------------------

    # Family Size: total family aboard
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # IsAlone: if travelling solo
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Age Group: bucket ages into categories
    df['AgeGroup'] = pd.cut(df['Age'],
                             bins=[0, 12, 18, 35, 60, 100],
                             labels=[0, 1, 2, 3, 4])
    df['AgeGroup'] = df['AgeGroup'].astype(int)

    # Fare Category: bucket fares
    df['FareCategory'] = pd.qcut(df['Fare'], q=4,
                                  labels=[0, 1, 2, 3])
    df['FareCategory'] = df['FareCategory'].astype(int)

    print("✅ Feature engineering done: FamilySize, IsAlone, AgeGroup, FareCategory")

    # ----------------------------
    # STEP 5: ENCODE CATEGORICAL COLUMNS
    # Machine learning needs NUMBERS, not text.
    # Sex: male=1, female=0
    # Embarked: C=0, Q=1, S=2
    # ----------------------------
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])         # male=1, female=0
    df['Embarked'] = le.fit_transform(df['Embarked'])  # C=0, Q=1, S=2

    print("✅ Encoding done: Sex, Embarked")

    # ----------------------------
    # STEP 6: SELECT FEATURES
    # ----------------------------
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                    'Embarked', 'FamilySize', 'IsAlone', 'AgeGroup', 'FareCategory']

    X = df[feature_cols]
    y = df['Survived']

    print(f"✅ Features selected: {feature_cols}")
    print(f"   X shape: {X.shape}, y shape: {y.shape}")

    # ----------------------------
    # STEP 7: TRAIN-TEST SPLIT
    # 80% for training, 20% for testing
    # random_state=42 ensures reproducibility
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n✅ Train-Test Split Done:")
    print(f"   Training samples : {X_train.shape[0]}")
    print(f"   Testing samples  : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, feature_cols


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, features = load_and_preprocess()
    print("\n🎯 Preprocessing complete and ready for model training!")