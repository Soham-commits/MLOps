#!/usr/bin/env python3
"""Train a simple classifier and save it with pickle.
This is Expt 1: Train ML model and deploy model to file (PKL FILE)
"""
from pathlib import Path
import pickle

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_save(model_path: Path) -> RandomForestClassifier:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=0)
    clf.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    return clf


def verify_load(model_path: Path, sample):
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)
    return loaded.predict([sample])


def main():
    model_path = Path(__file__).parent / "model.pkl"
    clf = train_and_save(model_path)
    sample = [5.1, 3.5, 1.4, 0.2]
    print("Saved model to:", model_path)
    print("Sample prediction (verify load):", verify_load(model_path, sample))


if __name__ == "__main__":
    main()
