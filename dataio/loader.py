"""Načítání trénovacích dat.

Funkce jen čte CSV a odděluje cílovou proměnnou. Nic nemaže ani neupravuje
(chybějící hodnoty a odlehlé hodnoty se ošetřují až v sklearn.Pipeline).
"""
from pathlib import Path

import pandas as pd

TARGET = "target"


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"


def nacti_xy(cesta: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Načte CSV a rozdělí ho na příznaky a cílovou proměnnou.

    Args:
        cesta: Cesta k CSV souboru. Výchozí je DATA_PATH (data/train.csv).

    Returns:
        Dvojice (X, y):
            X: DataFrame všech příznaků (bez `target`)
            y: vektor s hodnotami 0/1 (typ int), stejný index jako X.

    Raises:
        AssertionError: Pokud v souboru chybí sloupec `target`.
        FileNotFoundError: Pokud soubor neexistuje.
    """
    df = pd.read_csv(cesta)
    assert TARGET in df.columns, f"Chybí sloupec '{TARGET}'"
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return X, y
