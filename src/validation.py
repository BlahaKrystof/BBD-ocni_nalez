"""Validace modelů: opakovaná stratifikovaná k-fold cross validace s metrikou MCC.

Všechny kroky předzpracování a všechny modely se porovnávají přes tyto funkce
na STEJNÝCH foldech (pevný random_state), takže jsou výsledky srovnatelné.

Pravidla použití
    - Celé předzpracování (imputace, ořezání, škálování) musí být uvnitř
      sklearn.Pipeline, aby se učilo jen na trénovací části každého foldu
      (jinak vzniká data leakage a MCC vyjde příliš optimistické).
    - Pro odhad kvality po ladění hyperparametrů použijte `vnorena_cv`,
      ne `vyhodnot` na stejných datech.

Hlavní funkce
    vytvor_cv     připraví objekt RepeatedStratifiedKFold,
    vyhodnot      MCC jednoho modelu (průměr, směrodatná odchylka, všechny foldy),
    porovnej      tabulka MCC pro více modelů na stejných foldech,
    vnorena_cv    nezkreslený odhad MCC modelu s laděním hyperparametrů,
    vyber_prah    práh rozhodnutí maximalizující MCC (z out-of-fold pravděpodobností).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

SEED = 42
"""Výchozí náhodný seed"""

SCORING = "matthews_corrcoef"
"""Definování použité metriky (MCC)"""

_MCC_SCORER = make_scorer(matthews_corrcoef)


def vytvor_cv(
    n_splits: int = 5, n_repeats: int = 5, random_state: int = SEED
) -> RepeatedStratifiedKFold:
    """Vytvoří opakovanou stratifikovanou k-fold validaci.

    Args:
        n_splits: Počet foldů v jednom opakování (k). Výchozí 5.
        n_repeats: Počet opakování s jiným náhodným rozdělením. Výchozí 5,
            celkem tedy n_splits * n_repeats hodnot MCC.
        random_state: seed generátoru; ovlivňuje rozdělení do foldů.

    Returns:
        Objekt RepeatedStratifiedKFold; v každém foldu je zachován poměr tříd.
    """
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )


def vyhodnot(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: RepeatedStratifiedKFold | None = None,
) -> dict:
    """Spočítá MCC modelu křížovou validací.

    Args:
        model: Sklearn estimator nebo Pipeline.
        X: Příznaky (DataFrame, NaN povoleny, pokud je zvládne pipeline).
        y: Cílová proměnná 0/1.
        cv: Validační schéma; výchozí `vytvor_cv()`.

    Returns:
        Slovník s klíči:
            "mean":   průměrné MCC přes všechny foldy,
            "std":    směrodatná odchylka MCC přes všechny foldy,
            "scores": numpy pole MCC jednotlivých foldů (pořadí: opakování po opakování).
    """
    cv = cv or vytvor_cv()
    skore = cross_val_score(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {"mean": float(skore.mean()), "std": float(skore.std()), "scores": skore}


def porovnej(
    modely: dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    cv: RepeatedStratifiedKFold | None = None,
) -> pd.DataFrame:
    """Porovná více modelů nebo variant předzpracování na stejných foldech.

    Args:
        modely: Slovník {název: pipeline}.
        X: Příznaky.
        y: Cílová proměnná 0/1.
        cv: Validační schéma; výchozí `vytvor_cv()`. Foldy jsou pro všechny
            modely shodné, takže rozdíly nejsou způsobeny jiným dělením.

    Returns:
        DataFrame seřazený sestupně podle průměrného MCC se sloupci
        "mcc_mean", "mcc_std" a "mcc_se" (standardní chyba průměru; rozdíl
        mezi modely menší než zhruba 2 * mcc_se je v rámci šumu).
    """
    cv = cv or vytvor_cv()
    radky = {}
    for nazev, model in modely.items():
        v = vyhodnot(model, X, y, cv)
        radky[nazev] = {
            "mcc_mean": v["mean"],
            "mcc_std": v["std"],
            "mcc_se": v["std"] / np.sqrt(len(v["scores"])),
        }
    return pd.DataFrame(radky).T.sort_values("mcc_mean", ascending=False)


def vnorena_cv(
    model: BaseEstimator,
    param_distributions: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 20,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = SEED,
) -> dict:
    """Nezkreslený odhad MCC modelu včetně ladění hyperparametrů (vnořená CV).

    Vnější smyčka měří, vnitřní (RandomizedSearchCV) ladí hyperparametry jen
    na trénovací části vnějšího foldu. Odhad tak není optimistický, na rozdíl
    od ladění a měření na stejných foldech.

    Args:
        model: Sklearn Pipeline.
        param_distributions: Prostor hyperparametrů pro RandomizedSearchCV
            (klíče ve tvaru "krok__parametr").
        X: Příznaky.
        y: Cílová proměnná 0/1.
        n_iter: Počet vyzkoušených kombinací hyperparametrů ve vnitřní smyčce.
        outer_splits: Počet foldů vnější smyčky.
        inner_splits: Počet foldů vnitřní smyčky.
        random_state: seed pro obě smyčky.

    Returns:
        Slovník s klíči "mean", "std" a "scores" (MCC vnějších foldů),
        stejně jako u `vyhodnot`.
    """
    vnitrni = StratifiedKFold(inner_splits, shuffle=True, random_state=random_state)
    vnejsi = StratifiedKFold(outer_splits, shuffle=True, random_state=random_state + 1)
    hledani = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=n_iter,
        scoring=_MCC_SCORER,
        cv=vnitrni,
        random_state=random_state,
        n_jobs=-1,
    )
    skore = cross_val_score(hledani, X, y, cv=vnejsi, scoring=SCORING)
    return {"mean": float(skore.mean()), "std": float(skore.std()), "scores": skore}


def vyber_prah(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    prahy: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = SEED,
) -> tuple[float, float]:
    """Najde práh pravděpodobnosti třídy 1, který maximalizuje MCC.

    Pravděpodobnosti se získají out-of-fold (každý vzorek je predikován
    modelem, který ho při tréninku neviděl), takže výběr prahu není zkreslený
    přeučením. Model musí mít metodu `predict_proba`.

    Args:
        model: Pipeline s `predict_proba`.
        X: Příznaky.
        y: Cílová proměnná 0/1.
        prahy: Zkoušené prahy; výchozí 0.30 až 0.70 s krokem 0.01.
        n_splits: Počet foldů pro out-of-fold predikce.
        random_state: seed dělení.

    Returns:
        Dvojice (nejlepší práh, MCC při tomto prahu). Pozor: MCC je zde
        vybráno na stejných datech jako práh, je tedy lehce optimistické.
    """
    prahy = np.arange(0.30, 0.701, 0.01) if prahy is None else prahy
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    p = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1]
    mcc_hodnoty = [matthews_corrcoef(y, (p >= t).astype(int)) for t in prahy]
    i = int(np.argmax(mcc_hodnoty))
    return float(prahy[i]), float(mcc_hodnoty[i])


if __name__ == "__main__":
    # Ukázka: baseline modely (medián + škálování) na stejných foldech.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    from dataio.loader import nacti_xy

    X, y = nacti_xy()
    modely = {
        "logreg_median": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "rf_median": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=200, random_state=SEED),
        ),
        "knn_median": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=15, weights="distance"),
        ),
    }
    print(porovnej(modely, X, y).round(4).to_string())
