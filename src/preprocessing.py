"""Předzpracování dat

Všechny kroky jsou sklearn transformery pracující s pandas DataFrame, takže je lze
libovolně skládat do Pipeline, ukládat přes joblib a zavolat v testing.py.

Zásady:
  * vše, co se "učí" (kvantily, mediány, škálování), se učí jen ve fit() na trénovacích datech,
  * transform() nikdy neodstraní řádek,
  * NaN na vstupu je očekávaný (i při predikci), na výstupu už žádný nezůstane.

Rychlé použití:
    from src.preprocessing import build_preprocessor, load_xy
    X, y = load_xy("data/train.csv")
    prep = build_preprocessor(imputer="median", scaler="standard")
    Xt = prep.fit_transform(X)

Typický import jednotlivých částí:
    from src.preprocessing import Winsorizer, Log1p, MissingIndicator, ColumnWise, FeatureEngineer
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (nutné pro IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

# --------------------------------------------------------------------------- #
# Definice sloupců
# --------------------------------------------------------------------------- #
TARGET = "target"
MA_COLS = [f"ma_detekce_{i:02d}" for i in range(1, 7)]
EX_COLS = [f"exsudaty_{i:02d}" for i in range(1, 9)]
GEOM_COLS = ["vzdalenost_makula_papila", "prumer_papily"]
BINARY_COLS = ["kvalita", "prescreening", "am_fm_klasifikace"]  # bez NaN
NUM_COLS = MA_COLS + EX_COLS + GEOM_COLS                          # zde jsou NaN
LOG_COLS = MA_COLS + EX_COLS                                      # silně zešikmené počty / podíly
FEATURE_COLS = NUM_COLS + BINARY_COLS


# --------------------------------------------------------------------------- #
# Načítání
# --------------------------------------------------------------------------- #
def load_xy(path) -> tuple[pd.DataFrame, np.ndarray]:
    """Načte CSV a vrátí (X, y). Sloupec target do X nepatří."""
    data = pd.read_csv(path)
    y = data[TARGET].to_numpy(dtype=int)
    X = data.drop(columns=[TARGET])
    return X, y


def missing_report(X: pd.DataFrame) -> pd.DataFrame:
    """Počet a podíl chybějících hodnot po sloupcích (jen sloupce s NaN)."""
    n = X.isna().sum()
    rep = pd.DataFrame({"n_nan": n, "podil": (n / len(X)).round(3)})
    return rep[rep["n_nan"] > 0].sort_values("n_nan", ascending=False)


# --------------------------------------------------------------------------- #
# Základ pro transformery nad DataFrame
# --------------------------------------------------------------------------- #
class _FrameTransformer(BaseEstimator, TransformerMixin):
    """Společné chování: pamatuje si sloupce z fit() a v transform() je zarovná."""

    def _fit_columns(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} očekává pandas DataFrame.")
        self.columns_in_ = list(X.columns)

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} očekává pandas DataFrame.")
        missing = [c for c in self.columns_in_ if c not in X.columns]
        if missing:
            raise KeyError(f"Chybí sloupce: {missing}")
        # zarovnání pořadí sloupců na to z fit(); počet řádků se nemění
        return X[self.columns_in_].astype(float).copy()


# --------------------------------------------------------------------------- #
# Jednotlivé kroky
# --------------------------------------------------------------------------- #
class MissingIndicator(_FrameTransformer):
    """Přidá informaci o chybění (musí být PŘED imputací).

    mode="count": jeden sloupec `n_chybi` (počet NaN v řádku)
    mode="each" : sloupec `chybi_<col>` pro každý sloupec z `cols`
    """

    def __init__(self, cols=tuple(NUM_COLS), mode: str = "count"):
        self.cols = cols
        self.mode = mode

    def fit(self, X, y=None):
        self._fit_columns(X)
        return self

    def transform(self, X):
        out = self._prepare(X)
        cols = list(self.cols)
        mask = X[cols].isna()
        if self.mode == "count":
            out["n_chybi"] = mask.sum(axis=1).astype(float).to_numpy()
        elif self.mode == "each":
            for c in cols:
                out[f"chybi_{c}"] = mask[c].astype(float).to_numpy()
        else:
            raise ValueError("mode musí být 'count' nebo 'each'.")
        return out


class Winsorizer(_FrameTransformer):
    """Ořízne odlehlé hodnoty na kvantily naučené z trénovacích dat (NaN zůstává)."""

    def __init__(self, cols=tuple(NUM_COLS), lower: float = 0.01, upper: float = 0.99):
        self.cols = cols
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self._fit_columns(X)
        cols = list(self.cols)
        self.low_ = X[cols].quantile(self.lower)
        self.high_ = X[cols].quantile(self.upper)
        return self

    def transform(self, X):
        out = self._prepare(X)
        cols = list(self.cols)
        out[cols] = out[cols].clip(lower=self.low_, upper=self.high_, axis=1)
        return out


class Log1p(_FrameTransformer):
    """log(1 + x) na zadaných sloupcích; záporné hodnoty se před tím ořežou na 0."""

    def __init__(self, cols=tuple(LOG_COLS)):
        self.cols = cols

    def fit(self, X, y=None):
        self._fit_columns(X)
        return self

    def transform(self, X):
        out = self._prepare(X)
        cols = list(self.cols)
        out[cols] = np.log1p(out[cols].clip(lower=0))
        return out


class ColumnWise(_FrameTransformer):
    """Aplikuje sklearn transformer (scaler, imputer...) jen na vybrané sloupce, ostatní nechá.

    Transformer musí zachovat počet sloupců (SimpleImputer/KNNImputer/IterativeImputer
    bez add_indicator, StandardScaler, RobustScaler, ...).
    """

    def __init__(self, transformer, cols):
        self.transformer = transformer
        self.cols = cols

    def fit(self, X, y=None):
        self._fit_columns(X)
        self.transformer_ = clone(self.transformer).fit(X[list(self.cols)], y)
        return self

    def transform(self, X):
        out = self._prepare(X)
        cols = list(self.cols)
        out[cols] = self.transformer_.transform(out[cols])
        return out


ENGINEERED_FEATURES = ("ma_mean", "ma_slope", "ma_diff", "ex_mean", "ex_slope", "ex_diff")
"""Skupiny odvozených příznaků, které lze zapnout ve FeatureEngineer / build_preprocessor."""


class FeatureEngineer(_FrameTransformer):
    """Odvozené příznaky (volat PO logaritmu, imputaci i škálování).

    Protože jsou ma_* a exsudaty_* tentýž signál při různých prazích, lze je shrnout.
    Parametr `features` určuje, které skupiny se přidají:

      ma_mean / ex_mean   - průměr přes všechny úrovně spolehlivosti (1 příznak)
      ma_slope / ex_slope - nejpřísnější - nejmírnější úroveň (1 příznak)
      ma_diff / ex_diff   - rozdíly sousedních úrovní, další - předchozí
                            (5 příznaků pro ma, 7 pro ex)

    Zkratky "mean", "slope", "diff" zapnou skupinu pro ma i ex najednou.
    Výchozí (None) = všechny skupiny z ENGINEERED_FEATURES.

    Příklady:
        FeatureEngineer()                                # všech 6 skupin (16 příznaků)
        FeatureEngineer(features=("mean", "slope"))      # jen průměry a sklony (4)
        FeatureEngineer(features=("ma_diff", "ex_mean")) # jen rozdíly ma a průměr ex
    """

    def __init__(self, ma_cols=tuple(MA_COLS), ex_cols=tuple(EX_COLS), features=None):
        self.ma_cols = ma_cols
        self.ex_cols = ex_cols
        self.features = features

    def _resolved_features(self) -> list[str]:
        """Rozbalí zkratky a ověří názvy; vrací seznam bez duplicit v pevném pořadí."""
        zadano = ENGINEERED_FEATURES if self.features is None else self.features
        if isinstance(zadano, str):
            zadano = (zadano,)
        vybrano = set()
        for f in zadano:
            if f in ("mean", "slope", "diff"):
                vybrano |= {f"ma_{f}", f"ex_{f}"}
            elif f in ENGINEERED_FEATURES:
                vybrano.add(f)
            else:
                raise ValueError(
                    f"Neznámý odvozený příznak: {f!r}. "
                    f"Povolené: {ENGINEERED_FEATURES} nebo zkratky 'mean', 'slope', 'diff'."
                )
        return [f for f in ENGINEERED_FEATURES if f in vybrano]

    def fit(self, X, y=None):
        self._fit_columns(X)
        self.features_ = self._resolved_features()
        return self

    def transform(self, X):
        out = self._prepare(X)
        for prefix, cols in (("ma", list(self.ma_cols)), ("ex", list(self.ex_cols))):
            if f"{prefix}_mean" in self.features_:
                out[f"{prefix}_mean"] = out[cols].mean(axis=1)
            if f"{prefix}_slope" in self.features_:
                out[f"{prefix}_slope"] = out[cols[-1]] - out[cols[0]]
            if f"{prefix}_diff" in self.features_:
                for a, b in zip(cols[:-1], cols[1:]):
                    # název např. ma_diff_01_02 (číslo úrovně z konce názvu sloupce)
                    out[f"{prefix}_diff_{a[-2:]}_{b[-2:]}"] = out[b] - out[a]
        return out


class DropColumns(_FrameTransformer):
    """Odstraní sloupce (např. nejvíc korelované kopie). Řádky nechává."""

    def __init__(self, cols=()):
        self.cols = cols

    def fit(self, X, y=None):
        self._fit_columns(X)
        return self

    def transform(self, X):
        return self._prepare(X).drop(columns=list(self.cols))


# --------------------------------------------------------------------------- #
# Továrny na imputer / scaler a celá pipeline
# --------------------------------------------------------------------------- #
def make_imputer(kind: str = "median", n_neighbors: int = 7, random_state: int = 0):
    """Imputer pro číselné sloupce. kind: median | mean | knn | iterative."""
    if kind in ("median", "mean"):
        return SimpleImputer(strategy=kind)
    if kind == "knn":
        return KNNImputer(n_neighbors=n_neighbors, weights="distance")
    if kind == "iterative":
        return IterativeImputer(max_iter=20, sample_posterior=False, random_state=random_state)
    raise ValueError(f"Neznámý imputer: {kind}")


def make_scaler(kind: str | None = "standard"):
    """Scaler pro číselné sloupce. kind: standard | robust | None."""
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    if kind is None:
        return None
    raise ValueError(f"Neznámý scaler: {kind}")


def build_preprocessor(
    imputer: str = "median",
    scaler: str | None = "standard",
    winsor: tuple[float, float] | None = (0.01, 0.99),
    log: bool = True,
    missing: str | None = "count",
    engineer: bool | str | tuple = True,
    n_neighbors: int = 14,
) -> Pipeline:
    """Složí celé předzpracování do jedné Pipeline.

    Pořadí kroků:
      1. indikátor chybění  - musí vidět NaN před imputací
      2. winsorizace        - ořez odlehlých hodnot, NaN zůstává
      3. log1p              - zešikmené počty a podíly
      4. škálování          - standardní/robustní, NaN ignoruje (důležité pro KNN imputaci)
      5. imputace           - median / knn / iterative; použije i binární příznaky jako oporu
      6. odvozené příznaky  - průměry a sklony přes úrovně detektoru
    """
    steps: list[tuple[str, object]] = []
    if missing:
        steps.append(("chybeni", MissingIndicator(mode=missing)))
    if winsor:
        steps.append(("winsor", Winsorizer(lower=winsor[0], upper=winsor[1])))
    if log:
        steps.append(("log", Log1p()))
    sc = make_scaler(scaler)
    if sc is not None:
        steps.append(("scaler", ColumnWise(sc, NUM_COLS)))
    steps.append(("imputer", ColumnWise(make_imputer(imputer, n_neighbors), FEATURE_COLS)))
    if engineer:  # True = všechny skupiny; tuple/str = vybrané; False/None/() = žádné
        feats = None if engineer is True else engineer
        steps.append(("feat", FeatureEngineer(features=feats)))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# Samotestování: python -m src.preprocessing cesta/k/train.csv
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    from io import BytesIO
    from pathlib import Path
    import joblib

    SLOZKA = Path(__file__).resolve().parent.parent
    path = sys.argv[1] if len(sys.argv) > 1 else SLOZKA / "data" / "train.csv"

    X, y = load_xy(path)
    print("Chybějící hodnoty:\n", missing_report(X), "\n")

    for imp in ("median", "knn", "iterative"):
        prep = build_preprocessor(imputer=imp)
        Xt = prep.fit_transform(X)
        assert len(Xt) == len(X), "transform nesmí měnit počet řádků"
        assert not Xt.isna().any().any(), "po předzpracování nesmí zůstat NaN"

        buf = BytesIO()                     # uložení a načtení přes joblib musí dát stejný výstup
        joblib.dump(prep, buf)
        buf.seek(0)
        pd.testing.assert_frame_equal(joblib.load(buf).transform(X), Xt)

        # robustnost: jiné pořadí sloupců a řádek s mnoha NaN
        X2 = X[X.columns[::-1]].copy()
        X2.iloc[0, :] = np.where(X2.columns.isin(NUM_COLS), np.nan, X2.iloc[0, :])
        assert not prep.transform(X2).isna().any().any()
        print(f"[OK] imputer={imp:<9} výstup: {Xt.shape}")
    print("\nSloupce:", list(Xt.columns))
