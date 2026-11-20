"""Hledání nejlepšího nastavení předzpracování (build_preprocessor) pro jednotlivé modely.

Prohledává parametry funkce `src.preprocessing.build_preprocessor`
(imputer, n_neighbors, scaler, winsor, log, missing, engineer) pro modely
logistická regrese, LDA, random forest a KNN. Metrika je MCC.

Postup (dvě fáze, aby to doběhlo v rozumném čase):
  1. Hrubé hledání: náhodný vzorek konfigurací z mřížky (nebo celá mřížka
     s --vse), levnější CV 5 foldů x 2 opakování.
  2. Zpřesnění: pro každý model se vezme TOP-K konfigurací z fáze 1 a ty se
     znovu ohodnotí plnou CV 5 x 5 s JINÝM seedem. Jiné foldy omezují
     "prokletí vítěze" (konfigurace, která ve fázi 1 vyhrála jen díky šumu).

Úspora času: předzpracování se v každém foldu naučí jen jednou a výsledek
se použije pro všechny modely (předzpracování nezávisí na modelu). Vše se
učí jen na trénovací části foldu, takže nevzniká data leakage.

Výstupy:
  logs/ladeni_preprocessingu_<cas>.csv   všechny výsledky obou fází
  config/nejlepsi_preprocessing.json     nejlepší nastavení pro každý model

Spuštění (z kořene projektu):
    python -m src.ladeni_preprocessingu                   # 400 náhodných konfigurací
    python -m src.ladeni_preprocessingu --n-konfiguraci 150 --top 8
    python -m src.ladeni_preprocessingu --vse             # celá mřížka (dlouhé!)
    python -m src.ladeni_preprocessingu --hyper           # i pár variant hyperparametrů modelů
    python -m src.ladeni_preprocessingu --bez-rf          # bez random forestu (výrazně rychlejší)

Poznámka: MCC z fáze 2 je pořád lehce optimistické (vybíralo se z mnoha
konfigurací). Nezkreslený odhad finálního modelu dává `vnorena_cv`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier

SLOZKA = Path(__file__).resolve().parent.parent
if str(SLOZKA) not in sys.path:  # umožní i spuštění `python src/ladeni_preprocessingu.py`
    sys.path.insert(0, str(SLOZKA))

from dataio.loader import nacti_xy  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402
from src.validation import SEED, vytvor_cv  # noqa: E402

# --------------------------------------------------------------------------- #
# Prohledávaný prostor předzpracování
# --------------------------------------------------------------------------- #
# (imputer, n_neighbors) - n_neighbors má smysl jen pro knn, jinak se nepoužije
IMPUTERY = [
    ("median", None),
    ("mean", None),
    ("knn", 5),
    ("knn", 14),
    ("knn", 25),
    ("iterative", None),
]
SCALERY = ["standard", "robust", None]
WINSOR = [None, (0.005, 0.995), (0.01, 0.99), (0.025, 0.975), (0.05, 0.95)]
LOG = [True, False]
MISSING = [None, "count", "each"]
ENGINEER = [False, ("mean",), ("mean", "slope"), ("diff",), True]

# Výchozí nastavení z validation.py - vždy se zařadí do vyhodnocení jako reference.
REFERENCE = dict(
    imputer="iterative", n_neighbors=None, scaler="standard", winsor=(0.01, 0.99),
    log=True, missing="count", engineer=("diff",),
)


def vsechny_konfigurace() -> list[dict]:
    """Celá mřížka konfigurací (kartézský součin všech voleb)."""
    konfigurace = []
    for (imp, k), sc, w, lg, ms, eng in itertools.product(
        IMPUTERY, SCALERY, WINSOR, LOG, MISSING, ENGINEER
    ):
        konfigurace.append(dict(
            imputer=imp, n_neighbors=k, scaler=sc, winsor=w,
            log=lg, missing=ms, engineer=eng,
        ))
    return konfigurace


def kwargs_preprocessoru(konf: dict) -> dict:
    """Převede konfiguraci na argumenty build_preprocessor."""
    kw = dict(konf)
    if kw.pop("n_neighbors") is not None:
        kw["n_neighbors"] = konf["n_neighbors"]
    return kw


def popis(konf: dict) -> str:
    """Krátký čitelný popis konfigurace (slouží zároveň jako klíč)."""
    imp = konf["imputer"] + (f"{konf['n_neighbors']}" if konf["n_neighbors"] else "")
    w = "ne" if konf["winsor"] is None else f"{konf['winsor'][0]}-{konf['winsor'][1]}"
    eng = konf["engineer"]
    eng = "vse" if eng is True else ("ne" if not eng else "+".join(eng))
    return (f"imp={imp} | sc={konf['scaler']} | win={w} | log={int(konf['log'])} "
            f"| miss={konf['missing']} | eng={eng}")


# --------------------------------------------------------------------------- #
# Modely
# --------------------------------------------------------------------------- #
def vytvor_modely(hyper: bool, bez_rf: bool) -> dict[str, dict[str, object]]:
    """{model: {varianta: estimator}}. Bez --hyper má každý model jednu variantu
    se stejným nastavením jako ve validation.py."""
    if not hyper:
        modely = {
            "logreg": {"C=1": LogisticRegression(max_iter=2000)},
            "lda": {"lsqr_auto": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")},
            "rf": {"vychozi": RandomForestClassifier(n_estimators=200, random_state=SEED)},
            "knn": {"k=15_dist": KNeighborsClassifier(n_neighbors=15, weights="distance")},
        }
    else:
        modely = {
            "logreg": {
                f"C={c}": LogisticRegression(C=c, max_iter=2000) for c in (0.1, 1.0, 10.0)
            },
            "lda": {
                "lsqr_auto": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                "svd": LinearDiscriminantAnalysis(solver="svd"),
            },
            "rf": {
                f"leaf={leaf}_mf={mf}": RandomForestClassifier(
                    n_estimators=200, min_samples_leaf=leaf, max_features=mf,
                    random_state=SEED,
                )
                for leaf in (1, 5) for mf in ("sqrt", 0.5)
            },
            "knn": {
                f"k={k}_{w}": KNeighborsClassifier(n_neighbors=k, weights=w)
                for k in (9, 15, 25, 41) for w in ("uniform", "distance")
            },
        }
    if bez_rf:
        modely.pop("rf")
    return modely


# --------------------------------------------------------------------------- #
# Vyhodnocení jedné konfigurace předzpracování na všech foldech a modelech
# --------------------------------------------------------------------------- #
def vyhodnot_konfiguraci(konf, X, y, splity, modely) -> dict[tuple[str, str], np.ndarray]:
    """Vrátí {(model, varianta): pole MCC po foldech}. Předzpracování se učí
    v každém foldu jen jednou a sdílí se mezi všemi modely."""
    warnings.simplefilter("ignore")  # ConvergenceWarning apod. v paralelních workerech
    skore = {(m, v): [] for m, var in modely.items() for v in var}
    for tr, te in splity:
        prep = build_preprocessor(**kwargs_preprocessoru(konf))
        X_tr = prep.fit_transform(X.iloc[tr])
        X_te = prep.transform(X.iloc[te])
        for m, varianty in modely.items():
            for v, est in varianty.items():
                pred = clone(est).fit(X_tr, y[tr]).predict(X_te)
                skore[(m, v)].append(matthews_corrcoef(y[te], pred))
    return {k: np.asarray(s) for k, s in skore.items()}


def spust_fazi(nazev, konfigurace, X, y, cv, modely, n_jobs) -> pd.DataFrame:
    """Ohodnotí seznam konfigurací (paralelně přes konfigurace) a vrátí tabulku."""
    splity = list(cv.split(X, y))  # stejné foldy pro všechny konfigurace i modely
    print(f"\n=== {nazev}: {len(konfigurace)} konfigurací x {len(splity)} foldů ===")
    start = time.perf_counter()
    vysledky = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(vyhodnot_konfiguraci)(konf, X, y, splity, modely) for konf in konfigurace
    )
    print(f"{nazev} trvala {time.perf_counter() - start:.1f} s")

    radky = []
    for konf, skore in zip(konfigurace, vysledky):
        for (m, v), s in skore.items():
            radky.append({
                "faze": nazev, "model": m, "varianta": v, "konfigurace": popis(konf),
                **{k: (str(val) if isinstance(val, tuple) else val) for k, val in konf.items()},
                "mcc_mean": s.mean(), "mcc_std": s.std(),
                "mcc_se": s.std() / np.sqrt(len(s)),
            })
    return pd.DataFrame(radky)


# --------------------------------------------------------------------------- #
# Hlavní běh
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-konfiguraci", type=int, default=400,
                    help="počet náhodně vybraných konfigurací ve fázi 1 (výchozí 400)")
    ap.add_argument("--vse", action="store_true",
                    help="ve fázi 1 projít celou mřížku (%d konfigurací)" % len(vsechny_konfigurace()))
    ap.add_argument("--top", type=int, default=10,
                    help="kolik nejlepších konfigurací na model postoupí do fáze 2 (výchozí 10)")
    ap.add_argument("--hyper", action="store_true",
                    help="zkoušet i několik variant hyperparametrů každého modelu")
    ap.add_argument("--bez-rf", action="store_true", help="vynechat random forest")
    ap.add_argument("--n-jobs", type=int, default=-1, help="počet paralelních procesů")
    args = ap.parse_args()

    X, y = nacti_xy()
    y = y.to_numpy()
    modely = vytvor_modely(args.hyper, args.bez_rf)

    # ---- Fáze 1: hrubé hledání ------------------------------------------- #
    vsechny = vsechny_konfigurace()
    if args.vse or args.n_konfiguraci >= len(vsechny):
        faze1_konf = vsechny
    else:
        faze1_konf = random.Random(SEED).sample(vsechny, args.n_konfiguraci)
        if REFERENCE not in faze1_konf:
            faze1_konf.append(REFERENCE)
    print(f"Mřížka má {len(vsechny)} konfigurací, ve fázi 1 se zkouší {len(faze1_konf)}.")
    print("Modely a varianty:", {m: list(v) for m, v in modely.items()})

    cv1 = vytvor_cv(n_splits=5, n_repeats=2, random_state=SEED)
    tab1 = spust_fazi("faze1", faze1_konf, X, y, cv1, modely, args.n_jobs)

    # ---- Výběr TOP-K konfigurací pro každý model -------------------------- #
    # pro každý model a konfiguraci se bere nejlepší varianta modelu
    nejlepsi_var = tab1.groupby(["model", "konfigurace"])["mcc_mean"].max().reset_index()
    postupujici = set()
    for m, skup in nejlepsi_var.groupby("model"):
        postupujici |= set(skup.nlargest(args.top, "mcc_mean")["konfigurace"])
    postupujici.add(popis(REFERENCE))
    podle_popisu = {popis(k): k for k in faze1_konf}
    faze2_konf = [podle_popisu[p] for p in sorted(postupujici)]

    # ---- Fáze 2: zpřesnění na plné CV s jiným seedem ----------------------- #
    cv2 = vytvor_cv(n_splits=5, n_repeats=5, random_state=SEED + 1)
    tab2 = spust_fazi("faze2", faze2_konf, X, y, cv2, modely, args.n_jobs)

    # ---- Uložení všech výsledků ------------------------------------------- #
    cas = datetime.now().strftime("%Y%m%d_%H%M%S")
    (SLOZKA / "logs").mkdir(exist_ok=True)
    csv_cesta = SLOZKA / "logs" / f"ladeni_preprocessingu_{cas}.csv"
    pd.concat([tab1, tab2]).to_csv(csv_cesta, index=False)

    # ---- Souhrn pro každý model ------------------------------------------- #
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 120)
    nejlepsi = {}
    ref_popis = popis(REFERENCE)
    for m in modely:
        t = tab2[tab2["model"] == m].sort_values("mcc_mean", ascending=False)
        best = t.iloc[0]
        # konfigurace statisticky nerozlišitelné od nejlepší (do 1 SE)
        t = t.assign(do_1se=t["mcc_mean"] >= best["mcc_mean"] - best["mcc_se"])
        ref = t[t["konfigurace"] == ref_popis]["mcc_mean"].max()

        print(f"\n{'=' * 100}\nMODEL: {m}   (fáze 2, CV 5x5)")
        print(t[["varianta", "konfigurace", "mcc_mean", "mcc_se", "do_1se"]]
              .head(10).round(4).to_string(index=False))
        print(f"Reference z validation.py: MCC = {ref:.4f}")

        konf = podle_popisu[best["konfigurace"]]
        kw = kwargs_preprocessoru(konf)
        print("Nejlepší:  build_preprocessor(" + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + ")"
              + f"   + {m} [{best['varianta']}]")
        nejlepsi[m] = {
            "build_preprocessor": kw,
            "model_varianta": best["varianta"],
            "mcc_mean": round(float(best["mcc_mean"]), 4),
            "mcc_se": round(float(best["mcc_se"]), 4),
            "mcc_reference": round(float(ref), 4),
        }

    (SLOZKA / "config").mkdir(exist_ok=True)
    json_cesta = SLOZKA / "config" / "nejlepsi_preprocessing.json"
    # pozn.: n-tice (winsor, engineer) se v JSON uloží jako seznamy; build_preprocessor
    # je přijme i tak, případně je po načtení převeďte zpět na tuple.
    json_cesta.write_text(json.dumps(nejlepsi, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nVšechny výsledky: {csv_cesta}")
    print(f"Nejlepší nastavení: {json_cesta}")


if __name__ == "__main__":
    main()
