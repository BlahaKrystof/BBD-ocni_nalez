"""Systematické hľadanie najlepšieho predspracovania a hyperparametrov pre každý model.

Spúšťa sa z KORENE PROJEKTU (vedľa priečinkov src/ a dataio/):

    python search_all.py --estimate          # odhad času a počtu fitov, nič nepočíta
    python search_all.py --quick              # rýchly test (malá mriežka, pár minút)
    python search_all.py                      # plné hľadanie (môže trvať hodiny)
    python search_all.py --models lda logreg mlp --jobs 6

Čo skript robí (3 fázy, každá ukladá priebežné výsledky a dá sa po prerušení spustiť znova):

  FÁZA 1  Preosiatie predspracovania. Pre každý model sa vyskúšajú kombinácie
          imputer x scaler x winsor x log x missing x engineer s PREDVOLENÝMI
          hyperparametrami modelu, na 5-fold CV. Rýchle modely (logreg, lda, knn)
          prejdú VŠETKY kombinácie, pomalé (rf, et, mlp, hgb) náhodný výber
          --slow-configs kombinácií (plus referenčná konfigurácia).
          Dôležitá optimalizácia: imputer sa počíta raz na (základná konfigurácia,
          fold) a varianty missing/engineer sa aplikujú nad jeho výsledkom
          (je to matematicky to isté ako celý Pipeline, len rýchlejšie).

  FÁZA 2  Potvrdenie. Najlepších --top-k konfigurácií každého modelu sa znova
          vyhodnotí skutočným Pipeline na 5xN CV s INÝM seedom. Tým sa odstráni
          optimizmus z toho, že sme víťaza vyberali z veľkého množstva kombinácií.

  FÁZA 3  Hyperparametre. Pre --tune-top najlepších konfigurácií každého modelu
          sa spustí RandomizedSearchCV vo vnorenej CV (nested): odhad je nezaujatý
          a porovná sa s tým istým modelom bez ladenia na tých istých foldoch.
          Predspracovanie sa pri ladení kešuje, takže sa imputer nepočíta zbytočne.

Výstupy v priečinku search_out/ (zmeň cez --out):
    stage1.csv, stage2.csv, stage3.csv   tabuľky výsledkov (zoradené)
    report.txt                           súhrn: najlepšie nastavenie pre každý model
    best_<model>.joblib                  natrénovaný Pipeline (predspracovanie + model)
    best_overall.joblib                  najlepší model celkovo
    log.txt                              priebeh

Uložený Pipeline berie na vstupe DataFrame so stĺpcami ako v train.csv BEZ `target`
(v testing.py: predzpracuj() vráti vstup bezo zmeny a predikuj() zavolá model.predict(x)).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")  # aj pre pracovné procesy joblib
os.environ.setdefault("OMP_NUM_THREADS", "1")      # bez preťaženia CPU pri paralelných behoch
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from scipy.stats import loguniform, randint
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from dataio.loader import nacti_xy
from src.preprocessing import (
    FEATURE_COLS,
    NUM_COLS,
    ColumnWise,
    FeatureEngineer,
    Log1p,
    MissingIndicator,
    Winsorizer,
    build_preprocessor,
    make_imputer,
    make_scaler,
)

SLOZKA = Path(__file__).resolve().parent
MCC = make_scorer(matthews_corrcoef)

# --------------------------------------------------------------------------- #
# Modely: predvolené hyperparametre (fáza 1, 2) a prehľadávaný priestor (fáza 3)
# --------------------------------------------------------------------------- #
MODELS: dict[str, dict] = {
    "logreg": dict(
        slow=False,
        make=lambda s: LogisticRegression(solver="liblinear", max_iter=3000),
        space={"model__C": loguniform(1e-3, 1e2), "model__penalty": ["l1", "l2"]},
    ),
    "lda": dict(
        slow=False,
        make=lambda s: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        space={"model__shrinkage": ["auto", 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]},
    ),
    "knn": dict(
        slow=False,
        make=lambda s: KNeighborsClassifier(n_neighbors=15, weights="distance"),
        space={
            "model__n_neighbors": randint(3, 61),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
    ),
    "rf": dict(
        slow=True,
        make=lambda s: RandomForestClassifier(n_estimators=150, random_state=s, n_jobs=1),
        space={
            "model__n_estimators": [150, 300, 500],
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_leaf": randint(1, 12),
            "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        },
    ),
    "et": dict(
        slow=True,
        make=lambda s: ExtraTreesClassifier(n_estimators=200, random_state=s, n_jobs=1),
        space={
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_leaf": randint(1, 12),
            "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        },
    ),
    "mlp": dict(
        slow=True,
        make=lambda s: MLPClassifier(
            hidden_layer_sizes=(16,), alpha=0.1, max_iter=2000, random_state=s
        ),
        space={
            "model__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
            "model__alpha": loguniform(1e-3, 1.0),
            "model__learning_rate_init": loguniform(1e-3, 1e-2),
            "model__activation": ["relu", "tanh"],
        },
    ),
    "hgb": dict(
        slow=True,
        make=lambda s: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, random_state=s
        ),
        space={
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_iter": randint(50, 400),
            "model__max_depth": [2, 3, 4, None],
            "model__min_samples_leaf": randint(5, 40),
            "model__l2_regularization": loguniform(1e-3, 10.0),
        },
    ),
}

# Referenčná konfigurácia (aktuálny stav projektu); pridá sa ku všetkým modelom, ak je v mriežke.
BASELINE = ("iterative", None, "standard", (0.01, 0.99), True, "count", ("diff",))

FULL = dict(
    imputers=["median", "mean", "knn", "iterative"],
    knn_neighbors=[7, 14],
    scalers=["standard", "robust", "none"],
    winsors=["0.01:0.99", "none", "0.02:0.98", "0.05:0.95", "0.10:0.90"],
    logs=["1", "0"],
    missing=["none", "count", "each"],
    engineer=["none", "mean", "slope", "diff", "mean+slope", "mean+diff", "slope+diff", "mean+slope+diff"],
)
QUICK = dict(
    imputers=["median", "iterative"],
    knn_neighbors=[7],
    scalers=["standard"],
    winsors=["0.01:0.99", "0.05:0.95"],
    logs=["1", "0"],
    missing=["none", "count"],
    engineer=["none", "diff", "mean+slope+diff"],
)


# --------------------------------------------------------------------------- #
# Pomocné funkcie
# --------------------------------------------------------------------------- #
LOGFILE: Path | None = None


def log(*parts) -> None:
    text = " ".join(str(p) for p in parts)
    print(text, flush=True)
    if LOGFILE is not None:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def _none(s: str):
    return None if s.lower() == "none" else s


def parse_winsor(s: str):
    if s.lower() == "none":
        return None
    lo, hi = s.split(":")
    return (float(lo), float(hi))


def parse_engineer(s: str) -> tuple:
    return () if s.lower() == "none" else tuple(s.split("+"))


def fmt_eng(e: tuple) -> str:
    return "+".join(e) if e else "-"


def cfg_kwargs(cfg) -> dict:
    imputer, k, scaler, winsor, log_, missing, eng = cfg
    return dict(
        imputer=imputer,
        scaler=scaler,
        winsor=winsor,
        log=log_,
        missing=missing,
        engineer=eng if eng else False,
        n_neighbors=k or 7,
    )


def cfg_call(cfg) -> str:
    """Text volania build_preprocessor, ktorý sa dá skopírovať do kódu."""
    kw = cfg_kwargs(cfg)
    return "build_preprocessor(" + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + ")"


def make_pipe(cfg, model, memory=None) -> Pipeline:
    prep = build_preprocessor(**cfg_kwargs(cfg))
    return Pipeline(list(prep.steps) + [("model", model)], memory=memory)


def enumerate_bases(a) -> list[tuple]:
    imps: list[tuple] = []
    for imp in a.imputers:
        if imp == "knn":
            imps += [("knn", int(k)) for k in a.knn_neighbors]
        else:
            imps.append((imp, None))
    scalers = [_none(s) for s in a.scalers]
    winsors = [parse_winsor(w) for w in a.winsors]
    logs = [bool(int(x)) for x in a.logs]
    return [(i, k, s, w, l) for (i, k) in imps for s in scalers for w in winsors for l in logs]


def enumerate_variants(a) -> list[tuple]:
    return [(_none(m), parse_engineer(e)) for m in a.missing for e in a.engineer]


def assign_combos(a, bases, variants) -> dict[str, list]:
    """Pre každý model zoznam (base, variant); pomalé modely dostanú náhodný výber."""
    allc = [(b, v) for b in bases for v in variants]
    ref = None
    if (BASELINE[:5], BASELINE[5:]) in set(allc):
        ref = (BASELINE[:5], BASELINE[5:])
    rng = random.Random(a.seed)
    out = {}
    for name in a.models:
        if MODELS[name]["slow"] and len(allc) > a.slow_configs:
            sel = rng.sample(allc, a.slow_configs)
            if ref is not None and ref not in sel:
                sel.append(ref)
        else:
            sel = list(allc)
        out[name] = sel
    return out


# --------------------------------------------------------------------------- #
# Fáza 1: jeden fold jednej základnej konfigurácie
# --------------------------------------------------------------------------- #
def _fold_task(X, y, tr, va, fi, base, wanted, seed):
    """Predpripraví dáta (winsor, log, scaler, imputer) a vyhodnotí všetky požadované modely/varianty.

    wanted: {model: [(missing, engineer), ...]}
    Vracia (base, fi, {(model, variant): mcc}).
    """
    imputer, k, scaler, winsor, log_ = base
    Xtr, Xva = X.iloc[tr], X.iloc[va]
    ytr, yva = y[tr], y[va]
    steps = []
    if winsor:
        steps.append(Winsorizer(lower=winsor[0], upper=winsor[1]))
    if log_:
        steps.append(Log1p())
    sc = make_scaler(scaler)
    if sc is not None:
        steps.append(ColumnWise(sc, NUM_COLS))
    steps.append(ColumnWise(make_imputer(imputer, n_neighbors=k or 7), FEATURE_COLS))
    A, B = Xtr, Xva
    for s in steps:
        A = s.fit_transform(A)
        B = s.transform(B)

    fe_cache: dict = {}
    mi_cache: dict = {}
    var_cache: dict = {}

    def matrices(variant):
        if variant in var_cache:
            return var_cache[variant]
        m, e = variant
        if e not in fe_cache:
            if e:
                fe = FeatureEngineer(features=e).fit(A)
                fe_cache[e] = (fe.transform(A), fe.transform(B))
            else:
                fe_cache[e] = (A, B)
        FA, FB = fe_cache[e]
        if m is not None:
            if m not in mi_cache:
                mi = MissingIndicator(mode=m).fit(Xtr)
                mi_cache[m] = (
                    mi.transform(Xtr).drop(columns=Xtr.columns),
                    mi.transform(Xva).drop(columns=Xtr.columns),
                )
            ea, eb = mi_cache[m]
            FA = pd.concat([FA, ea], axis=1)
            FB = pd.concat([FB, eb], axis=1)
        var_cache[variant] = (FA.to_numpy(), FB.to_numpy())
        return var_cache[variant]

    out = {}
    for model, variants in wanted.items():
        for variant in variants:
            try:
                GA, GB = matrices(variant)
                clf = MODELS[model]["make"](seed).fit(GA, ytr)
                out[(model, variant)] = matthews_corrcoef(yva, clf.predict(GB))
            except Exception as exc:  # jeden zlyhaný fit nemá zhodiť celý beh
                out[(model, variant)] = float("nan")
    return base, fi, out


def run_stage1(a, X, y, outdir: Path, bases, variants, assignment):
    pkl = outdir / "stage1.pkl"
    meta = json.dumps(
        {k: getattr(a, k) for k in ("models", "imputers", "knn_neighbors", "scalers", "winsors", "logs",
                                     "missing", "engineer", "slow_configs", "screen_folds", "screen_repeats", "seed")},
        sort_keys=True,
    )
    state = {"meta": meta, "res": {}, "done": set()}
    if pkl.exists():
        old = joblib.load(pkl)
        if old["meta"] != meta:
            sys.exit(f"V {pkl} sú výsledky s INÝM nastavením. Použi --fresh alebo iný --out.")
        state = old
        log(f"[fáza 1] pokračujem: hotových {len(state['done'])}/{len(bases)} základných konfigurácií")

    folds = list(
        RepeatedStratifiedKFold(
            n_splits=a.screen_folds, n_repeats=a.screen_repeats, random_state=a.seed
        ).split(X, y)
    )
    per_base: dict = {b: {} for b in bases}
    for model, combos in assignment.items():
        for b, v in combos:
            per_base[b].setdefault(model, []).append(v)

    pending = [b for b in bases if b not in state["done"] and per_base[b]]
    n_eff = a.jobs if a.jobs > 0 else (os.cpu_count() or 1)
    batch = max(1, math.ceil(n_eff / len(folds)))
    t0 = time.perf_counter()
    done_now = 0
    for i in range(0, len(pending), batch):
        chunk = pending[i : i + batch]
        tasks = [
            delayed(_fold_task)(X, y, tr, va, fi, b, per_base[b], a.seed)
            for b in chunk
            for fi, (tr, va) in enumerate(folds)
        ]
        results = Parallel(n_jobs=a.jobs)(tasks)
        agg: dict = {}
        for b, fi, d in results:
            for key, val in d.items():
                agg.setdefault((b, key), np.full(len(folds), np.nan))[fi] = val
        for (b, (model, (m, e))), arr in agg.items():
            state["res"][(model, b + (m, e))] = arr
        state["done"].update(chunk)
        joblib.dump(state, pkl)
        done_now += len(chunk)
        el = time.perf_counter() - t0
        eta = el / done_now * (len(pending) - done_now)
        log(f"[fáza 1] {len(state['done'])}/{len(bases)} základných konfigurácií "
            f"({el/60:.1f} min, zostáva ~{eta/60:.1f} min)")
    return state["res"]


def frame_from_res(res: dict, extra: dict | None = None) -> pd.DataFrame:
    rows = []
    for (model, cfg), s in res.items():
        s = np.asarray(s, dtype=float)
        rows.append(
            dict(
                model=model, imputer=cfg[0], knn_k=cfg[1], scaler=cfg[2],
                winsor="none" if cfg[3] is None else f"{cfg[3][0]}:{cfg[3][1]}",
                log=cfg[4], missing="none" if cfg[5] is None else cfg[5], engineer=fmt_eng(cfg[6]),
                mean=np.nanmean(s), se=np.nanstd(s) / np.sqrt(np.sum(~np.isnan(s))), cfg=cfg,
            )
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["model", "mean"], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Fáza 2: potvrdenie na novom seede
# --------------------------------------------------------------------------- #
def run_stage2(a, X, y, outdir: Path, df1: pd.DataFrame):
    pkl = outdir / "stage2.pkl"
    res2 = joblib.load(pkl) if pkl.exists() else {}
    cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=a.confirm_repeats, random_state=a.seed + 100)
    todo = []
    for model in a.models:
        sub = df1[df1.model == model].dropna(subset=["mean"]).sort_values("mean", ascending=False)
        cfgs = list(sub.head(a.top_k)["cfg"])
        if BASELINE in set(sub["cfg"]) and BASELINE not in cfgs:
            cfgs.append(BASELINE)
        todo += [(model, c) for c in cfgs]
    t0 = time.perf_counter()
    for i, (model, cfg) in enumerate(todo, 1):
        if (model, cfg) in res2:
            continue
        pipe = make_pipe(cfg, MODELS[model]["make"](a.seed))
        res2[(model, cfg)] = cross_val_score(pipe, X, y, cv=cv2, scoring=MCC, n_jobs=a.jobs)
        joblib.dump(res2, pkl)
        log(f"[fáza 2] {i}/{len(todo)} {model:6s} {res2[(model, cfg)].mean():.4f}  ({(time.perf_counter()-t0)/60:.1f} min)")
    keep = {k: v for k, v in res2.items() if k in set(todo)}
    df2 = frame_from_res(keep)
    s1 = {(r.model, r.cfg): r["mean"] for _, r in df1.iterrows()}
    df2["stage1_mean"] = [s1.get((r.model, r.cfg), np.nan) for _, r in df2.iterrows()]
    return df2


# --------------------------------------------------------------------------- #
# Fáza 3: ladenie hyperparametrov (vnorená CV)
# --------------------------------------------------------------------------- #
def run_stage3(a, X, y, outdir: Path, df2: pd.DataFrame):
    pkl = outdir / "stage3.pkl"
    res3 = joblib.load(pkl) if pkl.exists() else {}
    cache_dir = outdir / "cache"
    mem = Memory(location=str(cache_dir), verbose=0)
    outer = StratifiedKFold(a.outer_folds, shuffle=True, random_state=a.seed + 200)
    inner = StratifiedKFold(a.inner_folds, shuffle=True, random_state=a.seed + 300)
    todo = []
    for model in a.models:
        sub = df2[df2.model == model].sort_values("mean", ascending=False)
        todo += [(model, c) for c in list(sub.head(a.tune_top)["cfg"])]
    t0 = time.perf_counter()
    for i, (model, cfg) in enumerate(todo, 1):
        key = (model, cfg)
        fpath = outdir / f"final_{model}_{hashlib.md5(repr(cfg).encode()).hexdigest()[:8]}.joblib"
        if key in res3 and fpath.exists():
            continue
        spec = MODELS[model]
        base_pipe = make_pipe(cfg, spec["make"](a.seed), memory=mem)
        untuned = cross_val_score(clone_pipe(base_pipe), X, y, cv=outer, scoring=MCC)

        def new_search():
            return RandomizedSearchCV(
                clone_pipe(base_pipe), spec["space"], n_iter=a.n_iter, scoring=MCC,
                cv=inner, random_state=a.seed, n_jobs=a.jobs, refit=True,
            )

        nested = cross_val_score(new_search(), X, y, cv=outer, scoring=MCC)
        search = new_search().fit(X, y)
        final = search.best_estimator_
        final.set_params(memory=None)  # samostatný Pipeline bez závislosti na cache
        joblib.dump(final, fpath)
        res3[key] = dict(
            untuned=untuned, nested=nested, best_params=clean_params(search.best_params_),
            inner_best=float(search.best_score_), file=fpath.name,
        )
        joblib.dump(res3, pkl)
        d = nested - untuned
        log(f"[fáza 3] {i}/{len(todo)} {model:6s} bez ladenia {untuned.mean():.4f} -> "
            f"ladené (nested) {nested.mean():.4f}  rozdiel {d.mean():+.4f} ± {d.std()/np.sqrt(len(d)):.4f}  "
            f"({(time.perf_counter()-t0)/60:.1f} min)")
    shutil.rmtree(cache_dir, ignore_errors=True)
    keep = {k: v for k, v in res3.items() if k in set(todo)}
    return keep


def clean_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(f"{float(v):.5g}")
        out[k.replace("model__", "")] = v
    return out


def clone_pipe(p: Pipeline) -> Pipeline:
    from sklearn.base import clone
    return clone(p)


# --------------------------------------------------------------------------- #
# Súhrn
# --------------------------------------------------------------------------- #
def make_report(a, outdir: Path, df1, df2, res3):
    rows = []
    for (model, cfg), r in res3.items():
        d = r["nested"] - r["untuned"]
        s2 = df2[(df2.model == model) & df2["cfg"].apply(lambda c: c == cfg)]
        rows.append(dict(
            model=model,
            stage1=float(df1[(df1.model == model) & df1["cfg"].apply(lambda c: c == cfg)]["mean"].iloc[0]),
            stage2=float(s2["mean"].iloc[0]),
            bez_ladenia=r["untuned"].mean(),
            ladene_nested=r["nested"].mean(),
            nested_se=r["nested"].std() / np.sqrt(len(r["nested"])),
            rozdiel=d.mean(), rozdiel_se=d.std() / np.sqrt(len(d)),
            preprocessing=cfg_call(cfg), hyperparametre=r["best_params"], subor=r["file"], cfg=cfg,
        ))
    df3 = pd.DataFrame(rows).sort_values("ladene_nested", ascending=False).reset_index(drop=True)
    df3.drop(columns=["cfg"]).to_csv(outdir / "stage3.csv", index=False)

    lines = ["VÝSLEDKY HĽADANIA", "=" * 70, ""]
    lines.append("Poradie (MCC z vnorenej CV; stage1 = preosiatie, stage2 = potvrdenie na novom seede):\n")
    lines.append(df3[["model", "stage1", "stage2", "bez_ladenia", "ladene_nested", "nested_se",
                      "rozdiel", "rozdiel_se"]].round(4).to_string(index=False))
    lines += ["", "NAJLEPŠIA KONFIGURÁCIA KAŽDÉHO MODELU", "-" * 70]
    best_per_model = df3.drop_duplicates("model")
    for _, r in best_per_model.iterrows():
        lines += [f"\n{r.model}:  nested MCC {r.ladene_nested:.4f} ± {r.nested_se:.4f}",
                  f"  {r.preprocessing}",
                  f"  hyperparametre: {r.hyperparametre}",
                  f"  uložený model: {outdir.name}/{r.subor}"]
    lines += ["", "POZNÁMKY",
              "- Rozdiel menší než ~2 * rozdiel_se je v rámci šumu: ladenie vtedy reálne nepomohlo.",
              "- Poradie modelov s rozdielom menším než ~2 * nested_se nepovažuj za rozhodnuté.",
              "- Preprocessing bol vyberaný z veľa kombinácií, preto je nested MCC mierne optimistické;",
              "  skutočná hodnota na skrytých dátach môže byť o niečo nižšia."]
    text = "\n".join(lines)
    (outdir / "report.txt").write_text(text, encoding="utf-8")
    best = df3.iloc[0]
    shutil.copy(outdir / best["subor"], outdir / "best_overall.joblib")
    for _, r in best_per_model.iterrows():
        shutil.copy(outdir / r["subor"], outdir / f"best_{r.model}.joblib")
    return text


# --------------------------------------------------------------------------- #
# Odhad času
# --------------------------------------------------------------------------- #
def estimate(a, X, y, bases, variants, assignment):
    folds = list(RepeatedStratifiedKFold(n_splits=a.screen_folds, n_repeats=a.screen_repeats,
                                         random_state=a.seed).split(X, y))
    tr, va = folds[0]
    log("Meriam čas jedného fitu (chvíľu to potrvá)...")
    imp_time = {}
    for b in bases:
        key = (b[0], b[1])
        if key in imp_time:
            continue
        t = time.perf_counter()
        _fold_task(X, y, tr, va, 0, b, {}, a.seed)
        imp_time[key] = time.perf_counter() - t
    ref_base = bases[0]
    fit_time = {}
    for name in a.models:
        t = time.perf_counter()
        _fold_task(X, y, tr, va, 0, ref_base, {name: [variants[0]]}, a.seed)
        fit_time[name] = max(time.perf_counter() - t - imp_time[(ref_base[0], ref_base[1])], 1e-3)
    n_eff = a.jobs if a.jobs > 0 else (os.cpu_count() or 1)
    f = len(folds)
    s1 = sum(imp_time[(b[0], b[1])] * f for b in bases)
    s1 += sum(len(assignment[m]) * f * fit_time[m] for m in a.models)
    imp_ref = imp_time.get(("iterative", None), max(imp_time.values()))
    s2 = sum((a.top_k + 1) * 5 * a.confirm_repeats * (imp_ref + fit_time[m]) for m in a.models)
    s3 = sum(a.tune_top * ((a.outer_folds * a.inner_folds * a.n_iter) + a.inner_folds * a.n_iter + 2 * a.outer_folds)
             * fit_time[m] for m in a.models)
    log(f"\nZákladných konfigurácií (imputer x scaler x winsor x log): {len(bases)}")
    log(f"Variantov (missing x engineer): {len(variants)}")
    for m in a.models:
        log(f"  {m:7s} kombinácií vo fáze 1: {len(assignment[m]):5d}   fit ~ {fit_time[m]:.3f} s")
    eff = 1.0 if n_eff == 1 else 0.7
    log(f"\nJadier: {n_eff}  (efektivita paralelizácie ~{eff*100:.0f} %)")
    for name, sec in (("fáza 1", s1), ("fáza 2", s2), ("fáza 3", s3)):
        log(f"  {name}: ~{sec/60:6.1f} min CPU  ->  ~{sec/60/max(n_eff*eff,1):6.1f} min reálne")
    tot = (s1 + s2 + s3) / 60 / max(n_eff * eff, 1)
    log(f"  SPOLU: ~{tot:.0f} min ({tot/60:.1f} h)")


# --------------------------------------------------------------------------- #
# Hlavný program
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Hľadanie predspracovania a hyperparametrov pre každý model.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=None, help="cesta k train.csv (predvolene data/train.csv)")
    p.add_argument("--out", default="search_out", help="výstupný priečinok")
    p.add_argument("--models", nargs="+", default=["logreg", "lda", "knn", "rf", "et", "mlp", "hgb"],
                   choices=list(MODELS))
    p.add_argument("--quick", action="store_true", help="malá mriežka na rýchle vyskúšanie")
    p.add_argument("--estimate", action="store_true", help="len odhad času a konfigurácií")
    p.add_argument("--fresh", action="store_true", help="zmazať predchádzajúce výsledky v --out")
    p.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--jobs", type=int, default=-1, help="počet procesov (-1 = všetky jadrá)")
    p.add_argument("--seed", type=int, default=42)
    g = p.add_argument_group("mriežka predspracovania (predvolené podľa --quick/plná)")
    g.add_argument("--imputers", nargs="+", choices=["median", "mean", "knn", "iterative"])
    g.add_argument("--knn-neighbors", dest="knn_neighbors", nargs="+", type=int)
    g.add_argument("--scalers", nargs="+", help="standard robust none")
    g.add_argument("--winsors", nargs="+", help='napr. 0.01:0.99 0.05:0.95 none')
    g.add_argument("--logs", nargs="+", help="1 = log zapnutý, 0 = vypnutý")
    g.add_argument("--missing", nargs="+", help="none count each")
    g.add_argument("--engineer", nargs="+", help="none mean slope diff mean+slope ... ma_diff ex_mean ...")
    s = p.add_argument_group("fázy")
    s.add_argument("--slow-configs", dest="slow_configs", type=int, default=60,
                   help="fáza 1: počet náhodných kombinácií pre pomalé modely (rf, et, mlp, hgb)")
    s.add_argument("--screen-folds", dest="screen_folds", type=int, default=5)
    s.add_argument("--screen-repeats", dest="screen_repeats", type=int, default=1)
    s.add_argument("--top-k", dest="top_k", type=int, default=6, help="fáza 2: koľko najlepších konfigurácií potvrdiť")
    s.add_argument("--confirm-repeats", dest="confirm_repeats", type=int, default=5)
    s.add_argument("--tune-top", dest="tune_top", type=int, default=2, help="fáza 3: koľko konfigurácií ladiť na model")
    s.add_argument("--n-iter", dest="n_iter", type=int, default=25, help="fáza 3: počet vyskúšaných hyperparametrov")
    s.add_argument("--outer-folds", dest="outer_folds", type=int, default=5)
    s.add_argument("--inner-folds", dest="inner_folds", type=int, default=3)
    a = p.parse_args()
    preset = QUICK if a.quick else FULL
    for k, v in preset.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def main():
    global LOGFILE
    a = parse_args()
    outdir = SLOZKA / a.out
    if a.fresh and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    LOGFILE = outdir / "log.txt"

    X, y = nacti_xy(a.data) if a.data else nacti_xy()
    bases = enumerate_bases(a)
    variants = enumerate_variants(a)
    assignment = assign_combos(a, bases, variants)

    if a.estimate:
        estimate(a, X, y, bases, variants, assignment)
        return

    t0 = time.perf_counter()
    log(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} | modely: {a.models} | "
        f"{len(bases)} základných konfigurácií x {len(variants)} variantov ===")
    res1 = run_stage1(a, X, y, outdir, bases, variants, assignment) if 1 in a.stages else \
        joblib.load(outdir / "stage1.pkl")["res"]
    df1 = frame_from_res(res1)
    df1.drop(columns=["cfg"]).to_csv(outdir / "stage1.csv", index=False)
    log("\nTOP 3 z fázy 1 pre každý model:")
    for m in a.models:
        top = df1[df1.model == m].head(3)
        for _, r in top.iterrows():
            log(f"  {m:6s} {r['mean']:.4f}  {r.imputer}/{r.knn_k} {r.scaler} wins={r.winsor} log={r.log} "
                f"missing={r.missing} eng={r.engineer}")
    if 2 not in a.stages:
        return
    df2 = run_stage2(a, X, y, outdir, df1)
    df2.drop(columns=["cfg"]).to_csv(outdir / "stage2.csv", index=False)
    if 3 not in a.stages:
        return
    res3 = run_stage3(a, X, y, outdir, df2)
    text = make_report(a, outdir, df1, df2, res3)
    log("\n" + text)
    log(f"\nCelkový čas: {(time.perf_counter()-t0)/60:.1f} min. Výstupy: {outdir}")


if __name__ == "__main__":
    main()
