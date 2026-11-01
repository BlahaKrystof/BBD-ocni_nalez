"""Průzkumná analýza dat (EDA) pro téma 1: detekce diabetické retinopatie.

Účel
    Prozkoumat trénovací data před návrhem předzpracování a modelu. Skript data
    pouze analyzuje: nic nemaže ani neupravuje. O předzpracování (imputace,
    ořezání odlehlých hodnot, transformace, škálování) se rozhoduje až podle
    výstupů a výsledné kroky patří do sklearn.Pipeline v train.py.

Vstup
    CSV soubor ve formátu zadání (oddělovač ",", UTF-8, desetinná tečka) se
    sloupci: 3 binární příznaky (kvalita, prescreening, am_fm_klasifikace),
    ma_detekce_01..06, exsudaty_01..08, vzdalenost_makula_papila, prumer_papily
    a cílová proměnná target (0/1). Chybějící hodnoty jsou prázdná pole.

Výstup
    - Textové výpisy na standardní výstup (sekce viz třída EDA).
    - Obrázky ve složce graphs/ vedle skriptu:
        korelace.png        heatmapa korelační matice všech sloupců,
        hist_<priznak>.png  histogram (podle tříd) a boxplot pro každý spojitý příznak.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # bez GUI, grafy se jen ukládají do souborů
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SLOZKA = Path(__file__).resolve().parent.parent
"""Adresář, ve kterém leží tento skript; cesty se skládají relativně k němu."""

DATA_PATH = SLOZKA / "data" / "ocni_nalez.csv"
"""Výchozí cesta k datům, pokud není zadána v příkazové řádce."""

GRAFY = SLOZKA / "graphs"
"""Složka pro ukládané obrázky (vytvoří se při vytvoření objektu EDA)."""

TARGET = "target"
"""Název cílového sloupce (0 = bez známek retinopatie, 1 = se známkami)."""

BINARNI = ["kvalita", "prescreening", "am_fm_klasifikace"]
"""Binární příznaky; neobsahují NaN a nepočítají se mezi spojité příznaky."""


def load_data(cesta: Path | str = DATA_PATH) -> pd.DataFrame:
    """Načte CSV soubor beze změny (nic se nemaže ani nepřepisuje).

    Args:
        cesta: Cesta k CSV souboru (str nebo Path). Výchozí je DATA_PATH.

    Returns:
        DataFrame se všemi sloupci včetně `target`, ve stejném pořadí řádků
        i sloupců jako v souboru. Chybějící hodnoty zůstávají jako NaN.

    Raises:
        AssertionError: Pokud v souboru chybí sloupec `target`.
        FileNotFoundError: Pokud soubor neexistuje.
    """
    df = pd.read_csv(cesta)
    assert TARGET in df.columns, f"Chybí sloupec '{TARGET}'"
    return df


def spojite_priznaky(df: pd.DataFrame) -> list[str]:
    """Vrátí názvy spojitých a počtových příznaků.

    Args:
        df: DataFrame načtený funkcí `load_data`.

    Returns:
        Seznam názvů sloupců v pořadí, v jakém jsou v `df`.
    """
    return [c for c in df.columns if c not in BINARNI + [TARGET]]


def hlavicka(text: str) -> None:
    """Vypíše na standardní výstup oddělovač a nadpis jedné sekce analýzy.

    Args:
        text: Text nadpisu sekce.
    """
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


class EDA:
    """Průzkumná analýza datové sady; každá metoda zpracuje jednu oblast.

    Metody pouze vypisují výsledky na standardní výstup a případně ukládají
    grafy do složky GRAFY. Žádná metoda nemění `self.df` ani nic nevrací.
    Celou analýzu v pořadí spustí metoda `run`.

    Attributes:
        df: Analyzovaná data (včetně `target`), beze změn oproti souboru.
        spojite: Názvy spojitých a počtových příznaků (viz `spojite_priznaky`).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Uloží data, určí spojité příznaky a vytvoří složku pro grafy.

        Args:
            df: DataFrame obsahující sloupec `target` (typicky z `load_data`).

        Raises:
            AssertionError: Pokud `df` není pandas.DataFrame.
        """
        assert isinstance(df, pd.DataFrame), "Vstup musí být pandas.DataFrame"

        self.df = df
        self.spojite = spojite_priznaky(df)
        GRAFY.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    def zakladni_info(self) -> None:
        """Vypíše základní informace o datech.

        Výstup:
            - tvar tabulky (řádky, sloupce),
            - počet duplicitních řádků (jen se počítají, nemažou),
            - počet záporných hodnot ve spojitých příznacích (mělo by být 0),
            - rozdělení tříd `target`: počty a podíly.
        """
        hlavicka("ZÁKLADNÍ INFO")
        print(f"Tvar: {self.df.shape}")
        print(f"Duplicitní řádky: {int(self.df.duplicated().sum())}")
        zaporne = int((self.df[self.spojite] < 0).sum().sum())
        print(f"Záporné hodnoty ve spojitých příznacích: {zaporne}")
        print("\nRozdělení tříd:")
        print(self.df[TARGET].value_counts().sort_index().to_string())
        print(self.df[TARGET].value_counts(normalize=True).sort_index().round(3).to_string())

    def chybejici_hodnoty(self) -> None:
        """Analyzuje chybějící hodnoty (NaN).

        Výstup:
            - tabulka sloupců s NaN: počet (NaN_N) a procento (NaN_%),
            - počet řádků s alespoň jedním NaN (kolik by se ztratilo mazáním),
            - kontrola, že binární sloupce a `target` NaN neobsahují (mělo by být 0),
            - histogram počtu chybějících hodnot na řádek,
            - pro každý spojitý příznak podíl třídy 1 mezi řádky, kde příznak
              chybí (sloupec `chybi`) a kde nechybí (`nechybi`). Výrazný rozdíl
              naznačuje, že samotné chybění nese informaci a vyplatí se přidat
              indikátor chybění.
        """
        hlavicka("CHYBĚJÍCÍ HODNOTY")
        n = self.df.isna().sum()
        tab = pd.DataFrame({"NaN_N": n, "NaN_%": (100 * n / len(self.df)).round(2)})
        print(tab[tab.NaN_N > 0].to_string())
        print(f"\nŘádků s alespoň jedním NaN: {int(self.df.isna().any(axis=1).sum())} z {len(self.df)}")
        print(f"Počet NaN v binárních sloupcích a targetu: "
              f"{int(self.df[BINARNI + [TARGET]].isna().sum().sum())} (mělo by být 0)")

        # Chybí hodnoty spolu (stejné řádky), nebo náhodně?
        maska = self.df[self.spojite].isna().astype(int)
        print("\nPočet chybějících hodnot na řádek (histogram):")
        print(maska.sum(axis=1).value_counts().sort_index().to_string())

        # Souvisí chybění s targetem? (kdyby ano, hodil by se indikátor chybění)
        print("\nPodíl třídy 1 mezi řádky, kde příznak chybí / nechybí:")
        radky = []
        for c in self.spojite:
            chybi = self.df[c].isna()
            radky.append((c, self.df.loc[chybi, TARGET].mean(), self.df.loc[~chybi, TARGET].mean()))
        print(pd.DataFrame(radky, columns=["priznak", "chybi", "nechybi"]).round(3).to_string(index=False))

    def popisne_statistiky(self) -> None:
        """Vypíše popisné statistiky spojitých příznaků (NaN se ignorují).

        Výstup (jeden řádek na příznak): count (počet nechybějících hodnot),
        mean, std, min, 25 %, 50 % (medián), 75 %, max a skew (šikmost).
        Šikmost výrazně nad 1 značí silně pravostranné rozdělení, kterému
        pomůže transformace (např. log1p) nebo ořezání extrémů.
        """
        hlavicka("POPISNÉ STATISTIKY (spojité příznaky)")
        d = self.df[self.spojite].describe().T
        d["skew"] = self.df[self.spojite].skew()
        pd.set_option("display.width", 200)
        print(d.round(4).to_string())

    def odlehle_hodnoty(self, prah_iqr: float = 3.0, prah_z: float = 3.0) -> None:
        """Spočítá odlehlé hodnoty ve spojitých příznacích dvěma metodami.

        Odlehlé hodnoty se pouze počítají; žádný řádek se neodstraňuje.
        NaN se při výpočtu ignorují.

        Args:
            prah_iqr: Násobek IQR; za odlehlou se považuje hodnota mimo interval
                [Q1 - k*IQR, Q3 + k*IQR]. Výchozí 3.0 (jen extrémní hodnoty).
            prah_z: Hodnota je odlehlá, pokud |z-skóre| > prah_z. Výchozí 3.0.

        Výstup (jeden řádek na příznak): IQR_N (počet odlehlých podle IQR),
        Z_N (počet podle Z-skóre), max a q99 (99. percentil). Velký poměr
        max / q99 ukazuje na dlouhý ocas rozdělení. Z-skóre je zkreslené
        samotnými extrémy, proto je pro rozhodování vhodnější IQR.
        """
        hlavicka(f"ODLEHLÉ HODNOTY (IQR, k = {prah_iqr}  |  Z-skóre > {prah_z})")
        radky = []
        for c in self.spojite:
            x = self.df[c].dropna()
            q1, q3 = x.quantile([0.25, 0.75])
            iqr = q3 - q1
            dole, nahore = q1 - prah_iqr * iqr, q3 + prah_iqr * iqr
            n_iqr = int(((x < dole) | (x > nahore)).sum())
            n_z = int((np.abs((x - x.mean()) / x.std()) > prah_z).sum())
            radky.append((c, n_iqr, n_z, round(x.max(), 3), round(x.quantile(0.99), 3)))
        print(pd.DataFrame(radky, columns=["priznak", "IQR_N", "Z_N", "max", "q99"]).to_string(index=False))

    def korelace(self) -> None:
        """Analyzuje korelace příznaků mezi sebou a s cílovou proměnnou.

        Výstup:
            - Spearmanova korelace každého příznaku s `target`, seřazená podle
              absolutní hodnoty (síla vztahu k cíli),
            - dvojice sloupců s |r| > 0.9 (Pearson) seřazené sestupně; ukazují
              redundantní příznaky a riziko multikolinearity u lineárních modelů,
            - heatmapa celé korelační matice (Pearson) uložená do graphs/korelace.png.
        """
        hlavicka("KORELACE")
        k = self.df.corr(numeric_only=True)
        print("Korelace příznaků s targetem (Spearman, seřazeno podle |r|):")
        sp = self.df.corr(method="spearman", numeric_only=True)[TARGET].drop(TARGET)
        print(sp.reindex(sp.abs().sort_values(ascending=False).index).round(3).to_string())

        # Dvojice s vysokou korelací -> redundance, riziko multikolinearity
        horni = k.where(np.triu(np.ones(k.shape, dtype=bool), k=1))
        dvojice = horni.stack().loc[lambda s: s.abs() > 0.9].sort_values(key=np.abs, ascending=False)
        print(f"\nDvojic příznaků s |r| > 0.9: {len(dvojice)} (prvních 15):")
        print(dvojice.head(15).round(3).to_string())

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(k, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(k)), k.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(k)), k.columns, fontsize=7)
        fig.colorbar(im)
        ax.set_title("Korelační matice")
        fig.tight_layout()
        fig.savefig(GRAFY / "korelace.png", dpi=120)
        plt.close(fig)

    def monotonie_prahu(self) -> None:
        """Ověří, zda hodnoty klesají s rostoucí přísností úrovně spolehlivosti.

        Pro skupiny `ma_detekce_01..06` a `exsudaty_01..08` porovnává sousední
        úrovně (01 vs. 02, 02 vs. 03, ...). Dvojice, kde chybí některá z hodnot,
        se přeskočí.

        Výstup: pro každou skupinu podíl sousedních dvojic s nerostoucí hodnotou
        (1.000 = vždy platí). Pokud platí, dávají smysl odvozené příznaky typu
        rozdíl nebo poměr sousedních úrovní a lze jimi i imputovat chybějící
        hodnotu z okolních úrovní.
        """
        hlavicka("MONOTÓNNOST ÚROVNÍ SPOLEHLIVOSTI")
        for predpona, n in (("ma_detekce_", 6), ("exsudaty_", 8)):
            sloupce = [f"{predpona}{i:02d}" for i in range(1, n + 1)]
            x = self.df[sloupce]
            klesajici = (x.diff(axis=1).iloc[:, 1:] <= 1e-12)
            platne = x.diff(axis=1).iloc[:, 1:].notna()
            podil = (klesajici & platne).sum().sum() / platne.sum().sum()
            print(f"{predpona}*: podíl sousedních dvojic s nerostoucí hodnotou = {podil:.3f}")

    def binarni_priznaky(self) -> None:
        """Vypíše kontingenční tabulky binárních příznaků proti `target`.

        Výstup: pro každý sloupec z BINARNI tabulka počtů (řádky = hodnota
        příznaku, sloupce = třída target) včetně součtů (All). Ukazuje, jak
        dobře daný binární příznak odděluje třídy a zda je některá jeho
        hodnota téměř nevyužitá.
        """
        hlavicka("BINÁRNÍ PŘÍZNAKY vs. TARGET")
        for c in BINARNI:
            print(f"\n{c}:")
            print(pd.crosstab(self.df[c], self.df[TARGET], margins=True).to_string())

    # ------------------------------------------------------------------
    def histogramy(self) -> None:
        """Vykreslí rozdělení každého spojitého příznaku a uloží ho do souboru.

        Pro každý příznak vznikne obrázek graphs/hist_<priznak>.png se dvěma
        grafy: vlevo překrývající se histogramy obou tříd (30 binů z rozsahu
        daného příznaku, NaN vynechány), vpravo boxploty podle třídy.
        Slouží k vizuální kontrole šikmosti, extrémů a rozdílu mezi třídami.
        """
        hlavicka("HISTOGRAMY A BOXPLOTY")
        for c in self.spojite:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
            x = self.df[c].dropna()
            hrany = np.histogram_bin_edges(x, bins=30)
            for t, barva in ((0, "tab:blue"), (1, "tab:red")):
                a1.hist(self.df.loc[self.df[TARGET] == t, c].dropna(), bins=hrany,
                        alpha=0.55, label=f"target = {t}", color=barva)
            a1.set(title=f"Histogram: {c}", xlabel="Hodnota", ylabel="Počet")
            a1.legend()
            a2.boxplot([self.df.loc[self.df[TARGET] == t, c].dropna() for t in (0, 1)],
                       tick_labels=["0", "1"])
            a2.set(title="Boxplot podle třídy", xlabel="target")
            fig.tight_layout()
            fig.savefig(GRAFY / f"hist_{c}.png", dpi=100)
            plt.close(fig)
        print(f"Grafy uloženy do: {GRAFY}")

    def run(self) -> None:
        """Spustí celou analýzu v tomto pořadí a vypíše/uloží všechny výstupy.

        Pořadí: zakladni_info, chybejici_hodnoty, popisne_statistiky,
        odlehle_hodnoty (výchozí prahy), korelace, monotonie_prahu,
        binarni_priznaky, histogramy.
        """
        self.zakladni_info()
        self.chybejici_hodnoty()
        self.popisne_statistiky()
        self.odlehle_hodnoty()
        self.korelace()
        self.monotonie_prahu()
        self.binarni_priznaky()
        self.histogramy()


if __name__ == "__main__":
    cesta = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    EDA(load_data(cesta)).run()