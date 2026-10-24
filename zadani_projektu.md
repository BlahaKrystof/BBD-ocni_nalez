# BPC-UIM Projekt

## Řešitelské týmy a výběr tématu

Řešiteli projektu budou týmy složené z maximálně 3 studentů/studentek.

Každý tým si zvolí jedno téma projektu (viz dole) dle libosti. Doporučujeme důkladně se seznámit s materiály k jednotlivým projektům (včetně popisu zadání níže a přiložených dat a kódů v e-Learningu ➔ Materiály k projektu).

Z každého týmu provede registraci k vybranému tématu v e-learningu právě jeden člen. Kapacita jednoho tématu je omezena na 11 týmů; pokud by kapacita nestačila, bude situace řešena individuálně. Student, který registraci provede, zároveň zašle na adresu Marina.Filipenska@vut.cz seznam členů svého týmu s předmětem:

* **BPC-UIM, projekt, sk. {jméno1, jméno2, jméno3}**

## Konzultace

Projekty bude možné konzultovat s přiřazeným vyučujícím. Konzultace si domlouvejte individuálně přímo s vyučujícím.

## Důležité termíny

Zaslání složení týmů a vybraného tématu projektu na Marina.Filipenska@vut.cz (informace za celý tým zasílá kontaktní osoba): ***do 12. 10. 2026***

Nejzazší termín odevzdání řešení (na 195715@vut.cz): ***6. 12. 2026, 23:59***

## Struktura repozitáře

Tento repozitář je šablona projektu. Soubor `train.csv` svého tématu vložte do složky `data/`.

```
├── testing.py         # vyhodnocení natrénovaného modelu (viz Odevzdání a nezávislé testování)
├── train.py           # trénování modelu
├── requirements.txt   # použité knihovny včetně verzí
├── config/            # nastavení experimentů
├── data/              # train.csv
├── dataio/            # načítání a ukládání dat
├── logs/              # záznamy z trénování
├── models/            # uložený natrénovaný model a naučené předzpracování
├── src/               # předzpracování, model, trénování
├── tests/             # jednotkové testy vašich funkcí
└── utils/             # pomocné funkce; metrics.py obsahuje výpočet matice záměn a MCC
```

Složka `tests/` je určena pro jednotkové testy vašeho kódu, kdežto `testing.py` vyhodnocuje hotový model na datech. Funkce `matice_zamen()` a `mcc()` můžete použít i při trénování a validaci (`from utils import matice_zamen, mcc`).

## Data

Všechna tři témata jsou úlohy binární klasifikace. Ke každému tématu dostanete soubor `train.csv` s trénovacími daty: každý řádek je jeden vzorek, sloupce jsou příznaky a poslední sloupec `target` je cílová proměnná (0/1). Význam sloupců popisuje datový slovník u každého tématu.

- Formát CSV: oddělovač `,`, kódování UTF-8, desetinná tečka.
- Pořadí sloupců i řádků je náhodné a nenese žádnou informaci.
- **Data obsahují chybějící hodnoty** (prázdné pole v CSV). Netýkají se binárních příznaků ani sloupce `target`.
- Hodnoty příznaků nebyly očištěny od odlehlých hodnot. Extrémní hodnoty jsou reálnou součástí dat a jejich ošetření je součástí úlohy.
- Pro řešení používejte pouze dodaná data.

## Odevzdání a nezávislé testování

Odevzdané algoritmy ***budou testovány na nezávislém testovacím souboru dat***, který mají k dispozici pouze vyučující. Testovací soubor má formát shodný s `train.csv`: stejné sloupce ve stejném pořadí, včetně sloupce `target`. Obsahuje chybějící hodnoty ve stejné míře a stejného charakteru jako trénovací data. Vaše řešení proto musí umět chybějící hodnoty ošetřit nejen při trénování, ale i při predikci.

Odevzdejte `.zip` archiv se všemi soubory, které jsou nutné pro spuštění vašeho projektu:

- `requirements.txt` s verzemi použitých knihoven,
- uložený natrénovaný model včetně naučeného předzpracování (např. imputace, škálování),
- `testing.py` vytvořený doplněním přiložené šablony,
- složku `utils/` a veškerý kód, který `testing.py` importuje,
- další potřebné soubory (např. kód, kterým jste model natrénovali).

Šablona `testing.py` načte testovací soubor, předzpracuje jej, zavolá váš model a vypíše matici záměn a MCC. Doplňte v ní pouze funkce `nacti_model()`, `predzpracuj()` a `predikuj()`:

- předzpracování testovacích dat (imputace, ošetření odlehlých hodnot, škálování apod.) používá parametry naučené na trénovacích datech a nesmí odstranit žádný řádek: predikce musí vzniknout pro každý vzorek,
- model se v `testing.py` znovu netrénuje, pouze se načte ze souboru a provede predikci (u neuronových sítí v režimu vyhodnocení, např. `model.eval()` v PyTorch),
- sloupec `target` slouží jen k výpočtu MCC a do predikce vstupovat nesmí,
- cestu k testovacímu souboru (`DATA_PATH`) vyučující při testování přepíše; skript proto nesmí záviset na jiných cestách mimo odevzdaný archiv.

Před odevzdáním si ověřte, že `testing.py` proběhne v čistém prostředí na souboru `train.csv`.

Výstupy odevzdá vybraný člen týmu (kontaktní osoba) za celý tým. O výsledcích testování se dozvíte v přehledové tabulce.

## Hodnocení

Úspěšnost klasifikace se u všech témat hodnotí pomocí matice záměn (confusion matrix) a z ní vypočítaného Matthewsova korelačního koeficientu (MCC):

$$
\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

MCC nabývá hodnot od −1 do 1: hodnota 1 znamená bezchybnou predikci, 0 predikci na úrovni náhody. Na rozdíl od accuracy je vypovídající i u nevyvážených tříd: model, který vždy predikuje většinovou třídu, může mít vysokou accuracy, ale jeho MCC je 0.

Řešení projektu bude hodnoceno soutěžní formou. Body se rozdělí do dvou kategorií:

* **Formální stránka (15 bodů):** Hodnotí se kvalita odevzdaného kódu, správná aplikace metod a celková funkčnost řešení.
* **Úspěšnost (10 bodů):** Všechna odevzdaná funkční řešení budou otestována na nezávislé sadě dat. Na základě dosaženého MCC bude sestaven žebříček, který určí počet získaných bodů (nejlepší tým 10 bodů, nejslabší 1 bod, ostatní budou ohodnoceni poměrově mezi těmito hodnotami). Každý tým soutěží pouze v rámci svého zadání.

***Hodnocení je nastaveno tak, aby podpořilo férovou soutěž mezi týmy a zároveň motivovalo k co nejlepším výsledkům.***

***Plagiátorství bude mít za následek neudělení zápočtu!***

# Téma 1: Detekce diabetické retinopatie z obrazu sítnice

## Motivace

Diabetická retinopatie je poškození drobných cév sítnice způsobené dlouhodobě zvýšenou hladinou glukózy v krvi. Patří mezi nejčastější příčiny ztráty zraku u lidí v produktivním věku, přitom v raných stadiích probíhá bez příznaků. Diabetici proto mají podstupovat pravidelný screening pomocí fotografií očního pozadí, jejichž ruční hodnocení oftalmologem je při velkém počtu pacientů kapacitně náročné. Automatické systémy, které z obrazu sítnice detekují typické léze (mikroaneurysmata, exsudáty) a na jejich základě rozhodnou o podezření na retinopatii, mohou screening výrazně zefektivnit.

## Cíl

Provést co možná nejpřesnější predikci přítomnosti známek diabetické retinopatie z příznaků extrahovaných algoritmy zpracování obrazu z fotografií sítnice.

## Data

Trénovací soubor obsahuje 802 vzorků a 19 příznaků. Každý řádek odpovídá jedné fotografii očního pozadí. Příznaky nebyly měřeny lékařem, ale vypočteny algoritmy zpracování obrazu, proto nemají přímou klinickou interpretaci a jsou navzájem silně korelované. Třídy jsou přibližně vyvážené: třída 0 má 378 vzorků (47,1 %), třída 1 má 424 vzorků (52,9 %).

| sloupec | typ | jednotka | popis |
|---|---|---|---|
| `kvalita` | binární | – | Výsledek automatického hodnocení kvality snímku: 0 = nedostatečná, 1 = dostatečná. |
| `prescreening` | binární | – | Výsledek předběžného screeningu: 1 = závažná abnormalita sítnice, 0 = bez ní. |
| `ma_detekce_01` … `ma_detekce_06` | celočíselný | počet | Počet mikroaneurysmat (MA) nalezených detektorem při 1. až 6. úrovni spolehlivosti (úrovně seřazeny od nejméně po nejvíce přísnou). |
| `exsudaty_01` … `exsudaty_08` | číselný | relativní | Výsledek detekce exsudátů při 1. až 8. úrovni spolehlivosti detektoru; počet lézí normalizovaný průměrem oblasti zájmu (ROI), aby byly srovnatelné snímky různé velikosti. |
| `vzdalenost_makula_papila` | číselný | relativní | Eukleidovská vzdálenost středu makuly (žluté skvrny) a středu terče zrakového nervu (papily), normalizovaná průměrem ROI. |
| `prumer_papily` | číselný | relativní | Průměr terče zrakového nervu (papily). |
| `am_fm_klasifikace` | binární | – | Binární výstup samostatného klasifikátoru založeného na AM/FM (amplitudově-frekvenční modulace) rozkladu obrazu. |
| `target` | binární | – | Cílová proměnná: 1 = snímek nese známky diabetické retinopatie, 0 = bez známek. |

# Téma 2: Diagnostika poruch prokrvení myokardu ze SPECT vyšetření

## Motivace

Ischemická choroba srdeční vzniká při nedostatečném prokrvení (perfuzi) srdečního svalu, nejčastěji v důsledku zúžení věnčitých tepen. Jednou ze standardních neinvazivních metod její diagnostiky je perfuzní scintigrafie myokardu pomocí jednofotonové emisní výpočetní tomografie (SPECT). Pacientovi je podáno radiofarmakum, které se v srdečním svalu hromadí úměrně jeho prokrvení, a snímky se pořizují v klidu a po zátěži. Hodnocení snímků je časově náročné a závisí na zkušenosti kardiologa. Automatická klasifikace z číselných příznaků extrahovaných ze snímků může lékaři posloužit jako druhý názor a urychlit vyhodnocení.

## Cíl

Provést co možná nejpřesnější klasifikaci SPECT vyšetření srdce na normální a abnormální nález z příznaků popisujících prokrvení jednotlivých oblastí myokardu v klidu a při zátěži.

## Data

Trénovací soubor obsahuje 187 vzorků a 44 příznaků. Každý řádek odpovídá jednomu pacientovi, referenční diagnóza pochází od kardiologů. Třídy jsou výrazně nevyvážené: třída 0 (normální nález) má 39 vzorků (20,9 %), třída 1 (abnormální nález) má 148 vzorků (79,1 %).

Snímek srdečního svalu levé komory byl rozdělen na několik oblastí zájmu (ROI) a pro každou oblast byla určena relativní hodnota zachyceného záření (celé číslo, vyšší hodnota = lepší prokrvení oblasti). Měření proběhlo dvakrát: v klidu (`_klid`) a po fyzické nebo farmakologické zátěži (`_zatez`). Každá oblast je tedy popsána dvojicí příznaků `oblastXX_klid` a `oblastXX_zatez`, kde `XX` je číslo oblasti.

Klinicky významné je především porovnání dvojice: oblast prokrvená v klidu, ale hůře po zátěži, ukazuje na zátěží vyvolanou ischemii; oblast se sníženou hodnotou v obou stavech ukazuje spíše na trvalé poškození (např. jizvu po infarktu).

| sloupec | typ | jednotka | popis |
|---|---|---|---|
| `oblastXX_klid` | celočíselný | relativní | Hodnota v oblasti `XX` v klidu. |
| `oblastXX_zatez` | celočíselný | relativní | Hodnota v oblasti `XX` po zátěži. |
| `target` | binární | – | Cílová proměnná: 1 = abnormální nález, 0 = normální nález. |

# Téma 3: Predikce úmrtí u pacientů se srdečním selháním

## Motivace

Srdeční selhání je stav, kdy srdce nedokáže přečerpávat dostatečné množství krve pro potřeby organismu. Patří mezi nejčastější důvody hospitalizace starších pacientů a i přes moderní léčbu má vysokou úmrtnost. Pro lékaře je klíčové včas rozpoznat pacienty s nejvyšším rizikem úmrtí, kterým je potřeba věnovat intenzivnější sledování, upravit medikaci nebo zvážit pokročilé terapeutické možnosti (např. implantabilní přístroje či transplantaci). Běžně dostupné klinické a laboratorní parametry, jako je ejekční frakce nebo hladina kreatininu v séru, přitom v sobě nesou cennou prognostickou informaci, kterou lze využít v predikčních modelech.

## Cíl

Provést co možná nejpřesnější predikci úmrtí pacienta se srdečním selháním během období sledování z dostupných klinických a laboratorních parametrů.

## Data

Trénovací soubor obsahuje 209 vzorků a 12 příznaků. Každý řádek odpovídá jednomu pacientovi se srdečním selháním, který byl hospitalizován a následně sledován. Část příznaků je binárních (komorbidity, pohlaví, kouření), ostatní jsou laboratorní a klinické hodnoty. Třídy jsou nevyvážené: třída 0 má 142 vzorků (67,9 %), třída 1 má 67 vzorků (32,1 %).

| sloupec | typ | jednotka | popis |
|---|---|---|---|
| `vek` | číselný | roky | Věk pacienta. Některé hodnoty nejsou celá čísla. |
| `pohlavi` | binární | – | Pohlaví pacienta: 0 = žena, 1 = muž. |
| `koureni` | binární | – | Pacient kouří: 0 = ne, 1 = ano. |
| `anemie` | binární | – | Anémie, tj. snížený počet červených krvinek nebo hladiny hemoglobinu: 0 = ne, 1 = ano. |
| `diabetes` | binární | – | Pacient má diabetes: 0 = ne, 1 = ano. |
| `hypertenze` | binární | – | Pacient má vysoký krevní tlak: 0 = ne, 1 = ano. |
| `ejekcni_frakce` | celočíselný | % | Ejekční frakce: podíl krve vypuzené z levé komory při každém stahu. |
| `cpk` | celočíselný | mcg/L | Hladina enzymu kreatinfosfokinázy (CPK) v krvi. Zvýšená hodnota může signalizovat poškození svalové tkáně, včetně srdečního svalu. |
| `kreatinin_serum` | číselný | mg/dL | Hladina kreatininu v séru, ukazatel funkce ledvin. |
| `sodik_serum` | celočíselný | mEq/L | Hladina sodíku v séru. |
| `trombocyty` | číselný | kiloplatelets/mL | Počet krevních destiček v krvi. |
| `doba_sledovani` | celočíselný | dny | Délka období, po které byl pacient sledován. |
| `target` | binární | – | Cílová proměnná: 1 = pacient během sledování zemřel, 0 = přežil. |
