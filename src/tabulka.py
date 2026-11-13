import pandas as pd

# 1. Načítanie súboru do DataFrame
df = pd.read_csv(r'C:\UIM_Projekt\BBD-ocni_nalez\data\train.csv')

# 2. Zobrazenie prvých 5 riadkov (aby si videl štruktúru stĺpcov)
print(df.head())

# 3. Zobrazenie zoznamu všetkých stĺpcov
print(df.columns)

# 4. Práca s konkrétnym stĺpcom (napríklad stĺpec 'kvalita')
print(df['kvalita'])