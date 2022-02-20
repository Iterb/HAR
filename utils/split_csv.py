import pandas as pd

full = pd.read_csv(
    "/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv", index_col=[0]
)
print(full)
col1 = [str(dx) for dx in range(26)]
col2 = [str(dx) for dx in range(26, 52)]
col3 = [str(dx) for dx in range(52, 77)]
print(col1)
print(col2)
print((full.columns))
df1 = full[col1]
df2 = full[col2]
df3 = full[col3]
print(df1, df2)
df1.to_csv("/media/sebastian/STORAGE_HDD/data/normalized_25P_short_pera.csv")
df2.to_csv("/media/sebastian/STORAGE_HDD/data/normalized_25P_short_perb.csv")
df3.to_csv("/media/sebastian/STORAGE_HDD/data/normalized_25P_short_dist.csv")
