import os
import pandas as pd

SEED_DIR = "seeds"

files = sorted([f for f in os.listdir(SEED_DIR) if f.endswith('.csv')])
print(f"\nFound {len(files)} CSV files in seeds\\ folder\n")

all_issues = []

for fname in files:
    path = os.path.join(SEED_DIR, fname)
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1')

    print("=" * 75)
    print(f"FILE : {fname}")
    print(f"ROWS : {len(df)}    COLS : {len(df.columns)}")
    print(f"{'COLUMN NAME':<42} {'DTYPE':<12} {'NULLS':<8} SAMPLE")
    print("-" * 75)

    for col in df.columns:
        dtype    = str(df[col].dtype)
        nulls    = int(df[col].isna().sum())
        nonempty = df[col].dropna()
        sample   = str(nonempty.iloc[0])[:28] if len(nonempty) > 0 else "ALL NULL"
        flag     = "  <-- ALL NULL" if nulls == len(df) else ""
        print(f"  {col:<40} {dtype:<12} {nulls:<8} {sample}{flag}")
        if nulls == len(df):
            all_issues.append(f"{fname}: '{col}' is completely empty")

    print()

print("=" * 75)
print("ISSUES FOUND:")
if all_issues:
    for issue in all_issues:
        print(f"  PROBLEM: {issue}")
else:
    print("  None. All columns have at least some data.")

print("\nDone.\n")
