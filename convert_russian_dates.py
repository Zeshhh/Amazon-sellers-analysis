import pandas as pd

df = pd.read_csv(
    r'C:\Users\User\source\repos\PLG_SAAS_ANALYSIS\data\raw\data.csv',
    sep=';',
    low_memory=False
)

month_map = {
    'Январь': 'Jan', 'Февраль': 'Feb', 'Март': 'Mar', 'Апрель': 'Apr',
    'Май': 'May', 'Июнь': 'Jun', 'Июль': 'Jul', 'Август': 'Aug',
    'Сентябрь': 'Sep', 'Октябрь': 'Oct', 'Ноябрь': 'Nov', 'Декабрь': 'Dec'
}

datetime_cols = [
    col for col in df.columns
    if 'DATE' in col.upper()
    or col.upper().endswith('_AT')
    or col.upper().endswith('_LOGIN')
]

for col in datetime_cols:
    if col in df.columns and df[col].dtype == object:
        for ru_str, eng_str in month_map.items():
            df[col] = df[col].astype(str).str.replace(ru_str, eng_str, regex=False)
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].fillna(pd.Timestamp('1900-01-01'))

out_path = r'C:\Users\User\source\repos\PLG_SAAS_ANALYSIS\data\raw\data_dates_converted.csv'
df.to_csv(out_path, sep=';', index=False)

print(f'Converted {len(datetime_cols)} datetime columns across {len(df)} rows.')
print(f'Output saved to: {out_path}')