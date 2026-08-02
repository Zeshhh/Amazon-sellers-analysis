import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer

df = pd.read_csv('data/raw/data.csv', sep=';', low_memory=False)

for col in df.columns:
    if df[col].dtype == 'object':
        try:
            converted = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
            if converted.notna().sum() / len(df) > 0.5:
                df[col] = converted
        except Exception:
            pass

missing_flags = pd.DataFrame()
for col in df.columns:
    if df[col].isnull().sum() > 0:
        missing_flags[f'{col}_was_missing'] = df[col].isnull().astype(int)

rus_months = {
    'Январь': 'January', 'Февраль': 'February', 'Март': 'March',
    'Апрель': 'April', 'Май': 'May', 'Июнь': 'June',
    'Июль': 'July', 'Август': 'August', 'Сентябрь': 'September',
    'Октябрь': 'October', 'Ноябрь': 'November', 'Декабрь': 'December'
}

def parse_russian_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    s = str(date_str)
    for rus, eng in rus_months.items():
        s = s.replace(rus, eng)
    return pd.to_datetime(s, errors='coerce')

date_cols = [c for c in df.columns if any(x in c.lower() for x in
            ['date', 'time', 'launched', 'activated', 'created', 'login',
             'meeting', 'shipped', 'pick'])]

for col in date_cols:
    df[col] = df[col].apply(parse_russian_date)
    df[col] = df[col].fillna(pd.Timestamp('2099-01-01'))

num_cols = df.select_dtypes(include='number').columns.tolist()
count_like = [
    'EMAILS_L28', 'CALLS_15_MIN_L28', 'CALLS_2_MIN_L28',
    'DEMOS_15_MIN_L28', 'DEMOS_2_MIN_L28',
    'RE_EMAILS', 'RE_EMAILS_BEFORE_SIGNUP', 'RE_EMAILS_BEFORE_LAUNCH',
    'NO_OF_USERS', 'NO_OF_ACTIVE_USERS_L_28D',
    'NO_OF_CONNECTED_CHANNELS', 'NO_OF_CONNECTED_CHANNELS_EXCL_DIRECT',
    'NO_OF_CONNECTED_CHANNELS_EXCL_AMAZON', 'NO_OF_WAREHOUSES',
    'FIRST_STOCK_TAKE', 'SHIPMENTS_UPS'
]
knn_like = [c for c in num_cols if c not in count_like and c != 'COMPANY_ID']

for col in count_like:
    if col in df.columns:
        df[col] = df[col].fillna(0)

knn_like = [c for c in knn_like if df[c].isna().sum() > 0 and df[c].isna().sum() < len(df)]
if knn_like:
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df[knn_like] = imputer.fit_transform(df[knn_like])

cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')

df = pd.concat([df, missing_flags], axis=1)

print(f'Final shape: {df.shape}')
print(f'Remaining missing: {df.isnull().sum().sum()}')
print(f'Missing flag columns: {missing_flags.shape[1]}')
print(f'KNN imputed: {knn_like}')
print(f'Zero-filled (counts): {count_like}')
