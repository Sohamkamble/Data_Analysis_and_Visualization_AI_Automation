import pandas as pd
import json

def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        with open(file_path) as f:
            return pd.DataFrame(json.load(f))
    else:
        raise ValueError("Unsupported file type. Use 'csv', 'xlsx', or 'json'.")

def clean_data():
    file_path = "olympics2024.xlsx"
    df = load_data(file_path)
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
    
    if df.isnull().sum().sum() > 0:
        missing_threshold = 0.2 * len(df)
        df.dropna(thresh=missing_threshold, inplace=True)
        df.fillna(df.mean(), inplace=True)
    
    for col in df.select_dtypes(include=['object']).columns:
        if pd.to_numeric(df[col], errors='coerce').notnull().all(): 
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Country Code' in df.columns:
        df.drop(columns=['Country Code'], inplace=True)
    
    return df

print(clean_data())