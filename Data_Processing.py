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

def clean_data(df):
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
        print("Duplicates removed.")
    
    if df.isnull().sum().sum() > 0:
        missing_threshold = 0.2 * len(df)
        df.dropna(thresh=missing_threshold, inplace=True)
        df.fillna(df.mean(), inplace=True)
        print("Missing values handled.")
    
    for col in df.select_dtypes(include=['object']).columns:
        if pd.to_numeric(df[col], errors='coerce').notnull().all(): 
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Converted column '{col}' to numeric.")

    def handle_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        print(f"Checking for outliers in numeric columns: {numeric_cols}")
        for col in numeric_cols:
            if df[col].notnull().any():
                df = handle_outliers(df, col)
                print(f"Outliers handled in column '{col}'.")

    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"Encoding categorical columns: {categorical_cols}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print("Categorical variables encoded.")

    return df
