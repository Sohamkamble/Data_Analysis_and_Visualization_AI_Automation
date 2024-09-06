import pandas as pd
import json

def load_data(file_path):
    """
    Load data from a given file path. The function supports CSV, XLSX, and JSON formats.
    
    Args:
        file_path (str): The path to the file containing the dataset.
        
    Returns:
        pd.DataFrame: A pandas DataFrame with the loaded data.
    
    Raises:
        ValueError: If the file format is not supported.
    """
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
    """
    Cleans the Olympics 2024 dataset by performing the following steps:
    
    1. Load the dataset using load_data function.
    2. Remove duplicate rows if any.
    3. Handle missing values by dropping rows with excessive missing data, 
       and filling others with the mean value.
    4. Convert numeric-like strings in object columns to proper numeric types.
    5. Remove 'Country Code' column if it exists.
    
    Returns:
        pd.DataFrame: The cleaned pandas DataFrame.
    """
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

def test_load_data():
    """
    Unit test for the load_data function. Ensures that data loading works for different formats.
    """
    df_csv = load_data("test_data.csv")
    assert isinstance(df_csv, pd.DataFrame), "CSV loading failed"
    
    df_xlsx = load_data("test_data.xlsx")
    assert isinstance(df_xlsx, pd.DataFrame), "XLSX loading failed"
    
    df_json = load_data("test_data.json")
    assert isinstance(df_json, pd.DataFrame), "JSON loading failed"
    
    try:
        load_data("test_data.txt")
    except ValueError as e:
        assert str(e) == "Unsupported file type. Use 'csv', 'xlsx', or 'json'.", "File type error handling failed"

def test_clean_data():
    """
    Unit test for the clean_data function. Ensures data cleaning logic is applied correctly.
    """
    df_clean = clean_data()
    
    assert df_clean.duplicated().sum() == 0, "Duplicates were not removed"
    
    assert df_clean.isnull().sum().sum() == 0, "Missing values were not handled"
    
    assert 'Country Code' not in df_clean.columns, "'Country Code' column was not removed"

if __name__ == "__main__":
    test_load_data()
    test_clean_data()
    print("All tests passed!")
