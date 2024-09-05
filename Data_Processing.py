import pandas as pd
import json
from fpdf import FPDF

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
    
    return df

def generate_pdf(df, output_path):
    """
    Generate a PDF report of the DataFrame.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Cleaned Data Report", ln=True, align='C')

    # Column names
    pdf.set_font("Arial", size=10)
    col_width = pdf.get_string_width(" | ".join(df.columns)) + 10
    pdf.cell(col_width, 10, txt=" | ".join(df.columns), ln=True)

    # Data rows
    for index, row in df.iterrows():
        row_text = ' | '.join(str(x) for x in row.values)
        pdf.cell(col_width, 10, txt=row_text, ln=True)

    pdf.output(output_path)

if __name__ == '__main__':
    file_path = 'olympics2024.xlsx'  # Update with your file path
    cleaned_df = clean_data(load_data(file_path))
    
    # Generate PDF of the cleaned data
    pdf_output_path = 'cleaned_data_report.pdf'
    generate_pdf(cleaned_df, pdf_output_path)
    print(f"PDF report generated: {pdf_output_path}")