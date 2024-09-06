import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os

def plot_ranking_analysis(ranking_df):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Total', y='Country', data=ranking_df, palette='viridis')
    plt.title('Top 10 Countries by Total Medals')
    plt.xlabel('Total Medals')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig('ranking_analysis_plot.png')
    plt.close()

def plot_correlation_analysis(correlation_df):
    correlation_df = correlation_df.set_index(correlation_df.columns[0])  
    correlation_df = correlation_df.astype(float) 
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_analysis_plot.png')
    plt.close()

def plot_country_wise_analysis(country_summary_df):
    plt.figure(figsize=(14, 10))
    country_summary_df = country_summary_df.sort_values(by='Total', ascending=False)
    sns.barplot(x='Country', y='Total', data=country_summary_df, palette='magma')
    plt.title('Total Medals by Country')
    plt.xlabel('Country')
    plt.ylabel('Total Medals')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('country_wise_analysis_plot.png')
    plt.close()

def plot_kmeans_clustering(kmeans_df):
    required_columns = ['Gold', 'Silver', 'Cluster']
    if not all(col in kmeans_df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Gold', y='Silver', hue='Cluster', data=kmeans_df, palette='Set2', s=100)
    plt.title('K-Means Clustering of Olympic Medals')
    plt.xlabel('Gold Medals')
    plt.ylabel('Silver Medals')
    plt.legend(title='Cluster', loc='upper right')
    plt.tight_layout()
    plt.savefig('kmeans_clustering_plot.png')
    plt.close()

def plot_pca_analysis(pca_df):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, palette='viridis')
    plt.title('PCA Analysis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig('pca_analysis_plot.png')
    plt.close()

def create_pdf_report():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Comprehensive Medal Analysis Report", ln=True, align='C')
    pdf.ln(10)  
    
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=(
        "This report provides a comprehensive analysis of medal data, including visualizations and summaries of various metrics. "
        "The following sections cover ranking analysis, correlation analysis, country-wise analysis, K-Means clustering, and PCA analysis."
    ))
    pdf.ln(10)

    sections = [
        ("Ranking Analysis", "ranking_analysis_plot.png", 
         "The ranking analysis bar plot illustrates the top 10 countries by total medals won. This visualization highlights the leading countries in the event, providing insights into the overall performance across nations. The plot clearly shows which countries excel in terms of medal counts, demonstrating their competitive strength and consistent performance."
        ),
        ("Correlation Analysis", "correlation_analysis_plot.png", 
         "The correlation matrix heatmap reveals the relationships between various medal types and total medals. Strong correlations indicate how different types of medals are related, which can provide insight into the performance trends of countries. For example, a high correlation between gold and total medals suggests that countries with more gold medals generally win more medals overall."
        ),
        ("Country-Wise Analysis", "country_wise_analysis_plot.png", 
         "The country-wise analysis bar plot displays the total number of medals won by each country. This graph provides a detailed view of individual country performances, showcasing the distribution of medals. It highlights the countries with the highest medal counts and offers insights into their performance compared to other nations."
        ),
        ("K-Means Clustering", "kmeans_clustering_plot.png", 
         "The K-Means clustering scatter plot groups countries based on their medal counts, classifying them into clusters. This analysis identifies patterns in performance by grouping countries with similar medal profiles. Clusters reveal trends such as top performers excelling across all medal types, while other clusters might show countries that excel in specific medal types."
        ),
        ("PCA Analysis", "pca_analysis_plot.png", 
         "The PCA (Principal Component Analysis) scatter plot visualizes the distribution of data along the principal components. This analysis helps in understanding the variance and structure of the data by projecting it into a lower-dimensional space. The plot highlights how different data points are distributed and clustered based on their principal components."
        )
    ]
    
    for title, plot_file, summary_text in sections:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=title, ln=True, align='L')
        pdf.ln(10)  
        
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=summary_text)
        pdf.ln(10)
        
        if os.path.exists(plot_file):
            pdf.image(plot_file, x=10, y=pdf.get_y(), w=180)
            pdf.ln(10) 
        else:
            pdf.multi_cell(0, 10, txt=f"Plot file {plot_file} not found.")

    pdf.output("medal_analysis_report.pdf")

if __name__ == "__main__":
    create_pdf_report()