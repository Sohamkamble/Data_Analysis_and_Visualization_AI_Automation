import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def ranking_analysis(clean_data):
    top_10_countries = clean_data.sort_values(by='Total', ascending=False).head(10)
    return top_10_countries[['Rank', 'Country', 'Gold', 'Silver', 'Bronze', 'Total']]

def correlation_analysis(clean_data):
    correlation_matrix = clean_data[['Gold', 'Silver', 'Bronze', 'Total']].corr()
    correlation_matrix.to_csv('correlation_analysis.csv') 
    return correlation_matrix

def country_wise_analysis(clean_data):
    country_summary = clean_data.groupby('Country').agg({
        'Gold': 'sum',
        'Silver': 'sum',
        'Bronze': 'sum',
        'Total': 'sum'
    }).reset_index()
    return country_summary

def linear_regression_analysis(clean_data):
    features = ['Gold', 'Silver', 'Bronze']
    target = 'Total'
    
    X = clean_data[features]
    y = clean_data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'Coefficients': model.coef_.tolist(), 
        'Intercept': model.intercept_,
        'MSE': mse,
        'R2 Score': r2
    }
    
    return results

def kmeans_clustering(clean_data, n_clusters=3):
    features = ['Gold', 'Silver', 'Bronze', 'Total']
    X = clean_data[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    clean_data['Cluster'] = clusters
    
    # Ensure the necessary columns are saved in the CSV
    kmeans_results = clean_data[['Country', 'Gold', 'Silver', 'Cluster']]
    kmeans_results.to_csv('kmeans_clustering.csv', index=False)
    
    return kmeans_results


def pca_analysis(clean_data, n_components=2):
    features = ['Gold', 'Silver', 'Bronze', 'Total']
    X = clean_data[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_clean_data = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    return pca_clean_data

def perform_general_analysis(df):
    results = {}
    results['ranking_analysis'] = ranking_analysis(df)
    results['correlation_analysis'] = correlation_analysis(df)
    results['country_wise_analysis'] = country_wise_analysis(df)
    results['linear_regression_analysis'] = linear_regression_analysis(df)
    results['kmeans_clustering'] = kmeans_clustering(df)
    results['pca_analysis'] = pca_analysis(df)
    
    print("Results:", results) 

    try:
        results['ranking_analysis'].to_csv('ranking_analysis.csv', index=False)
        print("Ranking analysis saved to ranking_analysis.csv") 
    except Exception as e:
        print("Error saving ranking_analysis:", e)
    
    try:
        results['correlation_analysis'].to_csv('correlation_analysis.csv')
        print("Correlation analysis saved to correlation_analysis.csv") 
    except Exception as e:
        print("Error saving correlation_analysis:", e)
    
    try:
        results['country_wise_analysis'].to_csv('country_wise_analysis.csv', index=False)
        print("Country-wise analysis saved to country_wise_analysis.csv")
    except Exception as e:
        print("Error saving country_wise_analysis:", e)
    
    try:
        with open('linear_regression_analysis.json', 'w') as f:
            json.dump(results['linear_regression_analysis'], f)
            print("Linear regression analysis saved to linear_regression_analysis.json")  
    except Exception as e:
        print("Error saving linear_regression_analysis:", e)
    
    try:
        results['kmeans_clustering'].to_csv('kmeans_clustering.csv', index=False)
        print("KMeans clustering saved to kmeans_clustering.csv") 
    except Exception as e:
        print("Error saving kmeans_clustering:", e)
    
    try:
        results['pca_analysis'].to_csv('pca_analysis.csv', index=False)
        print("PCA analysis saved to pca_analysis.csv") 
    except Exception as e:
        print("Error saving pca_analysis:", e)
    
    return results