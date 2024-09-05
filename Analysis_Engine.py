import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import train_test_split

def descriptive_statistics(df):
    return df.describe(include='all')

def correlation_analysis(df):
    """
    Calculate correlation matrix of the DataFrame.
    """
    return df.corr()

def plot_distributions(df):
    """
    Plot distributions for numerical features and count plots for categorical features.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    distributions = {}
    
    if len(numeric_cols) > 0:
        distributions['numeric'] = df[numeric_cols].describe()

    if len(categorical_cols) > 0:
        distributions['categorical'] = df[categorical_cols].value_counts()

    return distributions

def pairwise_relationships(df):
    """
    Get pairwise relationships between numerical features.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        return df[numeric_cols].corr()
    return pd.DataFrame()

def trend_analysis(df, time_col, value_col):
    """
    Analyze trends over time if applicable.
    """
    if time_col in df.columns and value_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df.set_index(time_col, inplace=True)
        return df[[value_col]].resample('M').mean()
    return pd.DataFrame()

def pca_analysis(df, n_components=2):
    """
    Perform PCA to reduce dimensionality and return principal components.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        return pca_df
    return pd.DataFrame()

def kmeans_clustering(df, n_clusters=3):
    """
    Perform K-Means clustering and return DataFrame with cluster labels.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        df['Cluster'] = clusters
        return df
    return df

def linear_regression_analysis(df, target_col):
    """
    Perform linear regression analysis and return DataFrame with predicted values.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if target_col in numeric_cols:
        features = numeric_cols.drop(target_col)
        X = df[features]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        df_test = X_test.copy()
        df_test[target_col] = y_test
        df_test['Predicted'] = y_pred
        
        return df_test
    return df

def hypothesis_testing(df, col1, col2):
    """
    Perform hypothesis test (e.g., t-test) between two columns.
    """
    if col1 in df.columns and col2 in df.columns:
        stat, p_value = stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
        return pd.DataFrame({'T-statistic': [stat], 'P-value': [p_value]})
    return pd.DataFrame()

def perform_general_analysis(df, time_col=None, value_col=None, target_col=None, col1=None, col2=None):
    """
    Perform general analysis on the DataFrame and return a DataFrame with results.
    """
    results = {}
    results['descriptive_statistics'] = descriptive_statistics(df)
    results['correlation_matrix'] = correlation_analysis(df)
    results['distributions'] = plot_distributions(df)
    results['pairwise_relationships'] = pairwise_relationships(df)
    
    if time_col and value_col:
        results['trend_analysis'] = trend_analysis(df, time_col, value_col)
    
    results['pca_analysis'] = pca_analysis(df)
    results['kmeans_clustering'] = kmeans_clustering(df)
    
    if target_col:
        results['linear_regression_analysis'] = linear_regression_analysis(df, target_col)
    
    if col1 and col2:
        results['hypothesis_testing'] = hypothesis_testing(df, col1, col2)
    
    return results
