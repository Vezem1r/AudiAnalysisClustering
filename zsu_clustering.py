import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

file_path = "./audi.csv"
df = pd.read_csv(file_path)

# 1.1 Exploratory Data Analysis for assigned dataset.
print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDescribe values:")
print(df.describe())

# 1.2 Description of data and hypotheses

# Hypothesis 1: Price depends on the year of manufacture and engine size
# Hypothesis 2: Diesel cars have better fuel efficiency (mpg)
# Hypothesis 3: Cars with automatic transmission are more expensive

# 1.3 Visualizing the distribution of numeric features
num_cols = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
df[num_cols].hist(figsize=(12, 8), bins=15, edgecolor='black')
plt.suptitle("Distribution of Numerical Variables")
plt.show()

# Model distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='model', hue='model', order=df['model'].value_counts().index)
plt.title('Car Model Distribution')
plt.show()

# Transmission distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='transmission', hue='transmission')
plt.title('Transmission Type Distribution')
plt.show()

# FuelType distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='fuelType', hue='fuelType')
plt.title('Fuel Type Distribution')
plt.show()

# Correlation between numeric features
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Numeric Features')
plt.show()

# Visualization for Hypothesis 1: Price depends on the year of manufacture and engine size
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='year', y='price', hue='engineSize', palette='viridis', size='mileage', sizes=(50, 200))
plt.title('Price vs Year (colored by engineSize)')
plt.show()

# Visualization for Hypothesis 2: Diesel cars have better fuel efficiency (mpg)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='fuelType', y='mpg', hue='fuelType')
plt.title('Fuel Efficiency (mpg) by Fuel Type')
plt.ylabel('Miles Per Gallon (mpg)')
plt.xlabel('Fuel Type')
plt.show()

# Visualization for Hypothesis 3: Cars with automatic transmission are more expensive
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='transmission', y='price', hue='transmission')
plt.title('Car Price by Transmission Type')
plt.ylabel('Price')
plt.xlabel('Transmission Type')
plt.show()

#------------------------------------------------------------------------------------------

# 2.1 Clustering Project
num_features = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
cat_features = ['model', 'transmission', 'fuelType']

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(drop='first', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

X = preprocessor.fit_transform(df)

# Elbow Method
inertia = []
silhouette_scores = []

for k in range(2, 11):  # Testing k values from 2 to 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

plt.figure(figsize=(12, 6))

# Elbow method
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

# Silhouette coefficient
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
plt.title('Silhouette Coefficient')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Perform K-Means Clustering with Optimal k
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Analyze and Visualize Clusters
# 1. Cluster distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Cluster', data=df, hue='Cluster', palette='viridis', legend=False)
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# 2. Boxplot for price by cluster
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Cluster', y='price', hue='Cluster', palette='coolwarm', legend=False)
plt.title('Price Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Price')
plt.show()

# 3. Scatter plot for mileage vs price, colored by cluster
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='mileage', y='price', hue='Cluster', palette='viridis', s=50, legend=False)
plt.title('Mileage vs Price by Cluster')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.show()

# 4. Boxplot for engineSize by cluster
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Cluster', y='engineSize', hue='Cluster', palette='Set2', legend=False)
plt.title('Engine Size Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Engine Size')
plt.show()

# Print information about each cluster
for i in range(optimal_k):
    print(f"Cluster {i}:")
    cluster_data = df[df['Cluster'] == i]
    
    print(f"  Number of points in this cluster: {len(cluster_data)}")
    print("  Mean values (for numeric features):")
    for feature in num_features:
        mean_value = cluster_data[feature].mean()
        print(f"    {feature}: {mean_value:.2f}")
    
    print("  Most common categorical features:")
    for feature in cat_features:
        most_common = cluster_data[feature].mode()[0]
        print(f"    {feature}: {most_common}")
    
    print("-" * 40)

# Part 2-----------------------------------------------------------

agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['Agglomerative_Cluster'] = agg_clustering.fit_predict(X)

# Analyze and Visualize Clusters for Agglomerative Clustering
# 1. Cluster distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Agglomerative_Cluster', data=df, hue='Agglomerative_Cluster', palette='viridis', legend=False)
plt.title('Agglomerative Clustering - Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# 2. Boxplot for price by cluster
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Agglomerative_Cluster', y='price', hue='Agglomerative_Cluster', palette='coolwarm', legend=False)
plt.title('Agglomerative Clustering - Price Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Price')
plt.show()

# 3. Scatter plot for mileage vs price, colored by cluster
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='mileage', y='price', hue='Agglomerative_Cluster', palette='viridis', s=50, legend=False)
plt.title('Agglomerative Clustering - Mileage vs Price by Cluster')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.show()

# 4. Boxplot for engineSize by cluster
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Agglomerative_Cluster', y='engineSize', hue='Agglomerative_Cluster', palette='Set2', legend=False)
plt.title('Agglomerative Clustering - Engine Size Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Engine Size')
plt.show()

for i in range(4):
    print(f"Cluster {i}:")
    cluster_data = df[df['Agglomerative_Cluster'] == i]
    
    print(f"  Number of points in this cluster: {len(cluster_data)}")
    
    print("  Most common values for numeric features:")
    for feature in num_features:
        mean_value = cluster_data[feature].mean()
        print(f"    {feature}: {mean_value:.2f}")
    
    print("  Most common categorical features:")
    for feature in cat_features:
        most_common = cluster_data[feature].mode()[0]
        print(f"    {feature}: {most_common}")
    
    print("-" * 40)