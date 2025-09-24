Experiment 7 – Clustering Human Activity Recognition Data
Objective

This assignment implements and analyzes clustering algorithms on the Human Activity Recognition (HAR) dataset. The goal is to cluster smartphone sensor data into human activities and compare the performance of different clustering techniques.

Dataset

Source: UCI HAR Dataset

Details:

30 volunteers (ages 19–48).

6 activities: WALKING, WALKING UPSTAIRS, WALKING DOWNSTAIRS, SITTING, STANDING, LAYING.

Data collected using accelerometer & gyroscope at 50 Hz.

Preprocessed into 2.56-second windows (128 readings per window).

Steps Performed
1. Preprocessing

Download and extract dataset.

Handle missing values (none found).

Standardize features using StandardScaler.

Map activity labels (1–6 → activity names).

2. Exploratory Data Analysis (EDA)

Distribution plots of selected features.

Correlation heatmap of features.

PCA applied for 2D visualization of activities.

3. K-Means Clustering

Applied K-Means for k=2–8.

Plotted Elbow Curve (k vs WCSS) and Silhouette Curve (k vs Silhouette Score).

Selected k=6 as optimal.

Evaluated clusters using internal and external metrics.

Generated confusion matrix vs. ground truth.

4. DBSCAN

Performed density-based clustering with various eps and minPts values.

Reported number of clusters and noise points detected.

Visualized DBSCAN clusters in PCA space.

5. Hierarchical Agglomerative Clustering (HAC)

Reduced dimensions with PCA (50D).

Applied HAC with Ward linkage.

Visualized clusters in 2D PCA space.

Plotted dendrogram (sample of 500 points).

6. Evaluation Metrics

For each clustering algorithm, the following were computed:

Internal Metrics: Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index.

External Metrics: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI).

Results compared across algorithms using bar plots.

7. Observations

K-Means: Best alignment with true labels, optimal at k=6.

DBSCAN: Detected noise/outliers, but sensitive to parameter choices.

HAC: Provided hierarchical structure, dendrogram useful for analysis.

Overall: K-Means gave the most stable and meaningful clusters.

Output Generated

EDA plots (feature distribution, correlation).

PCA scatter plots (ground truth and clusters).

Elbow and Silhouette curves.

Confusion matrix heatmap.

DBSCAN cluster visualizations.

HAC dendrogram.

Metric comparison bar chart.
