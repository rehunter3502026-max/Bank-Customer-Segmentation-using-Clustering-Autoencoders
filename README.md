# Bank-Customer-Segmentation-using-Clustering-Autoencoders

This project performs customer segmentation for a bank’s credit card dataset using:

K-Means Clustering

PCA (Principal Component Analysis)

Deep Learning Autoencoder for dimensionality reduction

The objective is to identify distinct customer groups based on spending behavior, credit usage, and payment patterns to support targeted marketing and risk analysis.

🚀 Project Goals

Understand customer spending behavior

Identify meaningful customer segments

Compare traditional clustering vs deep learning–based dimensionality reduction

Visualize clusters for business interpretation

🛠️ Tech Stack

Python

Pandas & NumPy

Seaborn & Matplotlib

Scikit-learn

TensorFlow / Keras

📊 Workflow
1️⃣ Data Preprocessing

Loaded credit card customer dataset

Handled missing values (MINIMUM_PAYMENTS, CREDIT_LIMIT)

Removed duplicate entries

Dropped non-numeric identifier (CUST_ID)

Standardized features using StandardScaler

2️⃣ Exploratory Data Analysis (EDA)

Distribution plots (KDE + Histograms)

Correlation heatmap

Observed strong relationships between:

Purchases & Installment Purchases

Purchases & Credit Limit

Purchase Frequency variables

3️⃣ K-Means Clustering

Used Elbow Method to determine optimal clusters

Selected 8 clusters

Analyzed cluster centers (inverse scaled for interpretation)

Identified Customer Types:

Transactors → Low balance, low cash advance, higher full payments

Revolvers → High balance, frequent cash advances, low full payments

VIP / Prime Customers → High credit limit, strong repayment behavior

Low Tenure Customers → Short relationship duration

4️⃣ PCA Visualization

Reduced dimensions to 2 principal components

Visualized cluster separation

5️⃣ Deep Learning Autoencoder

Built fully connected autoencoder

Reduced 17 features → 10 latent features

Trained using MSE loss

Performed clustering on encoded features

Compared:

Traditional K-Means inertia

Autoencoder + K-Means inertia

📈 Key Insights

Customer behavior varies significantly across balance usage and payment habits

Autoencoder-based dimensionality reduction improves clustering compactness

Segmentation can help in:

Targeted marketing campaigns

Credit risk profiling

Customer retention strategies

▶️ How to Run

Open notebook in Google Colab

Mount Google Drive

Place Marketing_data.csv in specified directory

Run all cells

💡 Business Applications

Personalized credit limit adjustments

Identifying high-value customers

Risk-based pricing models

Cross-selling financial products
