# 🎬 Netflix Content Clustering with K-Means & UMAP

## 📘 Project Overview

This project applies **unsupervised learning** techniques to cluster
Netflix content based on textual and numerical metadata.\
By combining **TF-IDF text embeddings**, **numerical attributes**, and
**dimensionality reduction (UMAP)**, the model reveals hidden patterns
and content groupings in the Netflix catalog.

------------------------------------------------------------------------

## 🚀 Objectives

-   Perform **text preprocessing** and feature extraction from Netflix
    titles, descriptions, and genres.\
-   Apply **clustering algorithms** (K-Means, GMM, DBSCAN) and evaluate
    them using multiple metrics.\
-   Use **UMAP** for 2D visualization to validate cluster separability.\
-   Save and load the final **K-Means model** using `joblib` for
    deployment readiness.

------------------------------------------------------------------------

## 🧠 Workflow Summary

### 1️⃣ Data Preprocessing

-   Handled missing values and cleaned text columns.\
-   Extracted duration, country, and rating features.\
-   Converted categorical variables to numerical encodings.

### 2️⃣ Feature Engineering

-   Applied **TF-IDF vectorization** on titles and descriptions.\
-   Combined text embeddings with numerical features.\
-   Reduced high-dimensional data using **Truncated SVD** before
    clustering.

### 3️⃣ Model Training & Evaluation

-   Trained multiple clustering models: **K-Means**, **GMM**,
    **DBSCAN**.\
-   Evaluated using:
    -   Silhouette Score\
    -   Calinski--Harabasz Index\
    -   Davies--Bouldin Score\
    -   BIC/AIC (for GMM)
-   **K-Means** was selected as the final model for its interpretability
    and stability.

### 4️⃣ Visualization

-   Used **UMAP** to project high-dimensional feature space into 2D.\
-   Plotted clusters to confirm separability and thematic coherence.

------------------------------------------------------------------------

## 🧩 Key Results

-   **K-Means** achieved the best clustering performance and most
    meaningful segmentation.\
-   **UMAP visualization** showed clearly distinct cluster boundaries.\
-   Each cluster corresponded to recognizable genres or content themes.

------------------------------------------------------------------------

## 💡 Business Applications

-   🎯 Personalized Recommendation Systems\
-   🎬 Automated Genre Tagging / Metadata Enrichment\
-   🌍 Content Strategy & Market Expansion Insights

------------------------------------------------------------------------

## 📈 Conclusion

The final model successfully grouped Netflix titles into meaningful
clusters.\
Using **UMAP**, these clusters were visually validated to be distinct
and cohesive.\
The **K-Means model**, saved in `.pkl` format, is fully deployable on
**Streamlit**, **FastAPI**, or **Azure ML** for real-time
recommendations.

------------------------------------------------------------------------

## 🧑‍💻 Author

**Ganpati Nath**\
Deep Learning & Data Science Enthusiast

------------------------------------------------------------------------

⭐ **If you found this helpful, consider starring the repository!**
