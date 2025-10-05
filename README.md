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

## ⚙️ Technologies Used

  Category          Tools / Libraries
  ----------------- ---------------------------------
  Language          Python 3.10
  Data Handling     pandas, numpy
  Text Processing   scikit-learn (TF-IDF, SVD)
  Clustering        K-Means, GMM, DBSCAN
  Visualization     matplotlib, seaborn, umap-learn
  Model Saving      joblib

------------------------------------------------------------------------

## 🧾 Model Saving & Prediction Example

``` python
import joblib
loaded_model = joblib.load("best_kmeans_model.pkl")

def predict_new(text, numeric_features):
    X_text_new = tfidf.transform([text])
    X_num_new = scaler.transform([numeric_features])
    X_text_new_reduced = svd.transform(X_text_new)
    X_combined_new = hstack([X_text_new_reduced, csr_matrix(X_num_new)]).toarray()
    return loaded_model.predict(X_combined_new)[0]

print(predict_new("A romantic comedy of two strangers", [90, 5, 100]))
```

------------------------------------------------------------------------

## 🗂️ Repository Structure

    📁 netflix-content-clustering/
    ├── 📄 README.md
    ├── 📄 netflix_dataset.csv
    ├── 📄 clustering_notebook.ipynb
    ├── 📄 best_kmeans_model.pkl
    ├── 📊 cluster_visualization_umap.png
    └── 📂 src/
        ├── data_preprocessing.py
        ├── feature_engineering.py
        ├── model_training.py
        ├── visualization.py

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
Deep Learning & Data Science Enthusiast\
📧 \[LinkedIn Profile / GitHub Portfolio - Add yours here\]

------------------------------------------------------------------------

⭐ **If you found this helpful, consider starring the repository!**
