import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Preprocess tha data
def preprocess_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df

# Random Forest Model
def run_random_forest(df):
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1_weighted')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    st.write(f"Best Parameters: {grid.best_params_}")

    y_pred = best_model.predict(X_test)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")

    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1_weighted')
    st.write(f"**Cross-validated F1 Score**: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

    importances = best_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    st.subheader("Top 10 Important Features")
    st.dataframe(importance_df.head(10))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title('Top 10 Important Features')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    st.pyplot(fig)

# SVM Model
def run_svm(df):
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    linear_model = SVC(kernel='linear', probability=True)
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    y_probs_linear = linear_model.predict_proba(X_test)[:, 1]

    rbf_model = SVC(kernel='rbf', probability=True)
    rbf_model.fit(X_train, y_train)
    y_pred_rbf = rbf_model.predict(X_test)
    y_probs_rbf = rbf_model.predict_proba(X_test)[:, 1]

    st.subheader("SVM - Linear Kernel")
    st.write("Accuracy:", accuracy_score(y_test, y_pred_linear))
    st.write("Precision:", precision_score(y_test, y_pred_linear, average='weighted'))
    st.write("Recall:", recall_score(y_test, y_pred_linear, average='weighted'))
    st.write("AUC:", roc_auc_score(y_test, y_probs_linear))
    
    st.markdown("**Classification Report:**")
    report_linear = classification_report(y_test, y_pred_linear, output_dict=True)
    st.dataframe(pd.DataFrame(report_linear).transpose())

    st.subheader("SVM - RBF Kernel")
    st.write("Accuracy:", accuracy_score(y_test, y_pred_rbf))
    st.write("Precision:", precision_score(y_test, y_pred_rbf, average='weighted'))
    st.write("Recall:", recall_score(y_test, y_pred_rbf, average='weighted'))
    st.write("AUC:", roc_auc_score(y_test, y_probs_rbf))
    
    st.markdown("**Classification Report:**")
    report_rbf = classification_report(y_test, y_pred_rbf, output_dict=True)
    st.dataframe(pd.DataFrame(report_rbf).transpose())

    # Decision Boundary Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    rbf_model.fit(X_pca, y)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = rbf_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
    ax.set_title('SVM Decision Boundary (RBF kernel in PCA space)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True)
    st.pyplot(fig)

# UI
st.title("ML Model: Random Forest & SVM")

uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Dataset Preview:")
    st.dataframe(df.head())

    df_processed = preprocess_data(df)
    st.write("Processed Data Preview:")
    st.dataframe(df_processed.head())

    model_choice = st.selectbox("Choose a Model to Run", ['Random Forest', 'SVM'])

    if st.button("Run Model"):
        if model_choice == 'Random Forest':
            run_random_forest(df_processed)
        else:
            run_svm(df_processed)
