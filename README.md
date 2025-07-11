# Machine Learning Model Deployment: Random Forest & SVM (Streamlit App)

This project implements two machine learning models — **Random Forest Classifier** and **Support Vector Machine (SVM)** — to perform classification on any uploaded CSV dataset. The app is built using **Streamlit** for an interactive user interface.

---

## Demo

https://github.com/user-attachments/assets/178b1613-6fc8-4e2c-b9a2-cc3aed9e2195

---

## Features

-  Upload any dataset in CSV format.
-  Preprocesses the data automatically (handling missing values, encoding categoricals).
-  Train and evaluate:
   - **Random Forest Classifier** (with hyperparameter tuning)
   - **SVM Classifier** (Linear and RBF kernels)
-  Shows performance metrics:
   - Accuracy, Precision, Recall, F1-Score, AUC
   - Classification report 
-  Visualizes:
   - Feature Importance (for Random Forest)
   - Decision Boundary (for SVM using PCA)

---

## Libraries Used

- `streamlit`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn` (`RandomForestClassifier`, `SVC`, `GridSearchCV`, etc.)

---

## How to Run the App

### 1. Clone the Repository
```bash
git clone https://github.com/Kiranwaqar/RandomForest-SVM.git
cd Level_3_Task_1,2
```
2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # on Unix or Mac
venv\Scripts\activate     # on Windows
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Run the App
```bash
streamlit run app.py
```
## File Structure
```bash
├── app.py                  # Main Streamlit application
├── requirements.txt        # All dependencies
├── README.md               # You're reading it!
```


