Customer Purchase Behavior Analysis
This project analyzes customer purchase behavior using machine learning techniques. It involves data preprocessing, exploratory data analysis (EDA), model building, evaluation, and comparison to predict customer purchasing patterns.

Project Overview
The project aims to:

Understand patterns in customer behavior.
Build machine learning models to predict purchases based on demographic and behavioral features.
Compare different models to identify the best-performing algorithm.
Visualize results for actionable insights.
Key models used:

Logistic Regression
Decision Tree
Random Forest
XGBoost
Features of the Project
1. Exploratory Data Analysis (EDA)
Visualized feature correlations using heatmaps.
Explored customer segments using PCA and clustering techniques.
Analyzed distributions of key features like age, gender, and clusters.
2. Data Preprocessing
Handled missing values and performed label encoding and one-hot encoding for categorical features.
Balanced the dataset using SMOTE to address class imbalance.
3. Machine Learning Models
Built and evaluated multiple machine learning models:
Logistic Regression
Decision Tree
Random Forest
XGBoost
Compared model performance using accuracy, precision, recall, F1-score, and AUC-ROC.
4. Hyperparameter Tuning
Optimized models using GridSearchCV for better performance.
5. Feature Importance
Visualized the importance of features in influencing customer purchase behavior for Random Forest and XGBoost.
6. Performance Evaluation
Plotted ROC curves to assess model performance.
Compared models with cross-validation scores.
Setup Instructions
Dependencies
Ensure you have Python 3.x installed along with the following libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
plotly
Installation
Clone the repository:
bash
Copy code
git clone [https://github.com/username/repository-name.git](https://github.com/manzar79md/Customer-Purchase-Behavior-Analysis)
Navigate to the project folder:
bash
Copy code
cd repository-name
Install dependencies:
bash
Copy code
pip install -r requirements.txt
How to Run
Open the Jupyter Notebook or Google Colab.
Upload the dataset to your environment or update the dataset path in the notebook:
python
Copy code
file_path = "/content/drive/MyDrive/L&T Data Engineering in python/read.csv/Customer-Purchase-Behaviour-Analysis.csv"
Run each cell sequentially to execute the analysis.
Results
The best-performing model was identified using metrics such as accuracy, F1-score, and AUC-ROC.
Key features influencing customer behavior include demographic factors and purchase history.
Insights from the clustering analysis revealed distinct customer segments.
Dataset
The dataset includes features like age, gender, purchase amount, and product category.
For privacy reasons, the dataset is not included in this repository. Replace the file path with your dataset in the notebook.
Visualization Highlights
Heatmaps for feature correlation.
Scatter plots for PCA and customer clustering.
Feature importance visualizations for Random Forest and XGBoost.
ROC curves for model evaluation.
Future Improvements
Deploy the best-performing model using Streamlit for real-time predictions.
Experiment with additional machine learning algorithms or deep learning models.
Perform time-series analysis for seasonal patterns.
Acknowledgments
This project was part of the L&T EduTech Data Engineering Internship, where various machine learning and data engineering techniques were applied.

License
This project is licensed under the MIT License. See the LICENSE file for details.
