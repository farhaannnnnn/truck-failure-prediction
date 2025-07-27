
# truck-failure-prediction
            Predictive Maintenance for Commercial Truck Fleets  
1. Project Goal & Business Problem
The primary goal of this project is to build a machine learning model to predict component failure in a fleet of heavy-duty Scania trucks. The specific component system analyzed is the Air Pressure System (APS), which is critical for braking and gear shifting.

By accurately predicting failures before they happen, fleet operators can move from a costly reactive maintenance strategy (fixing things after they break) to a cost-effective proactive maintenance strategy (scheduling repairs during planned downtime). This directly translates to increased vehicle uptime, reduced on-road breakdown costs, and improved safety.

2. Dataset Used
This project utilizes the "APS Failure at Scania Trucks" dataset, a well-known industrial dataset from the UCI Machine Learning Repository.

Link to Dataset: https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks

Description: The dataset consists of anonymized sensor readings from Scania trucks. A key characteristic and challenge of this dataset is its severe class imbalance, where positive cases (failures) are extremely rare compared to negative cases. This is a realistic reflection of real-world industrial data.

3. My Methodology & Workflow
My approach followed a structured data science workflow from data ingestion to model evaluation:

Data Ingestion & Cleaning:

Loaded the training and test datasets using Pandas, correctly identifying 'na' strings as missing values.

Performed an initial assessment which revealed a large number of missing values across many features.

To handle these missing values, I chose median imputation. This strategy is more robust to outliers than mean imputation, which is a common issue in sensor data.

Handling Class Imbalance:

The most critical challenge was the class imbalance (~98% negative vs. ~2% positive cases). Training a model on this data directly would result in a model that simply predicts "no failure" every time.

To solve this, I applied the SMOTE (Synthetic Minority Over-sampling Technique). It's crucial to note that SMOTE was applied only to the training data to prevent data leakage and ensure the test set remains a true, unseen representation of the real world.

Feature Scaling:

After resampling, I applied StandardScaler to all features. This standardizes the data to have a mean of 0 and a standard deviation of 1, ensuring that all features contribute equally to the model's learning process.

Model Training & Selection:

I trained three different classification models to compare their performance on this complex task:

Logistic Regression: A great, interpretable baseline model.

Random Forest Classifier: A powerful ensemble method that can capture non-linear relationships.

XGBoost Classifier: A state-of-the-art gradient boosting algorithm known for its high performance and speed.

Evaluation Strategy:

Accuracy is a misleading metric for imbalanced problems. Instead, my primary evaluation metric was the F1-Score, which provides a harmonic mean of Precision and Recall. This is ideal for finding a balance between minimizing false alarms (unnecessary maintenance) and catching actual failures.

I also heavily analyzed the Confusion Matrix for the best model to understand its trade-offs, particularly the number of False Negatives (missed failures), which carry the highest business cost.

4. Results & Conclusion
The XGBoost Classifier demonstrated the best performance, achieving the highest F1-Score. It proved to be the most effective at identifying the complex patterns in the sensor data to predict APS failures.

The final model successfully identifies a significant portion of failing trucks, allowing a fleet manager to take preventative action. This directly validates the business case for using predictive analytics to reduce operational costs and improve fleet reliability.

5. Technologies Used
Language: Python

Core Libraries: Pandas, NumPy, Scikit-learn, Imbalanced-learn

Modeling: XGBoost

Visualization: Matplotlib, Seaborn

Environment: Jupyter Notebook (or any Python IDE)

6. How to Run This Project
Clone the repository:

git clone https://github.com/your-username/truck-failure-prediction.git
cd truck-failure-prediction

Create the folder structure:
Ensure your project has a data/ subfolder.

Download the data:
Place aps_failure_training_set.csv and aps_failure_test_set.csv into the data/ folder.

Install dependencies:

pip install -r requirements.txt

(You will need to create a requirements.txt file with libraries like pandas, scikit-learn, etc.)

Run the script/notebook.

