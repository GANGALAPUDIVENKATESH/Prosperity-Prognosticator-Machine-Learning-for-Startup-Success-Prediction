# Prosperity Prognosticator: Machine Learning for Startup Success Prediction


**📌 Project Description**

**Prosperity Prognosticator**:

Machine Learning for Startup Success Prediction is a data-driven system that predicts whether a startup will be successful (acquired/survived) or failed (closed) based on historical startup data.
The project analyzes funding patterns, team strength, market size, operational duration, and growth indicators using supervised machine learning algorithms. The final trained model is deployed using a Flask web application, allowing users to input startup details and instantly receive a success prediction.

**This solution supports:**
Investors → smarter investment decisions
Entrepreneurs → better business planning
Policy makers → data-backed startup support strategies

**🧠 Machine Learning Algorithms Used**
The following supervised learning algorithms were explored during model building and evaluation:

**1️⃣ Random Forest Classifier (Final Model)**
Ensemble algorithm using multiple decision trees
Reduces overfitting and improves generalization
Best performing model in this project

**2️⃣ Decision Tree Classifier**
Rule-based model for interpretability
Used for comparison with ensemble methods


**3️⃣ Support Vector Machine (SVM)**
Effective for classification with clear margins
Tested with scaled features


**4️⃣ Logistic Regression**
Baseline linear classification model
Used to compare performance against advanced models
✔ Hyperparameter tuning was applied using GridSearchCV to optimize model performance.


**📊 Evaluation Metrics Used**
To evaluate and compare model performance, the following metrics were used:
🔹 Accuracy
Measures overall correctness of predictions
Helps identify how often the model predicts correctly
🔹 Precision
Indicates how many predicted successes were actually successful
Important for investor decision-making
🔹 Recall
Measures how well the model identifies actual successful startups
Important to avoid missing high-potential startups
🔹 F1-Score
Harmonic mean of precision and recall
Balances false positives and false negatives
🔹 Confusion Matrix
Shows True Positives, True Negatives, False Positives, False Negatives
Helps understand classification errors
🔹 ROC-AUC (optional/analytical)
Measures class separation capability of the model
📈 Accuracy Achieved in the Project

**Dataset	Accuracy
Training Accuracy	~100%
Testing Accuracy	~80%**

**📌 Interpretation**
High training accuracy indicates strong learning capability
Slightly lower test accuracy indicates mild overfitting, which is common in tree-based models
GridSearchCV and feature scaling helped improve generalization
The final Random Forest model achieved approximately 80% accuracy on unseen data, which is strong for real-world startup prediction problems.
🔍 Features Used for Training
The model was trained using the following startup features:
Funding rounds
Total funding amount
Market size indicator
Team size
Years active
Revenue growth indicator
Target variable:
success →
1 = Successful / Acquired
0 = Failed / Closed


**📊 Exploratory Data Analysis (EDA**)
EDA was performed to understand the dataset before modeling:
Missing value analysis
Descriptive statistics
Correlation heatmap
Feature distribution analysis


**Key Insights:**
Funding and market size show strong correlation with success
Revenue growth and team size are important predictors
No severe multicollinearity after preprocessing


**🧰 Libraries & Tools Used**
🔹 Python Libraries
pandas → Data manipulation and preprocessing
numpy → Numerical computations
matplotlib → Data visualization
seaborn → Statistical visualizations
🔹 Machine Learning
scikit-learn
RandomForestClassifier
DecisionTreeClassifier
SVM
LogisticRegression
GridSearchCV
StandardScaler
Evaluation metrics
🔹 Model Persistence
joblib / pickle → Saving and loading trained models
🔹 Web Framework
Flask → Model deployment and UI integration
🌐 Output of the Project
A trained Random Forest ML model
A Flask web application
User enters startup parameters
Model predicts:
“Acquired / Successful”
“Closed / Failed”
Result displayed instantly on the UI


**🎯 Final Outcome**
✔ End-to-end ML pipeline
✔ Real-world startup use case
✔ Clean deployment using Flask
✔ Investor-ready prediction system
✔ Academically and industry suitable project
