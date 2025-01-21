import pandas as pd
import joblib  # For saving the trained model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load the dataset
df = pd.read_csv('train.csv')

print("Dataset Preview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Encode categorical columns
encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = encoder.fit_transform(df[column])

# Handle missing values and scale features
scaler = MinMaxScaler()
for column in df.columns:
    mean = df[column].mean()
    df[column].fillna(mean, inplace=True)
    df[column] = scaler.fit_transform(df[[column]])

# Correlation with target variable
correlation = df.corr()
target_variable = correlation['is_claim']
print("\nFeature Correlation with Target Variable:")
print(target_variable.sort_values(ascending=False))

# Specify the desired columns for training
selected_columns = ['policy_tenure', 'age_of_policyholder', 'is_adjustable_steering', 'cylinder', 'is_claim']
new_df = df[selected_columns]

# Split data into features and target
X = new_df.drop('is_claim', axis=1)
y = new_df['is_claim']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy}")

# Export the trained model
joblib.dump(model, 'xgb_model.joblib')
print("\nModel saved as 'xgb_model.joblib'")
