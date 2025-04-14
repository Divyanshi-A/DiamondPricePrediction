import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load dataset
df = pd.read_csv('diamonds.csv')

# EDA
print(df.info())
print(df.describe())

# Visualizations
plt.figure(figsize=(10, 5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Diamond Price Distribution')
plt.savefig('static/price_distribution.png')
plt.close()

sns.pairplot(df, hue='cut')
plt.savefig('static/pairplot.png')
plt.close()

# Convert categorical columns into numeric values before correlation
df_encoded = df.copy()
df_encoded['cut'] = df_encoded['cut'].astype('category').cat.codes
df_encoded['color'] = df_encoded['color'].astype('category').cat.codes
df_encoded['clarity'] = df_encoded['clarity'].astype('category').cat.codes

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig('static/correlation_heatmap.png')
plt.close()

# Preprocessing
X = df.drop('price', axis=1)
y = df['price']
categorical_cols = ['cut', 'color', 'clarity']
numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

# Save Preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train Baseline Models
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_preds = lin_reg.predict(X_test)

# Train Deep Learning Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Model Evaluation
models = {'Random Forest': rf_preds, 'Linear Regression': lin_preds, 'Neural Network': model.predict(X_test).flatten()}
eval_results = {}
for name, preds in models.items():
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    eval_results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")

# Save evaluation results
pd.DataFrame(eval_results).to_csv('model_evaluation.csv')

# Feature Importance
explainer = shap.Explainer(rf)
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, X_test[:100], show=False)
plt.savefig('static/shap_summary.png')
plt.close()

# Save Model
model.save('diamond_price_model.keras')
print("Model saved in .keras format and preprocessor saved successfully!")