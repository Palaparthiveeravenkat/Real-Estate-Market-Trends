import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Step 1: Generate Sample Dataset
# -----------------------------
np.random.seed(42)
num_records = 1000

suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D']
regions = ['Asia', 'Europe', 'North America', 'South America']
transport_modes = ['Air', 'Sea', 'Road']

# Create synthetic supply chain data
data = {
    'Supplier': np.random.choice(suppliers, num_records),
    'Region': np.random.choice(regions, num_records),
    'Transport_Mode': np.random.choice(transport_modes, num_records),
    'Lead_Time_Days': np.random.randint(1, 30, num_records),
    'Cost_USD': np.random.randint(500, 5000, num_records),
    'Defect_Rate': np.round(np.random.uniform(0, 0.15, num_records), 2),
    'Delay_Days': np.random.randint(0, 15, num_records)
}

# Convert to DataFrame
supply_chain_df = pd.DataFrame(data)

# Create Risk Level target column
supply_chain_df['Risk_Level'] = np.where(
    (supply_chain_df['Delay_Days'] > 7) |
    (supply_chain_df['Defect_Rate'] > 0.08),
    1,
    0
)

# Save dataset
supply_chain_df.to_csv('supply_chain_data.csv', index=False)

print("Dataset Created Successfully!\n")
print(supply_chain_df.head())

# -----------------------------
# Step 2: Data Preprocessing
# -----------------------------
encoded_df = pd.get_dummies(
    supply_chain_df,
    columns=['Supplier', 'Region', 'Transport_Mode']
)

# Features and target
X = encoded_df.drop('Risk_Level', axis=1)
y = encoded_df['Risk_Level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Step 3: Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -----------------------------
# Step 4: Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Step 5: Visualizations
# -----------------------------

# Supplier Risk Analysis
supplier_risk = supply_chain_df.groupby('Supplier')['Risk_Level'].sum()
plt.figure(figsize=(8, 5))
supplier_risk.plot(kind='bar')
plt.title('Supplier Risk Analysis')
plt.xlabel('Supplier')
plt.ylabel('High Risk Shipments')
plt.tight_layout()
plt.show()

# Regional Delay Analysis
region_delay = supply_chain_df.groupby('Region')['Delay_Days'].mean()
plt.figure(figsize=(8, 5))
region_delay.plot(kind='bar')
plt.title('Average Delay by Region')
plt.xlabel('Region')
plt.ylabel('Average Delay Days')
plt.tight_layout()
plt.show()

# Transport Mode Risk Distribution
transport_risk = supply_chain_df.groupby('Transport_Mode')['Risk_Level'].mean()
plt.figure(figsize=(8, 5))
transport_risk.plot(kind='pie', autopct='%1.1f%%')
plt.title('Risk Distribution by Transport Mode')
plt.ylabel('')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 6: Feature Importance
# -----------------------------
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
feature_importance.head(10).plot(kind='bar')
plt.title('Top 10 Important Supply Chain Risk Factors')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 7: Conclusion
# -----------------------------
print("""
Supply Chain Risk Analysis Completed Successfully!

Key Insights:
- Identified high-risk suppliers
- Evaluated regional shipment delays
- Predicted risky shipments using machine learning
- Highlighted major operational risk factors
""")
