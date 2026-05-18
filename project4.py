import pandas as pd
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Step 5: Visualizations
# -----------------------------
# Fraud Distribution
fraud_counts = fraud_df['Fraud_Risk'].value_counts()
plt.figure(figsize=(6, 4))
fraud_counts.plot(kind='bar')
plt.title('Fraud vs Non-Fraud Transactions')
plt.xlabel('Fraud Risk (0=No, 1=Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Transaction Type Fraud Rate
transaction_fraud = fraud_df.groupby('Transaction_Type')['Fraud_Risk'].mean()
plt.figure(figsize=(8, 5))
transaction_fraud.plot(kind='bar')
plt.title('Fraud Rate by Transaction Type')
plt.xlabel('Transaction Type')
plt.ylabel('Fraud Probability')
plt.tight_layout()
plt.show()

# Location Risk Analysis
location_risk = fraud_df.groupby('Location')['Fraud_Risk'].mean()
plt.figure(figsize=(6, 4))
location_risk.plot(kind='pie', autopct='%1.1f%%')
plt.title('Fraud Risk by Location')
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
plt.title('Top 10 Fraud Detection Risk Factors')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 7: Conclusion
# -----------------------------
print("""
Fraud Detection Analytics Completed Successfully!

Key Insights:
- Detected suspicious transactions
- Identified major fraud indicators
- Evaluated transaction channel risks
- Predicted fraud probability using machine learning
""")