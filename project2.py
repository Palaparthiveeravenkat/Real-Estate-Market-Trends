import pandas as pd
# -----------------------------
# Credit Score Distribution
plt.figure(figsize=(8, 5))
financial_df['Credit_Score'].hist(bins=30)
plt.title('Credit Score Distribution')
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Loan Amount vs Risk
risk_by_loan = financial_df.groupby('Risk_Level')['Loan_Amount'].mean()
plt.figure(figsize=(6, 4))
risk_by_loan.plot(kind='bar')
plt.title('Average Loan Amount by Risk Level')
plt.xlabel('Risk Level (0=Low, 1=High)')
plt.ylabel('Loan Amount')
plt.tight_layout()
plt.show()

# Missed Payments Impact
missed_payments_risk = financial_df.groupby('Missed_Payments')['Risk_Level'].mean()
plt.figure(figsize=(8, 5))
missed_payments_risk.plot()
plt.title('Risk Probability by Missed Payments')
plt.xlabel('Missed Payments')
plt.ylabel('Risk Probability')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 6: Feature Importance
# -----------------------------
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title('Financial Risk Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 7: Conclusion
# -----------------------------
print("""
Financial Risk Modeling Completed Successfully!

Key Insights:
- Identified high-risk borrowers
- Evaluated debt and credit behavior
- Predicted loan default probability
- Highlighted major financial risk indicators
""")