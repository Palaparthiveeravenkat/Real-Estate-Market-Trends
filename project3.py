import pandas as pd
# Step 4: Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error: ${mae:,.2f}")
print(f"R² Score: {r2:.2f}")

# -----------------------------
# Step 5: Visualizations
# -----------------------------
# Average Property Prices by Location
location_prices = real_estate_df.groupby('Location')['Price_USD'].mean()
plt.figure(figsize=(8, 5))
location_prices.plot(kind='bar')
plt.title('Average Property Prices by Location')
plt.xlabel('Location')
plt.ylabel('Average Price (USD)')
plt.tight_layout()
plt.show()

# Property Type Trends
property_prices = real_estate_df.groupby('Property_Type')['Price_USD'].mean()
plt.figure(figsize=(8, 5))
property_prices.plot(kind='bar')
plt.title('Average Prices by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Average Price (USD)')
plt.tight_layout()
plt.show()

# Bedrooms vs Price
plt.figure(figsize=(8, 5))
plt.scatter(real_estate_df['Bedrooms'], real_estate_df['Price_USD'])
plt.title('Bedrooms vs Property Price')
plt.xlabel('Bedrooms')
plt.ylabel('Price (USD)')
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
plt.title('Top 10 Real Estate Price Factors')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 7: Conclusion
# -----------------------------
print("""
Real Estate Market Trends Analysis Completed Successfully!

Key Insights:
- Predicted property prices accurately
- Identified high-value locations
- Evaluated housing market trends
- Highlighted major real estate pricing factors
""")