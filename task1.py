import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Loading the data
train = pd.read_csv('train.csv')

# Selecting features and target
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

# Droping missing values
train = train.dropna(subset=features + [target])

# Removing outliers
train = train[(train['GrLivArea'] < 4500) & (train['SalePrice'] < 600000)]

# Preparing training data
X = train[features]
y = train[target]

# Split data to evaluate model performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Accuracy After Outlier Removal ---")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:,.2f}")

# User's input
print("\n--- Enter house details to predict price ---")
sqft = float(input("Enter square footage (GrLivArea): "))
bed = int(input("Enter number of bedrooms: "))
bath = int(input("Enter number of full bathrooms: "))

# Prediction
input_data = pd.DataFrame([[sqft, bed, bath]], columns=features)
predicted_price = model.predict(input_data)[0]

print(f"\nPredicted House Price: ${predicted_price:,.2f}")

# Comparing with similar houses in training data
similar = train[
    (train['GrLivArea'] >= sqft - 100) & (train['GrLivArea'] <= sqft + 100) &
    (train['BedroomAbvGr'] == bed) &
    (train['FullBath'] == bath)
]

if not similar.empty:
    print("\n--- Similar houses from training data ---")
    print(similar[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']])
    print(f"\nAverage Actual Sale Price (Similar Homes): ${similar['SalePrice'].mean():,.2f}")
else:
    print("\nNo exact similar entries found in training data.")
