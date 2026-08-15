import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv("output/train.csv")
test_df = pd.read_csv("output/test.csv")

app_features = ["LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet"]

X_train = train_df[app_features]
y_train = train_df["ClosePrice"]
X_test = test_df[app_features]
y_test = test_df["ClosePrice"]

app_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
app_model.fit(X_train, y_train)

print("Test R²:", app_model.score(X_test, y_test))

joblib.dump(app_model, "model.pkl")
print("Saved model.pkl")