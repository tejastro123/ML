
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from preprocessing import build_preprocessor

df = pd.read_csv("data/data.csv")

X = df.drop("target", axis=1)
y = df["target"]

num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(exclude='number').columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preprocessor = build_preprocessor(num_cols, cat_cols)
model = GradientBoostingClassifier(random_state=42)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "models/model.pkl")
print("Model trained and saved.")
