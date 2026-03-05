import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# -----------------------
# Load Data
# -----------------------
df = pd.read_csv(r"D:\Model_chem02_nia\data_CLEANED_ML_Ready.csv")
df.columns = df.columns.str.strip()
df = pd.read_csv(r"D:\Model_chem02_nia\data_CLEANED_ML_Ready.csv")
df.columns = df.columns.str.strip()

# บังคับทุกคอลัมน์ที่ใช้ให้เป็นตัวเลข
cols_to_numeric = [
    "NaCl_Percent",
    "Mercury_Temp",
    "Temp_M01",
    "EC_M01",
    "Temp_M02",
    "EC_M02"
]

for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ลบแถวที่มีค่า NaN ออก
df = df.dropna(subset=cols_to_numeric)

print("Data ready:", len(df), "rows")

# ใช้เฉพาะ sensor M01
X = df[["Temp_M01", "EC_M01", "Mercury_Temp"]]
y = df["NaCl_Percent"]

# -----------------------
# Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------
# Train Model
# -----------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------
# Test
# -----------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# -----------------------
# Plot Predicted vs Actual (ถูกต้องสำหรับหลาย feature)
# -----------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual NaCl_Percent")
plt.ylabel("Predicted NaCl_Percent")
plt.title("Linear Regression (M01)")
plt.show()