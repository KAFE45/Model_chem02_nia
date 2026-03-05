import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_sensor_model(df, model_name):
    print(f"\n{'='*40}\n🚀 เริ่มเทรน AI สำหรับเครื่อง: {model_name}\n{'='*40}")
    
    # 1. กำหนดตัวแปรต้น (Features: X) และตัวแปรตาม (Target: y)
    if model_name == "M01":
        features = ['Temp_M01', 'EC_M01', 'NaCl_Percent']
    else:
        features = ['Temp_M02', 'EC_M02', 'NaCl_Percent']
        
    X = df[features]
    y = df['Mercury_Temp'] # อุณหภูมิจริงจากปรอท (Ground Truth)
    
    # 2. แบ่งข้อมูลสำหรับ เทรน (80%) และ ทดสอบ (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. สร้างและสอน AI (Train Model) ด้วย Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 4. ให้ AI ลองทำนายข้อสอบ (Test Data)
    y_pred = rf_model.predict(X_test)
    
    # 5. ประเมินความแม่นยำ (Evaluation Metrics)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 ผลการประเมินความแม่นยำ (บน Test Set):")
    print(f" - MAE  (คลาดเคลื่อนเฉลี่ย): {mae:.4f} °C")
    print(f" - RMSE (ความคลาดเคลื่อนรุนแรง): {rmse:.4f} °C")
    print(f" - R²   (ความอธิบายข้อมูลได้): {r2:.4f} (ยิ่งใกล้ 1.0 ยิ่งดี)")
    
    # ดูความสำคัญของตัวแปร (Feature Importance)
    print("\n🔍 ตัวแปรที่มีผลต่อการชดเชยค่ามากที่สุด:")
    for feature, imp in zip(features, rf_model.feature_importances_):
        print(f" - {feature}: {imp*100:.2f}%")
        
    return rf_model
# ==========================================
# ส่วนสั่งรันโปรแกรม
# ==========================================
if __name__ == "__main__":
    # ระบุพาทไฟล์ที่คลีนแล้วของคุณ
    CLEANED_FILE = r"D:\Model_chem02_nia\data_CLEANED_ML_Ready.csv"
    
    try:
        # โหลดข้อมูล
        df = pd.read_csv(CLEANED_FILE)
        
        print(f"จำนวนข้อมูลก่อนล้างขีด (-): {len(df)} แถว")
        
        # 🛠️ พอยต์ที่แก้ Error: บังคับแปลงคอลัมน์ที่ต้องใช้ให้เป็นตัวเลข 
        # (ถ้าเจอ '-' หรือตัวอักษร มันจะเปลี่ยนเป็น NaN อัตโนมัติ)
        cols_to_check = ['NaCl_Percent', 'Mercury_Temp', 'Temp_M01', 'EC_M01', 'Temp_M02', 'EC_M02']
        for col in cols_to_check:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # ลบแถวที่มีค่าว่าง (NaN) ทิ้งไปอย่างปลอดภัย
        df = df.dropna(subset=cols_to_check)
        
        print(f"จำนวนข้อมูลที่พร้อมเทรนจริงๆ: {len(df)} แถว\n")
        
        # เทรนโมเดลเปรียบเทียบ M01 และ M02
        ai_m01 = train_sensor_model(df, "M01")
        ai_m02 = train_sensor_model(df, "M02")
        
    except FileNotFoundError:
        print("❌ ไม่พบไฟล์ข้อมูล กรุณาตรวจสอบพาทไฟล์อีกครั้ง")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดอื่นๆ: {e}")