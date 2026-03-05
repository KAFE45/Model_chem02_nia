import pandas as pd

def clean_experimental_data(input_path, output_path):
    print(f"กำลังอ่านและคลีนไฟล์: {input_path} ...")
    
    try:
        # 1. โหลดข้อมูลและจัดการหัวตาราง (Index)
        df = pd.read_excel(input_path, header=[0, 1], engine='openpyxl')
        
        cols = list(df.columns)
        for i, name in enumerate(['NaCl_Percent', 'model_group', 'Variable']):
            cols[i] = (name, '')
        df.columns = pd.MultiIndex.from_tuples(cols)
        
        # ถมช่องว่าง (Forward Fill) ให้เต็ม
        df.iloc[:, :3] = df.iloc[:, :3].ffill()
        
        # 2. พลิกตารางเป็นแนวตั้ง (Melt) และทำความสะอาดคอลัมน์อุณหภูมิ
        df_m = df.melt(id_vars=df.columns[:3].tolist())
        df_m.columns = ['NaCl_Percent', 'model_group', 'Variable', 'Target_Temp', 'Rep', 'Value']
        
        df_m['Target_Temp'] = df_m['Target_Temp'].astype(str).str.replace('°C', '').str.strip()
        df_m = df_m[df_m['Target_Temp'].str.isnumeric()].dropna(subset=['Value'])
        
        # 3. จัดกลุ่มข้อมูล (Pivot) ให้เป็น 1 บรรทัด = 1 การทดสอบ
        df_f = df_m.pivot_table(index=['NaCl_Percent', 'Target_Temp', 'Rep'], 
                                columns='Variable', values='Value', aggfunc='first').reset_index()
        df_f.columns.name = None 
        
        # 4. แปลงชนิดข้อมูล จัดเรียงคอลัมน์ และเซฟไฟล์
        df_f[['Target_Temp', 'Rep']] = df_f[['Target_Temp', 'Rep']].astype(int)
        
        target_cols = ['NaCl_Percent', 'Target_Temp', 'Rep', 'Mercury_Temp', 'Temp_M01', 'EC_M01', 'Temp_M02', 'EC_M02']
        df_f = df_f[[c for c in target_cols if c in df_f.columns]] # เลือกเฉพาะคอลัมน์ที่มี
        df_f = df_f.sort_values(['NaCl_Percent', 'Target_Temp', 'Rep']).reset_index(drop=True)
        
        df_f.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ คลีนสำเร็จ! ได้ข้อมูล {len(df_f)} แถว บันทึกที่: {output_path}")
        return df_f

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

# ==========================================
if __name__ == "__main__":
    INPUT_FILE = r"D:\Model_chem02_nia\data_RAW.xlsx"
    OUTPUT_FILE = r"D:\Model_chem02_nia\data_CLEANED_ML_Ready.csv"
    
    final_dataset = clean_experimental_data(INPUT_FILE, OUTPUT_FILE)
    if final_dataset is not None:
        print("\nตัวอย่างข้อมูล 5 บรรทัดแรก:")
        print(final_dataset.head())