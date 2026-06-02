import pandas as pd
from datetime import timedelta
import datetime as dt
import numpy as np


vitals = {
    "Heart Rate (max)":     {'itemid': 220045,           'agg': 'max', 'valid': (0, 300)},
    "Blood Pressure (min)": {'itemid': [220181, 220052], 'agg': 'min', 'valid': (0, 300)},
    "SpO2 (min)":           {'itemid': 220277,           'agg': 'min', 'valid': (0, 100)},
    "FiO2 (max)":           {'itemid': 223835,           'agg': 'max', 'valid': (21, 100)},
    "temperature_max_C":    {'itemid': 223762,           'agg': 'max', 'valid': (25, 45)},
    "temperature_max_F":    {'itemid': 223761,           'agg': 'max', 'valid': (77, 113)},
    "GCS Motor (min)":      {'itemid': 223901,           'agg': 'min', 'valid': (1, 6)},
    "GCS Verbal (min)":     {'itemid': 223900,           'agg': 'min', 'valid': (1, 5)},
    "GCS Eye (min)":        {'itemid': 220739,           'agg': 'min', 'valid': (1, 4)},
    "CVP (max)":            {'itemid': 220074,           'agg': 'max', 'valid': (-5, 30)},
}

labevents = {
    "Sodium (max)":          {'itemid': [50983,52623],  'agg': 'max', 'valid': (80, 200)},
    "Sodium (min)":          {'itemid': [50983,52623],  'agg': 'min', 'valid': (80, 200)},
    "Potassium (max)":       {'itemid': [52610,50971],  'agg': 'max', 'valid': (1, 15)},
    "BUN (max)":             {'itemid': [51006,52647],  'agg': 'max', 'valid': (0, 300)},
    "Creatinine (max)":      {'itemid': [50912,52546],  'agg': 'max', 'valid': (0, 50)},
    "Glucose (min)":         {'itemid': [50931,52569],  'agg': 'min', 'valid': (0, 2000)},
    "pH (min)":              {'itemid': [50820],        'agg': 'min', 'valid': (6.5, 8.0)},
    "Lactate (max)":         {'itemid': [50813,53154],  'agg': 'max', 'valid': (0, 50)},
    "Platelet (max)":        {'itemid': [51704,51265],  'agg': 'max', 'valid': (0, 3000)},
    "WBC (max)":             {'itemid': [51301,51516],  'agg': 'max', 'valid': (0, 500)},
    "Hemoglobin (min)":      {'itemid': [50811,51222],  'agg': 'min', 'valid': (0, 30)},
    "AST (max)":             {'itemid': [53088,50878],  'agg': 'max', 'valid': (0, 50000)},
    "ALT (max)":             {'itemid': [50861],        'agg': 'max', 'valid': (0, 50000)},
    "Bilirubin (max)":       {'itemid': [50885,53089],  'agg': 'max', 'valid': (0, 150)},
    "INR (max)":             {'itemid': [51675,51237],  'agg': 'max', 'valid': (0.5, 20)},
    "Lymphocytes Abs (min)": {'itemid': [51133,52769],  'agg': 'min', 'valid': (0, 50)},
}

antibiotics = ['Vancomycin', 'Piperacillin-Tazobactam', 'Ciprofloxacin', 'Ciprofloxacin HCl', 'Meropenem', 'CefePIME', 'CeftriaXONE', 'MetRONIDAZOLE (FLagyl)', 'CefTRIAXone', 'Acyclovir', 'CefazoLIN', 'Sulfameth/Trimethoprim DS', 'Tobramycin', 'Azithromycin', 'Levofloxacin', 'Ampicillin', 'Erythromycin', 'Clindamycin', 'Aztreonam', 'CeFAZolin', 'moxifloxacin', 'Linezolid', 'Micafungin', 'Sulfamethoxazole-Trimethoprim', 'Doxycycline Hyclate', 'CefTAZidime', 'MetroNIDAZOLE', 'Sulfameth/Trimethoprim SS']
antibiotic_patterns = [
    'adoxa', 'ala-tet', 'alodox', 'amikacin', 'amikin', 'amoxicill', 'amphotericin',
    'anidulafungin', 'ancef', 'clavulanate', 'ampicillin', 'augmentin', 'avelox',
    'avidoxy', 'azactam', 'azithromycin', 'aztreonam', 'axetil', 'bactocill', 'bactrim',
    'bactroban', 'bethkis', 'biaxin', 'bicillin l-a', 'cayston', 'cefazolin', 'cedax',
    'cefoxitin', 'ceftazidime', 'cefaclor', 'cefadroxil', 'cefdinir', 'cefditoren',
    'cefepime', 'cefotan', 'cefotetan', 'cefotaxime', 'ceftaroline', 'cefpodoxime',
    'cefpirome', 'cefprozil', 'ceftibuten', 'ceftin', 'ceftriaxone', 'cefuroxime',
    'cephalexin', 'cephalothin', 'cephapririn', 'chloramphenicol', 'cipro', 'ciprofloxacin',
    'claforan', 'clarithromycin', 'cleocin', 'clindamycin', 'cubicin', 'dicloxacillin',
    'dirithromycin', 'doryx', 'doxycy', 'duricef', 'dynacin', 'ery-tab', 'eryped', 'eryc',
    'erythrocin', 'erythromycin', 'factive', 'flagyl', 'fortaz', 'furadantin', 'garamycin',
    'gentamicin', 'kanamycin', 'keflex', 'kefzol', 'ketek', 'levaquin', 'levofloxacin',
    'lincocin', 'linezolid', 'macrobid', 'macrodantin', 'maxipime', 'mefoxin',
    'metronidazole', 'meropenem', 'methicillin', 'minocin', 'minocycline', 'monodox',
    'monurol', 'morgidox', 'moxatag', 'moxifloxacin', 'mupirocin', 'myrac', 'nafcillin',
    'neomycin', 'nicazel doxy 30', 'nitrofurantoin', 'norfloxacin', 'noroxin', 'ocudox',
    'ofloxacin', 'omnicef', 'oracea', 'oraxyl', 'oxacillin', 'pc pen vk', 'pce dispertab',
    'panixine', 'pediazole', 'penicillin', 'periostat', 'pfizerpen', 'piperacillin',
    'tazobactam', 'primsol', 'proquin', 'raniclor', 'rifadin', 'rifampin', 'rocephin',
    'smz-tmp', 'septra', 'septra ds', 'solodyn', 'spectracef', 'streptomycin',
    'sulfadiazine', 'sulfamethoxazole', 'trimethoprim', 'sulfatrim', 'sulfisoxazole',
    'suprax', 'synercid', 'tazicef', 'tetracycline', 'timentin', 'tobramycin', 'unasyn',
    'vancocin', 'vancomycin', 'vantin', 'vibativ', 'vibra-tabs', 'vibramycin', 'zinacef',
    'zithromax', 'zosyn', 'zyvox'
]

def is_antibiotic(medication):
    med_lower = str(medication).lower()
    return any(pattern in med_lower for pattern in antibiotic_patterns)

vasoactive_agents = ['Norepinephrine', 'Epinephrine', 'Vasopressin', 'Phenylephrine','Dopamine','Dobutamine','Milrinone']


    
def set_time_window(df, before, after):
    df = df.copy()
    df["start_window"] = df["sepsis_onset_time"] - timedelta(hours=before)
    df["end_window"]   = df["sepsis_onset_time"] + timedelta(hours=after)
    return df

def get_vitals(df, chartevents):
    df = df.copy()
    c = chartevents.copy()
    c["charttime"] =pd.to_datetime(c["charttime"])
    merged  = c.merge(df[["stay_id","intime","end_window","start_window"]],on="stay_id", how="right")
    mask =  (merged['start_window'] <= merged['charttime']) & (merged['charttime']<=merged["end_window"])
    merged = merged[mask]
    for vital, info in vitals.items():
        itemids = info["itemid"] if isinstance(info["itemid"], list) else [info["itemid"]]
        subset = merged[merged["itemid"].isin(itemids)]
        if "valid" in info:
            lo, hi = info["valid"]
            subset = subset[(subset["valuenum"] >= lo) & (subset["valuenum"] <= hi)]
        test = subset.groupby("stay_id")["valuenum"].agg(info["agg"]).reset_index(name=vital)
        df = df.merge(test, on="stay_id",how="left")
    return df

def get_labs(df, labs):
    df = df.copy()
    l = labs.copy()
    l["charttime"] = pd.to_datetime(l["charttime"])
    merged = df[["hadm_id","start_window","end_window"]].merge(l, on="hadm_id", how="left")
    mask = (merged['start_window'] <= merged['charttime']) & (merged['charttime'] <= merged["end_window"])
    merged = merged[mask]
    for event, info in labevents.items():
        subset = merged[merged["itemid"].isin(info["itemid"])]
        if "valid" in info:
            lo, hi = info["valid"]
            subset = subset[(subset["valuenum"] >= lo) & (subset["valuenum"] <= hi)]
        test = subset.groupby("hadm_id")["valuenum"].agg(info["agg"]).reset_index(name=event)
        df = df.merge(test, on="hadm_id", how="left")
    return df

        
def get_medications(df,pharmacy):
    df = df.copy()
    p = pharmacy.copy()
    p["starttime"] = pd.to_datetime(p["starttime"],errors="coerce")
    merged = p.merge(df[["hadm_id","start_window","end_window"]], on="hadm_id", how="inner")
    mask = (merged["starttime"] >= merged["start_window"]) & (merged["starttime"] <= merged["end_window"])
    merged = merged[mask]

    merged['is_antibiotic'] = merged['medication'].apply(is_antibiotic)
    ab = merged[merged['is_antibiotic']]
    vaso = merged[merged["medication"].isin(vasoactive_agents)]

    ab_flag = ab.groupby("hadm_id").size().rename("antibiotics_given") > 0
    vaso_flag = vaso.groupby("hadm_id").size().rename("vaso_given") > 0

    df = df.merge(ab_flag, on="hadm_id", how="left")
    df = df.merge(vaso_flag, on="hadm_id", how="left")
    df["Antibiotics"] = df["antibiotics_given"].notna().astype(int)
    df["Vasopressors"] = df["vaso_given"].notna().astype(int)
    return df
    
def get_max_creatinine_bun(df,labs):
    creatinine = labs[ (labs["itemid"].isin([50912,52546])) & (labs["hadm_id"].isin(df["hadm_id"]))]
    max_cre = creatinine.groupby("hadm_id")["valuenum"].max().reset_index(name="creatinine_admission_max")
    bun = labs[(labs["itemid"].isin([51006,52647])) & (labs["hadm_id"].isin(df["hadm_id"]))]
    max_bun = bun.groupby("hadm_id")["valuenum"].max().reset_index(name="bun_admission_max")
    df = df.merge(max_cre, on="hadm_id", how="left")
    df = df.merge(max_bun, on="hadm_id", how="left")
    return df

def get_time_to_first_antibiotic(df,pharmacy):
    df = df.copy()
    p = pharmacy.copy()
    p["starttime"] = pd.to_datetime(p["starttime"],errors="coerce")
    merged = p.merge(df[["hadm_id","admittime"]],on="hadm_id", how="right")
    antibiotics_df = merged[merged["medication"].isin(antibiotics)]
    mask = antibiotics_df["starttime"] >= antibiotics_df["admittime"]
    antibiotics_df = antibiotics_df[mask]
    first = antibiotics_df.groupby("hadm_id")["starttime"].min().reset_index(name="first_antibiotic_time")
    df = df.merge(first, on="hadm_id", how="left")
    df["time_to_first_antibiotic_hrs"] = (df["first_antibiotic_time"] - df["admittime"]).dt.total_seconds() / 3600
    return df

    
def get_bmi(df, omr):
    o = omr.copy()
    o["chartdate"] = pd.to_datetime(o["chartdate"])
    o["result_value"] = pd.to_numeric(o["result_value"], errors="coerce")
    
    merged = o.merge(df[["subject_id", "hadm_id", "admittime"]], 
                     on="subject_id", how="right")
    
    # Source 1 — BMI directly from OMR up to 1 year before admission, any time after
    bmi_direct = (
        merged[
            (merged["result_name"] == "BMI (kg/m2)") &
            (merged["chartdate"] >= merged["admittime"] - pd.Timedelta(days=365)) &
            (merged["result_value"] >= 10) & (merged["result_value"] <= 80)
        ]
        .sort_values(["hadm_id", "chartdate"])
        .groupby("hadm_id")["result_value"].first()
        .rename("BMI")
    )

    # Source 2 — calculate from height + weight
    # Height: no time restriction 
    height = (
        merged[merged["result_name"].isin(["Height (Inches)", "Height"])]
        .sort_values(["hadm_id", "chartdate"])
        .groupby("hadm_id")["result_value"].first()
        .rename("height")
    )

    # Weight: within 1 year before to 30 days after admission
    weight = (
        merged[
            (merged["result_name"].isin(["Weight (Lbs)", "Weight"])) &
            (merged["chartdate"] >= merged["admittime"] - pd.Timedelta(days=365)) &
            (merged["chartdate"] <= merged["admittime"] + pd.Timedelta(days=30))
        ]
        .sort_values(["hadm_id", "chartdate"])
        .groupby("hadm_id")["result_value"].first()
        .rename("weight")
    )

    bmi_calc = pd.concat([height, weight], axis=1).reset_index()
    bmi_calc["BMI_calc"] = (bmi_calc["weight"] / (bmi_calc["height"] ** 2)) * 703
    bmi_calc["BMI_calc"] = bmi_calc["BMI_calc"].where(
    (bmi_calc["BMI_calc"] >= 10) & (bmi_calc["BMI_calc"] <= 80), np.nan)
    bmi_calc = bmi_calc[["hadm_id", "BMI_calc"]]

    # Merge both sources — prefer direct BMI, fallback to calculated
    df = df.merge(bmi_direct.reset_index(), on="hadm_id", how="left")
    df = df.merge(bmi_calc, on="hadm_id", how="left")
    df["BMI"] = df["BMI"].fillna(df["BMI_calc"])
    df = df.drop(columns=["BMI_calc"])

    return df

def get_max_temperature(row):
    temp_f = row['temperature_max_F']
    temp_c = row['temperature_max_C']
    if pd.isna(temp_f) and pd.isna(temp_c):
        return np.nan

    elif pd.isna(temp_c):
        return temp_f

    elif pd.isna(temp_f):
        return temp_c * 9/5 + 32

    else:
        temp_c_as_f = temp_c * 9/5 + 32
        return max(temp_f, temp_c_as_f)

def get_fluid_balance(df, inputevents, outputevents):
    i = inputevents.copy()
    o = outputevents.copy()
    i = i.merge(df[["stay_id", "start_window", "end_window"]], on="stay_id", how="inner")
    o = o.merge(df[["stay_id", "start_window", "end_window"]], on="stay_id", how="inner")

    i = i[(i["starttime"] >= i["start_window"]) & (i["starttime"] <= i["end_window"])]
    o = o[(o["charttime"] >= o["start_window"]) & (o["charttime"] <= o["end_window"])]
    i.loc[i["amountuom"] == "L", "amount"] *= 1000
    o.loc[o["valueuom"] == "L", "value"]  *= 1000
    i = i[(i["amount"] >= 0) & (i["amount"] <= 50000)] 
    o = o[(o["value"] >= 0) & (o["value"] <= 50000)]
    total_in = i.groupby("stay_id")["amount"].sum().reset_index(name="total_fluid_input")
    total_out = o.groupby("stay_id")["value"].sum().reset_index(name="total_fluid_output")
    balance = total_in.merge(total_out, on="stay_id", how="outer").fillna(0)
    balance["Fluid Balance (mL)"] = balance["total_fluid_input"] - balance["total_fluid_output"]
    balance = balance.drop(columns=["total_fluid_input", "total_fluid_output"])
    balance["Fluid Balance (mL)"] = balance["Fluid Balance (mL)"].clip(-20000, 20000)
    df = df.merge(balance, on="stay_id", how="left")
    return df

def get_diuretics(df, pharmacy):
    p = pharmacy.copy()
    p["starttime"] = pd.to_datetime(p["starttime"], errors="coerce")
    p = p.merge(df[["hadm_id", "start_window", "end_window"]], on="hadm_id", how="inner")
    mask = (p["starttime"] >= p["start_window"]) & (p["starttime"] <= p["end_window"])
    p = p[mask]
    diuretic_patterns = ['furosemide', 'lasix', 'bumetanide', 'bumex', 'torsemide']
    p['is_diuretic'] = p['medication'].str.lower().str.contains('|'.join(diuretic_patterns), na=False)
    diuretic_flag = p[p['is_diuretic']].groupby("hadm_id").size().reset_index(name="diuretics_given")
    diuretic_flag["diuretics_given"] = 1
    df = df.merge(diuretic_flag[["hadm_id", "diuretics_given"]], on="hadm_id", how="left")
    df["Diuretics"] = df["diuretics_given"].fillna(0).astype(int)
    return df

def explore_bmi_missingness(df, omr):
    o = omr.copy()
    o["chartdate"] = pd.to_datetime(o["chartdate"])
    
    # What result_names exist in OMR?
    print("All result_names in OMR:")
    print(o["result_name"].value_counts().head(20))
    
    # How many unique patients have height/weight at all?
    has_height = o[o["result_name"] == "Height (Inches)"]["subject_id"].nunique()
    has_weight = o[o["result_name"] == "Weight (Lbs)"]["subject_id"].nunique()
    total = df["subject_id"].nunique()
    print(f"\nPatients with any height: {has_height} ({has_height/total*100:.1f}%)")
    print(f"Patients with any weight: {has_weight} ({has_weight/total*100:.1f}%)")
    
    # How many have height/weight AFTER admittime?
    merged = o.merge(df[["subject_id","hadm_id","admittime"]], on="subject_id", how="right")
    after_admit = merged[merged["chartdate"] >= merged["admittime"]]
    has_height_after = after_admit[after_admit["result_name"] == "Height (Inches)"]["subject_id"].nunique()
    has_weight_after = after_admit[after_admit["result_name"] == "Weight (Lbs)"]["subject_id"].nunique()
    print(f"\nPatients with height after admittime: {has_height_after} ({has_height_after/total*100:.1f}%)")
    print(f"Patients with weight after admittime: {has_weight_after} ({has_weight_after/total*100:.1f}%)")
    
    # Check chartevents for weight — itemid 224639
    print("\nNote: weight is often in chartevents (itemid 224639), not OMR")

