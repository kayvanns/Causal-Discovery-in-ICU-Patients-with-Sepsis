import pandas as pd
from datetime import timedelta
import datetime as dt
import numpy as np


vitals = {"heart_rate_max":{'itemid':220045, 'agg':'max'}, "blood_pressure_min":{'itemid':220181,"agg":'min'},"spO2_min":{'itemid':220277,'agg':'min'},"FiO2_max":{'itemid':223835, 'agg':'max'},"temperature_max_C":{'itemid':223762, 'agg':'max'},"temperature_max_F":{'itemid':223761,'agg':'max'},"gsc_motor_min":{'itemid':223901,'agg':'min'},"gsc_verbal_min":{'itemid':223900,'agg':'min'},"gsc_eye_min":{'itemid':220739,'agg':'min'}}

labevents = {"sodium_max":{'itemid':[50983,52623],'agg':'max'}, "sodium_min":{'itemid':[50983,52623],'agg':'min'},"potassium_max":{'itemid':[52610,50971],'agg':'max'},"bun_max":{'itemid':[51006,52647], 'agg':'max'},"creatinine_max":{'itemid':[50912,52546],'agg':'max'},"glucose_min":{'itemid':[50931,52569],'agg':'min'},"pH_min":{'itemid':[50820],'agg':'min'},"lactate_max":{'itemid':[50813, 52442, 53154],'agg':'max'}, "platelet_max":{'itemid':[51704,51265],'agg':'max'},"wbc_max":{'itemid':[51301, 51755, 51756],'agg':'max'},"hemoglobin_min":{'itemid':[50811, 51222, 51640],'agg':'min'},"ast_max":{'itemid':[53088,50878],'agg':'max'},"alt_max":{'itemid':[50861],'agg':'max'},"bilirubin_max":{'itemid':[50885,53089],'agg':'max'},"inr_max":{'itemid':[51675,51237],'agg':'max'}}

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
        test = merged[merged["itemid"]==info["itemid"]].groupby("stay_id")["valuenum"].agg(info["agg"]).reset_index(name=vital)
        df = df.merge(test, on="stay_id",how="left")
    return df

def get_labs(df,labs):
    df = df.copy()
    l = labs.copy()
    l["charttime"] = pd.to_datetime(l["charttime"])
    merged = df[["hadm_id","start_window","end_window"]].merge(l, on="hadm_id", how="left")
    mask =  (merged['start_window'] <= merged['charttime']) & (merged['charttime']<=merged["end_window"])
    merged = merged[mask]
    for event, info in labevents.items():
        test = merged[merged["itemid"].isin(info["itemid"])].groupby("hadm_id")["valuenum"].agg(info["agg"]).reset_index(name=event)
        df = df.merge(test, on="hadm_id",how="left")
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
    df["antibiotics_given"] = df["antibiotics_given"].notna().astype(int)
    df["vaso_given"] = df["vaso_given"].notna().astype(int)
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
    o = o[o["result_name"].isin(["Height (Inches)", "Weight (Lbs)"])]
    merged = o.merge(df[["subject_id", "hadm_id", "admittime"]],on="subject_id", how="right")
    merged = merged[merged["chartdate"] >= merged["admittime"]]
    pivoted = merged.pivot_table(index=["subject_id", "hadm_id", "chartdate"], columns="result_name",values="result_value", aggfunc="first").reset_index()
    pivoted = pivoted.sort_values(["hadm_id", "chartdate"]).groupby("hadm_id").first().reset_index()
    pivoted["Height (Inches)"] = pd.to_numeric(pivoted["Height (Inches)"], errors="coerce")
    pivoted["Weight (Lbs)"] = pd.to_numeric(pivoted["Weight (Lbs)"], errors="coerce")
    pivoted["BMI"] = (pivoted["Weight (Lbs)"] / (pivoted["Height (Inches)"] ** 2))*703
    pivoted = pivoted[["hadm_id", "BMI"]]
    df = df.merge(pivoted, on="hadm_id", how="left")
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
    total_in = i.groupby("stay_id")["amount"].sum().reset_index(name="total_fluid_input")
    total_out = o.groupby("stay_id")["value"].sum().reset_index(name="total_fluid_output")
    balance = total_in.merge(total_out, on="stay_id", how="outer").fillna(0)
    balance["fluid_balance"] = balance["total_fluid_input"] - balance["total_fluid_output"]
    balance = balance.drop(columns=["total_fluid_input", "total_fluid_output"])
    df = df.merge(balance, on="stay_id", how="left")
    return df