import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
import feature_engineering as fe

MIMIC_HOSP  = "/Users/kayvans/Documents/mimic-iv-3.1/hosp"
MIMIC_ICU   = "/Users/kayvans/Documents/mimic-iv-3.1/icu"

COHORT_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/sepsis_cohort.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../../data/processed/analysis.csv")


WINDOW_BEFORE_HRS = 24
WINDOW_AFTER_HRS  = 24


USECOLS = {
    # hosp
    "patients":       ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"],
    "admissions":     ["subject_id", "hadm_id", "admittime", "dischtime", "race", "hospital_expire_flag"],
    "labevents":      ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
    "d_labitems":     ["itemid", "label"],
    "pharmacy":       ["subject_id", "hadm_id", "medication", "starttime"],
    "diagnoses_icd":  ["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num"],
    "d_icd_diagnoses":["icd_code", "long_title"],
    "procedures_icd": ["subject_id", "hadm_id", "icd_code", "icd_version"],
    "d_icd_procedures":["icd_code", "icd_version", "long_title"],
    "omr":            ["subject_id", "result_name", "result_value", "chartdate"],
    # icu
    "icustays":       ["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    "chartevents":    ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"],
    "d_items":        ["itemid", "label", "category"],
    "inputevents":  ["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "amount", "amountuom"],
    "outputevents": ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "value", "valueuom"],
}


def load_all() -> dict:
    def read(path, name):
        print(f"  Loading {name}...")
        return pd.read_csv(f"{path}/{name}.csv.gz", usecols=USECOLS.get(name))

    print("Loading MIMIC tables...")
    tables = {
        # hosp
        "patients":        read(MIMIC_HOSP, "patients"),
        "admissions":      read(MIMIC_HOSP, "admissions"),
        "labs":            read(MIMIC_HOSP, "labevents"),
        "d_labitems":      read(MIMIC_HOSP, "d_labitems"),
        "pharmacy":        read(MIMIC_HOSP, "pharmacy"),
        "diagnoses":       read(MIMIC_HOSP, "diagnoses_icd"),
        "d_diagnoses":     read(MIMIC_HOSP, "d_icd_diagnoses"),
        "procedures":      read(MIMIC_HOSP, "procedures_icd"),
        "d_procedures":    read(MIMIC_HOSP, "d_icd_procedures"),
        "omr":             read(MIMIC_HOSP, "omr"),
        # icu
        "icustays":        read(MIMIC_ICU, "icustays"),
        "chartevents":     read(MIMIC_ICU, "chartevents"),
        "d_items":         read(MIMIC_ICU, "d_items"),
        "inputevents":   read(MIMIC_ICU, "inputevents"),
        "outputevents":  read(MIMIC_ICU, "outputevents"),
    }
    print(f"Done. {len(tables)} tables loaded.\n")
    return tables


def filter_to_cohort(tables: dict, cohort: pd.DataFrame) -> dict:
    stay_ids = set(cohort["stay_id"])
    hadm_ids = set(cohort["hadm_id"])
    subj_ids = set(cohort["subject_id"])

    return {
        "patients":     tables["patients"][tables["patients"]["subject_id"].isin(subj_ids)],
        "admissions":   tables["admissions"][tables["admissions"]["hadm_id"].isin(hadm_ids)],
        "labs":         tables["labs"][tables["labs"]["hadm_id"].isin(hadm_ids)],
        "d_labitems":   tables["d_labitems"],
        "pharmacy":     tables["pharmacy"][tables["pharmacy"]["hadm_id"].isin(hadm_ids)],
        "diagnoses":    tables["diagnoses"][tables["diagnoses"]["hadm_id"].isin(hadm_ids)],
        "d_diagnoses":  tables["d_diagnoses"],
        "procedures":   tables["procedures"][tables["procedures"]["hadm_id"].isin(hadm_ids)],
        "d_procedures": tables["d_procedures"],
        "omr":          tables["omr"][tables["omr"]["subject_id"].isin(subj_ids)],
        "icustays":     tables["icustays"][tables["icustays"]["stay_id"].isin(stay_ids)],
        "chartevents":  tables["chartevents"][tables["chartevents"]["stay_id"].isin(stay_ids)],
        "d_items":      tables["d_items"],
        "inputevents":  tables["inputevents"][tables["inputevents"]["stay_id"].isin(stay_ids)],
        "outputevents": tables["outputevents"][tables["outputevents"]["stay_id"].isin(stay_ids)]
    }


def build_base(cohort: pd.DataFrame, tables: dict) -> pd.DataFrame:
    df = cohort.rename(columns={"icu_intime": "intime", "icu_outtime": "outtime"})

    for col in ["intime", "outtime", "sepsis_onset_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["ICU_length"] = (df["outtime"] - df["intime"]).dt.total_seconds() / (24 * 3600)

    df = df.merge(
        tables["patients"][["subject_id", "gender", "anchor_age"]],
        on="subject_id", how="left"
    )
    df = df.merge(
        tables["admissions"][["hadm_id", "admittime", "dischtime", "race", "hospital_expire_flag"]],
        on="hadm_id", how="left"
    )
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")
    return df


def build_features(base: pd.DataFrame, tables: dict) -> pd.DataFrame:

    df = base.copy()

    df = fe.set_time_window(df, before=WINDOW_BEFORE_HRS, after=WINDOW_AFTER_HRS)

    df = fe.get_vitals(df, chartevents=tables["chartevents"])

    df["Temperature (Max)"] = df.apply(fe.get_max_temperature, axis=1)
    df = df.drop(columns=["temperature_max_F", "temperature_max_C"], errors="ignore")

    df = fe.get_medications(df, pharmacy=tables["pharmacy"])

    df = fe.get_labs(df, labs=tables["labs"])

    df = fe.get_bmi(df, omr=tables["omr"])


    df = fe.get_fluid_balance(df,inputevents=tables["inputevents"], outputevents=tables["outputevents"])

    df = fe.get_diuretics(df, pharmacy=tables["pharmacy"])

    return df


if __name__ == "__main__":
    cohort = pd.read_csv(COHORT_PATH)
    print(f"Cohort loaded: {cohort.shape[0]} stays\n")

    tables = load_all()
    tables = filter_to_cohort(tables, cohort)

    base   = build_base(cohort, tables)
    result = build_features(base, tables)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  {result.shape}")
