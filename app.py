import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_PATH = os.path.join(BASE_DIR, "Data Template.xlsx")

MODELS_DIR = os.path.join(BASE_DIR, "models")
POINT_DIR  = os.path.join(MODELS_DIR, "point_models")
DIST_DIR   = os.path.join(MODELS_DIR, "distribution_models")
TEMP_DIR   = os.path.join(MODELS_DIR, "temporal_models")

MIN_POINTS_FOR_TEMPORAL = 8


# =========================
# Template download
# =========================
def get_template_bytes() -> bytes:
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "rb") as f:
            return f.read()
    return b""


# =========================
# Robust datetime parsing
# =========================
def parse_meter_datetime(series: pd.Series) -> pd.Series:
    """
    Handles strings like:
    'Feb 01, 2026, 23:45:00:000000'
    (note the last colon before microseconds)
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"(?<=\d{2}:\d{2}:\d{2}):(?=\d{6}$)", ".", regex=True)

    out = pd.to_datetime(s, format="%b %d, %Y, %H:%M:%S.%f", errors="coerce")

    m = out.isna()
    if m.any():
        out.loc[m] = pd.to_datetime(s[m], format="%b %d, %Y, %H:%M:%S", errors="coerce")

    m = out.isna()
    if m.any():
        out.loc[m] = pd.to_datetime(s[m], errors="coerce")

    return out


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def normalize_to_0_100(s: pd.Series) -> pd.Series:
    """Robust scaling to 0..100 using 5th/95th percentiles."""
    if len(s) == 0:
        return s
    lo = np.nanpercentile(s, 5)
    hi = np.nanpercentile(s, 95)
    if (hi - lo) < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    out = (s - lo) / (hi - lo)
    return np.clip(out, 0, 1) * 100.0


# =========================
# Feature Engineering
# =========================
def compute_point_features(df: pd.DataFrame) -> pd.DataFrame:
    V = df[["V1", "V2", "V3"]].astype(float)
    A = df[["A1", "A2", "A3"]].astype(float)

    X = pd.DataFrame(index=df.index)

    X["V_mean"] = V.mean(axis=1)
    X["V_std"]  = V.std(axis=1)
    X["V_min"]  = V.min(axis=1)
    X["V_max"]  = V.max(axis=1)
    X["V_imb"]  = safe_div((X["V_max"] - X["V_min"]), X["V_mean"])

    X["A_mean"] = A.mean(axis=1)
    X["A_std"]  = A.std(axis=1)
    X["A_min"]  = A.min(axis=1)
    X["A_max"]  = A.max(axis=1)
    X["A_imb"]  = safe_div((X["A_max"] - X["A_min"]), X["A_mean"])

    X["A_phase_ratio_max_min"] = safe_div(X["A_max"], X["A_min"])
    X["S_proxy"] = (V["V1"] * A["A1"]) + (V["V2"] * A["A2"]) + (V["V3"] * A["A3"])

    X["V1_V3"] = safe_div(V["V1"], V["V3"])
    X["A1_A3"] = safe_div(A["A1"], A["A3"])

    # flags
    X["zero_A1"] = (A["A1"] == 0).astype(int)
    X["zero_A2"] = (A["A2"] == 0).astype(int)
    X["zero_A3"] = (A["A3"] == 0).astype(int)
    X["zero_V1"] = (V["V1"] == 0).astype(int)
    X["zero_V2"] = (V["V2"] == 0).astype(int)
    X["zero_V3"] = (V["V3"] == 0).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def compute_distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for meter, g in df.groupby("Meter Number", sort=False):
        V = g[["V1","V2","V3"]].astype(float)
        A = g[["A1","A2","A3"]].astype(float)

        V_mean_row = V.mean(axis=1)
        A_mean_row = A.mean(axis=1)

        V_imb_row = safe_div((V.max(axis=1) - V.min(axis=1)), V_mean_row)
        A_imb_row = safe_div((A.max(axis=1) - A.min(axis=1)), A_mean_row)

        rows.append({
            "Meter Number": meter,
            "n_points": len(g),

            "V_mean": V_mean_row.mean(),
            "V_std":  V_mean_row.std(ddof=0),
            "V_min":  V.min().min(),
            "V_max":  V.max().max(),
            "V_imb_mean": V_imb_row.mean(),
            "V_imb_std":  V_imb_row.std(ddof=0),

            "A_mean": A_mean_row.mean(),
            "A_std":  A_mean_row.std(ddof=0),
            "A_min":  A.min().min(),
            "A_max":  A.max().max(),
            "A_imb_mean": A_imb_row.mean(),
            "A_imb_std":  A_imb_row.std(ddof=0),

            "zeroA_ratio": (A == 0).mean().mean(),
            "zeroV_ratio": (V == 0).mean().mean(),
        })

    Xd = pd.DataFrame(rows).set_index("Meter Number")
    Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for meter, g in df.groupby("Meter Number", sort=False):
        if len(g) < MIN_POINTS_FOR_TEMPORAL:
            continue
        g = g.sort_values("Meter Datetime")
        V_series = g[["V1","V2","V3"]].astype(float).mean(axis=1).values
        A_series = g[["A1","A2","A3"]].astype(float).mean(axis=1).values

        dV = np.diff(V_series)
        dA = np.diff(A_series)

        def mad(x):
            if len(x) == 0:
                return 1.0
            med = np.median(x)
            return np.median(np.abs(x - med)) + 1e-6

        spike_A = int(np.sum(np.abs(dA) > 6.0 * mad(dA))) if len(dA) else 0
        spike_V = int(np.sum(np.abs(dV) > 6.0 * mad(dV))) if len(dV) else 0

        t = np.arange(len(A_series))
        slope_A = float(np.polyfit(t, A_series, 1)[0]) if len(t) >= 2 else 0.0
        slope_V = float(np.polyfit(t, V_series, 1)[0]) if len(t) >= 2 else 0.0

        rows.append({
            "Meter Number": meter,
            "n_points": len(g),
            "dA_std": float(np.std(dA)) if len(dA) else 0.0,
            "dA_max": float(np.max(np.abs(dA))) if len(dA) else 0.0,
            "dV_std": float(np.std(dV)) if len(dV) else 0.0,
            "dV_max": float(np.max(np.abs(dV))) if len(dV) else 0.0,
            "spike_A": spike_A,
            "spike_V": spike_V,
            "slope_A": slope_A,
            "slope_V": slope_V,
        })

    if not rows:
        return pd.DataFrame()
    Xt = pd.DataFrame(rows).set_index("Meter Number")
    Xt = Xt.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xt


# =========================
# Explainability (compact)
# =========================
def reason_from_features(pf_row: pd.Series) -> str:
    reasons = []
    if pf_row.get("A_imb", 0) >= 0.60:
        reasons.append("عدم اتزان تيار مرتفع")
    if pf_row.get("V_imb", 0) >= 0.08:
        reasons.append("عدم اتزان فولت مرتفع")
    if pf_row.get("A_phase_ratio_max_min", 0) >= 10:
        reasons.append("طور مسيطر/اختلال كبير بين الأطوار")
    if (pf_row.get("zero_A1", 0) + pf_row.get("zero_A2", 0) + pf_row.get("zero_A3", 0)) >= 1:
        reasons.append("تيار صفر بطور (احتمال CT/طور/حمل)")
    if pf_row.get("S_proxy", 0) < 1:
        reasons.append("حمل شبه معدوم/قراءة غير معتادة")
    if not reasons:
        reasons.append("نمط غير معتاد (Ensemble)")
    return "، ".join(reasons[:4])


def confidence_label(agreement: float, duration_points: int) -> str:
    if agreement >= 0.75 and duration_points >= 4:
        return "High"
    if agreement >= 0.50 and duration_points >= 2:
        return "Medium"
    return "Low"


# =========================
# Period extraction
# =========================
def extract_period_cases(g: pd.DataFrame, meter_thr: float, min_points: int = 2):
    """
    Extract contiguous segments where final_score >= meter_thr.
    Returns list of df segments.
    """
    g = g.sort_values("Meter Datetime").copy()
    high = g["final_score"] >= meter_thr

    cases = []
    start_i = None

    for i, is_high in enumerate(high.values):
        if is_high and start_i is None:
            start_i = i
        elif (not is_high) and start_i is not None:
            seg = g.iloc[start_i:i]
            if len(seg) >= min_points:
                cases.append(seg)
            start_i = None

    if start_i is not None:
        seg = g.iloc[start_i:]
        if len(seg) >= min_points:
            cases.append(seg)

    return cases


# =========================
# Load models
# =========================
@st.cache_resource
def load_models():
    scaler_p = joblib.load(os.path.join(POINT_DIR, "scaler.pkl"))
    iso_point = joblib.load(os.path.join(POINT_DIR, "isolation_forest.pkl"))
    mcd_point = joblib.load(os.path.join(POINT_DIR, "robust_cov_mcd.pkl"))

    scaler_d = joblib.load(os.path.join(DIST_DIR, "scaler.pkl"))
    iso_dist = joblib.load(os.path.join(DIST_DIR, "isolation_forest.pkl"))

    scaler_t = joblib.load(os.path.join(TEMP_DIR, "scaler.pkl"))
    iso_temp = joblib.load(os.path.join(TEMP_DIR, "isolation_forest.pkl"))

    return scaler_p, iso_point, mcd_point, scaler_d, iso_dist, scaler_t, iso_temp


# =========================
# Consolidation: many periods -> ONE case per meter
# =========================
def consolidate_meter_cases(meter: str, office: str, segs: list[pd.DataFrame], fallback_peak_row: pd.Series) -> dict:
    """
    Convert multiple detected segments (periods) into ONE consolidated case row per meter.
    - repeat_count: how many segments detected
    - total_duration_min: sum durations
    - longest_duration_min: max duration
    - start_time/end_time: earliest start, latest end among segments
    - suggested_visit_time: peak time of worst segment
    - electrical peaks: taken from worst peak row (highest final_score)
    - component scores: peak across all segments (Point/Dist/Temp/Final)
    - reason: from worst peak row
    - confidence: based on best agreement and evidence points (worst segment length)
    """
    # Flatten all rows in all segments
    all_seg = pd.concat(segs, ignore_index=True) if len(segs) else pd.DataFrame()
    if len(all_seg) == 0:
        # fallback: instant-only
        peak = fallback_peak_row
        pf_row = peak.filter(like="pf_").rename(lambda x: x.replace("pf_", ""))
        A_mean = float((peak["A1"] + peak["A2"] + peak["A3"]) / 3.0)
        V_mean = float((peak["V1"] + peak["V2"] + peak["V3"]) / 3.0)

        return {
            "case_type": "Instant",
            "Meter Number": meter,
            "Office": office,
            "risk_final_%": round(float(peak["final_score"]), 2),
            "risk_point_%": round(float(peak["risk_point_comp"]), 2),
            "risk_dist_%":  round(float(peak["risk_dist_comp"]), 2),
            "risk_temp_%":  round(float(peak["risk_temp_comp"]), 2),
            "paths_hit": peak.get("paths_hit", "Ensemble"),
            "confidence": confidence_label(float(peak["agreement"]), 1),
            "reason": reason_from_features(pf_row),

            "repeat_count": 1,
            "total_duration_min": 0,
            "longest_duration_min": 0,
            "points_total": 1,

            "start_time": peak["Meter Datetime"],
            "end_time": peak["Meter Datetime"],
            "suggested_visit_time": peak["Meter Datetime"],

            "V1_peak": round(float(peak["V1"]), 3),
            "V2_peak": round(float(peak["V2"]), 3),
            "V3_peak": round(float(peak["V3"]), 3),
            "A1_peak": round(float(peak["A1"]), 3),
            "A2_peak": round(float(peak["A2"]), 3),
            "A3_peak": round(float(peak["A3"]), 3),

            "V_mean_peak": round(V_mean, 3),
            "A_mean_peak": round(A_mean, 3),

            "V_imb_%": round(float(pf_row.get("V_imb", 0) * 100.0), 2),
            "A_imb_%": round(float(pf_row.get("A_imb", 0) * 100.0), 2),
            "Amax_Amin_ratio": round(float(pf_row.get("A_phase_ratio_max_min", 0)), 2),

            "S_proxy_peak": round(float(pf_row.get("S_proxy", 0)), 3),
        }

    # Compute segment-level summaries
    seg_summaries = []
    for seg in segs:
        seg = seg.sort_values("Meter Datetime")
        start = seg["Meter Datetime"].iloc[0]
        end = seg["Meter Datetime"].iloc[-1]
        duration_min = int((end - start).total_seconds() / 60) if pd.notna(start) and pd.notna(end) else 0
        peak = seg.sort_values("final_score", ascending=False).iloc[0]
        seg_summaries.append({
            "start": start,
            "end": end,
            "duration_min": duration_min,
            "points": int(len(seg)),
            "peak_row": peak,
            "peak_risk": float(peak["final_score"]),
            "peak_agreement": float(peak["agreement"]),
        })

    # Worst segment by peak risk (engineering: worst-case drives field visit)
    worst = max(seg_summaries, key=lambda x: x["peak_risk"])
    peak = worst["peak_row"]
    pf_row = peak.filter(like="pf_").rename(lambda x: x.replace("pf_", ""))

    # Consolidated time span
    start_time = min(s["start"] for s in seg_summaries)
    end_time   = max(s["end"]   for s in seg_summaries)

    # Consolidated durations
    total_duration = int(sum(s["duration_min"] for s in seg_summaries))
    longest_duration = int(max(s["duration_min"] for s in seg_summaries))
    points_total = int(sum(s["points"] for s in seg_summaries))
    repeat_count = int(len(seg_summaries))

    # Electrical / load summaries across ALL segments
    A_all = all_seg[["A1","A2","A3"]].astype(float)
    V_all = all_seg[["V1","V2","V3"]].astype(float)

    A_mean_series = A_all.mean(axis=1)
    V_mean_series = V_all.mean(axis=1)

    A_mean_peak = float(A_mean_series.max()) if len(A_mean_series) else 0.0
    A_mean_avg  = float(A_mean_series.mean()) if len(A_mean_series) else 0.0
    V_mean_peak = float(V_mean_series.max()) if len(V_mean_series) else 0.0
    V_mean_avg  = float(V_mean_series.mean()) if len(V_mean_series) else 0.0

    S_proxy_peak = float(all_seg["pf_S_proxy"].max()) if "pf_S_proxy" in all_seg.columns and len(all_seg) else float(pf_row.get("S_proxy", 0))
    S_proxy_avg  = float(all_seg["pf_S_proxy"].mean()) if "pf_S_proxy" in all_seg.columns and len(all_seg) else float(pf_row.get("S_proxy", 0))

    # Component scores (peak across ALL segments)
    risk_point_peak = float(all_seg["risk_point_comp"].max()) if "risk_point_comp" in all_seg.columns else float(peak["risk_point_comp"])
    risk_dist_peak  = float(all_seg["risk_dist_comp"].max())  if "risk_dist_comp"  in all_seg.columns else float(peak["risk_dist_comp"])
    risk_temp_peak  = float(all_seg["risk_temp_comp"].max())  if "risk_temp_comp"  in all_seg.columns else float(peak["risk_temp_comp"])
    risk_final_peak = float(all_seg["final_score"].max())

    # Paths hit (based on peaks)
    paths = []
    if risk_point_peak >= 80: paths.append("Point")
    if risk_dist_peak  >= 80: paths.append("Dist")
    if risk_temp_peak  >= 80: paths.append("Temp")
    paths_hit = "+".join(paths) if paths else "Ensemble"

    # Confidence: use worst segment evidence (agreement + points)
    conf = confidence_label(float(worst["peak_agreement"]), int(worst["points"]))

    return {
        "case_type": "Consolidated",
        "Meter Number": meter,
        "Office": office,

        "risk_final_%": round(risk_final_peak, 2),
        "risk_point_%": round(risk_point_peak, 2),
        "risk_dist_%":  round(risk_dist_peak, 2),
        "risk_temp_%":  round(risk_temp_peak, 2),
        "paths_hit": paths_hit,

        "confidence": conf,
        "reason": reason_from_features(pf_row),

        "repeat_count": repeat_count,
        "total_duration_min": total_duration,
        "longest_duration_min": longest_duration,
        "points_total": points_total,

        "start_time": start_time,
        "end_time": end_time,
        "suggested_visit_time": peak["Meter Datetime"],

        # Peak snapshot taken from WORST peak row (highest final_score)
        "V1_peak": round(float(peak["V1"]), 3),
        "V2_peak": round(float(peak["V2"]), 3),
        "V3_peak": round(float(peak["V3"]), 3),
        "A1_peak": round(float(peak["A1"]), 3),
        "A2_peak": round(float(peak["A2"]), 3),
        "A3_peak": round(float(peak["A3"]), 3),

        # Summary across all segments
        "V_mean_avg": round(V_mean_avg, 3),
        "V_mean_peak": round(V_mean_peak, 3),
        "A_mean_avg": round(A_mean_avg, 3),
        "A_mean_peak": round(A_mean_peak, 3),

        "V_imb_%": round(float(pf_row.get("V_imb", 0) * 100.0), 2),
        "A_imb_%": round(float(pf_row.get("A_imb", 0) * 100.0), 2),
        "Amax_Amin_ratio": round(float(pf_row.get("A_phase_ratio_max_min", 0)), 2),

        "S_proxy_avg": round(S_proxy_avg, 3),
        "S_proxy_peak": round(S_proxy_peak, 3),
    }


# =========================
# Inference (NO Top% + consolidated cases)
# =========================
@st.cache_data(show_spinner=False)
def run_inference(file_bytes: bytes,
                  min_case_risk: float,
                  meter_quantile_for_periods: float,
                  min_period_points: int,
                  compute_periods: bool,
                  max_rows_export: int = 5000):

    df = pd.read_excel(io.BytesIO(file_bytes))

    required_cols = ["Meter Number", "Meter Datetime", "Office", "V1","V2","V3","A1","A2","A3"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Meter Datetime"] = parse_meter_datetime(df["Meter Datetime"])
    df = df.dropna(subset=["Meter Datetime"]).copy()

    for c in ["V1","V2","V3","A1","A2","A3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["V1","V2","V3","A1","A2","A3"]).copy()

    df = df.sort_values(["Meter Number", "Meter Datetime"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Meter Number", "Meter Datetime"], keep="last")

    scaler_p, iso_point, mcd_point, scaler_d, iso_dist, scaler_t, iso_temp = load_models()

    # Point
    pf = compute_point_features(df)
    Xp_scaled = scaler_p.transform(pf)
    df["score_point_iforest"] = -iso_point.decision_function(Xp_scaled)
    df["score_point_mcd"] = mcd_point.mahalanobis(Xp_scaled)

    # Dist
    Xd = compute_distribution_features(df)
    Xd_scaled = scaler_d.transform(Xd)
    Xd["score_dist_iforest"] = -iso_dist.decision_function(Xd_scaled)
    df = df.merge(Xd[["score_dist_iforest","n_points"]], left_on="Meter Number", right_index=True, how="left")

    # Temp (optional)
    Xt = compute_temporal_features(df)
    if len(Xt) > 0:
        Xt_scaled = scaler_t.transform(Xt)
        Xt["score_temp_iforest"] = -iso_temp.decision_function(Xt_scaled)
        df = df.merge(Xt[["score_temp_iforest"]], left_on="Meter Number", right_index=True, how="left")
    else:
        df["score_temp_iforest"] = np.nan

    # Normalize model scores
    df["n_p_if"]  = normalize_to_0_100(df["score_point_iforest"])
    df["n_p_mcd"] = normalize_to_0_100(df["score_point_mcd"])
    df["n_d_if"]  = normalize_to_0_100(df["score_dist_iforest"])

    temp_fill = df["score_temp_iforest"].copy()
    if temp_fill.notna().any():
        temp_fill = temp_fill.fillna(temp_fill.median())
    else:
        temp_fill = temp_fill.fillna(0.0)
    df["n_t_if"]  = normalize_to_0_100(temp_fill)

    # Dynamic weights (temp optional)
    has_temp = df["score_temp_iforest"].notna().astype(float)
    w_point, w_dist, w_temp = 0.55, 0.30, 0.15

    w_temp_eff = w_temp * has_temp
    w_missing = w_temp * (1.0 - has_temp)
    w_point_eff = w_point + 0.6 * w_missing
    w_dist_eff  = w_dist  + 0.4 * w_missing

    df["final_score"] = (
        w_point_eff * (0.7 * df["n_p_if"] + 0.3 * df["n_p_mcd"]) +
        w_dist_eff  * df["n_d_if"] +
        w_temp_eff  * df["n_t_if"]
    )

    # Component scores (for reporting)
    df["risk_point_comp"] = (0.7 * df["n_p_if"] + 0.3 * df["n_p_mcd"])
    df["risk_dist_comp"]  = df["n_d_if"]
    df["risk_temp_comp"]  = df["n_t_if"]

    # Meter summary (all meters)
    meters = df.groupby("Meter Number", as_index=False).agg(
        Office=("Office","first"),
        n_points=("Meter Datetime","count"),
        start_time=("Meter Datetime","min"),
        end_time=("Meter Datetime","max"),
        risk_max=("final_score","max"),
        risk_mean=("final_score","mean"),
    ).sort_values("risk_max", ascending=False).reset_index(drop=True)

    # Attach point-features + agreement for explainability
    df_top = df.copy()
    pf_top = compute_point_features(df_top).add_prefix("pf_")
    df_top = df_top.join(pf_top)

    df_top["agree_count"] = (
        (df_top["risk_point_comp"] >= 80).astype(int) +
        (df_top["risk_dist_comp"]  >= 80).astype(int) +
        (df_top["risk_temp_comp"]  >= 80).astype(int)
    )
    df_top["agreement"] = df_top["agree_count"] / 3.0

    # =========================
    # Build consolidated cases: ONE ROW per meter
    # =========================
    consolidated_rows = []
    raw_period_rows = []  # optional drill-down (not shown by default)

    for meter, g in df_top.groupby("Meter Number", sort=False):
        g = g.sort_values("Meter Datetime")
        office = str(g["Office"].iloc[0])

        peak_row = g.sort_values("final_score", ascending=False).iloc[0]
        peak_risk = float(peak_row["final_score"])

        segs = []
        if compute_periods:
            thr = g["final_score"].quantile(meter_quantile_for_periods)
            segs = extract_period_cases(g, meter_thr=thr, min_points=min_period_points)

            # Store raw periods for profile (optional)
            for idx, seg in enumerate(segs, start=1):
                seg = seg.sort_values("Meter Datetime")
                s = seg["Meter Datetime"].iloc[0]
                e = seg["Meter Datetime"].iloc[-1]
                dur = int((e - s).total_seconds() / 60) if pd.notna(s) and pd.notna(e) else 0
                p = seg.sort_values("final_score", ascending=False).iloc[0]
                raw_period_rows.append({
                    "Meter Number": meter,
                    "Office": office,
                    "period_index": idx,
                    "start_time": s,
                    "end_time": e,
                    "duration_min": dur,
                    "points": int(len(seg)),
                    "risk_peak_%": round(float(p["final_score"]), 2),
                    "risk_point_%": round(float(p["risk_point_comp"]), 2),
                    "risk_dist_%":  round(float(p["risk_dist_comp"]), 2),
                    "risk_temp_%":  round(float(p["risk_temp_comp"]), 2),
                })

        # Decide: include meter only if anomalous (no healthy meters)
        # Criteria: peak risk >= min_case_risk OR (has any segments whose peak >= min_case_risk)
        include = False
        if peak_risk >= float(min_case_risk):
            include = True
        else:
            for seg in segs:
                if float(seg["final_score"].max()) >= float(min_case_risk):
                    include = True
                    break

        if not include:
            continue  # omit healthy meters completely

        # Build one consolidated row
        row = consolidate_meter_cases(meter=str(meter), office=office, segs=segs, fallback_peak_row=peak_row)
        consolidated_rows.append(row)

    cases = pd.DataFrame(consolidated_rows)
    raw_periods = pd.DataFrame(raw_period_rows)

    if len(cases) > 0:
        # Sort: from LOW to HIGH as you requested (user can sort in UI)
        cases = cases.sort_values(["risk_final_%", "confidence"], ascending=[True, True]).reset_index(drop=True)

    # Export rows for audit (only anomalous rows)
    df_export_rows = df_top[df_top["final_score"] >= float(min_case_risk)].copy()
    df_export_rows = df_export_rows.sort_values("final_score", ascending=False).head(max_rows_export)[
        ["Meter Number","Office","Meter Datetime","final_score","V1","V2","V3","A1","A2","A3",
         "risk_point_comp","risk_dist_comp","risk_temp_comp",
         "n_p_if","n_p_mcd","n_d_if","n_t_if"]
    ].copy()

    return df, meters, cases, raw_periods, df_export_rows


# =========================
# Excel export
# =========================
def build_excel_bytes(cases: pd.DataFrame, meters: pd.DataFrame, raw_periods: pd.DataFrame, rows_top: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        cases.to_excel(writer, sheet_name="Cases_Consolidated", index=False)
        raw_periods.to_excel(writer, sheet_name="Cases_Periods_Raw", index=False)
        meters.to_excel(writer, sheet_name="Meters", index=False)
        rows_top.to_excel(writer, sheet_name="TopRows", index=False)
    return output.getvalue()


# =========================
# UI
# =========================
st.set_page_config(page_title="NTL Detector - CT Meters", layout="wide")
st.title("NTL Detector – عرض هندسي للحالات الشاذة (بدون إغفال + بدون تكرار)")

st.sidebar.header("إعدادات التشغيل")
min_case_risk = st.sidebar.slider("حد الشذوذ لإظهار الحالات فقط (Risk >=)", 0, 100, 60, 1)

compute_periods = st.sidebar.checkbox("تحليل داخلي لفترات الشذوذ (للاستدلال فقط)", value=True)
meter_quantile_for_periods = st.sidebar.slider("حساسية تحديد الفترة داخل العداد (Quantile)", 0.80, 0.99, 0.95, 0.01)
min_period_points = st.sidebar.slider("أقل عدد نقاط لتكوين فترة", 1, 12, 2, 1)

st.sidebar.markdown("---")
st.sidebar.caption("ملاحظة: يتم دمج كل الفترات داخليًا في حالة واحدة لكل عداد (Consolidated).")

st.subheader("قالب الإدخال")
st.caption("حمّل القالب، عبّيه بالبيانات المطلوبة، ثم ارفعه هنا للتحليل.")

template_bytes = get_template_bytes()
if template_bytes:
    st.download_button(
        label="📥 تحميل قالب Excel (Data Template)",
        data=template_bytes,
        file_name="Data_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
else:
    st.warning("لم يتم العثور على ملف القالب داخل المشروع. تأكد أنه موجود باسم: Data Template.xlsx")

uploaded = st.file_uploader("ارفع ملف البيانات (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("ارفع ملف Excel يحتوي الأعمدة: Meter Number, Meter Datetime, Office, V1..V3, A1..A3")
    st.stop()

try:
    with st.spinner("جاري التحليل..."):
        file_bytes = uploaded.getvalue()
        rows_scored, meters_ranked, cases_table, raw_periods_table, rows_top_export = run_inference(
            file_bytes=file_bytes,
            min_case_risk=min_case_risk,
            meter_quantile_for_periods=meter_quantile_for_periods,
            min_period_points=min_period_points,
            compute_periods=compute_periods
        )
except Exception as e:
    st.error("حدث خطأ أثناء التحليل.")
    st.exception(e)
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("عدد الصفوف (Valid)", f"{len(rows_scored):,}")
c2.metric("عدد العدادات", f"{meters_ranked['Meter Number'].nunique():,}")
c3.metric("متوسط قراءات/عداد", f"{meters_ranked['n_points'].mean():.1f}")
c4.metric("عدد الحالات الشاذة (Consolidated)", f"{len(cases_table):,}")

tab1, tab2, tab3 = st.tabs(["Cases Explorer (حالة واحدة لكل عداد)", "Meters Explorer (كل العدادات)", "Meter Profile (ملف العداد)"])

with tab1:
    st.subheader("الحالات الشاذة (بدون تكرار) – مرتبة من الأقل إلى الأعلى Risk")
    st.caption("كل عداد يظهر كسطر واحد فقط. الأعمدة تشمل نتائج النماذج (لحظي/توزيعي/زمني) + النتيجة النهائية + قيم فولت/تيار/حمل رقمية.")

    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        meter_search = st.text_input("بحث برقم العداد (اختياري)", value="")
    with f2:
        office_vals = ["All"] + sorted(cases_table["Office"].dropna().astype(str).unique().tolist()) if len(cases_table) else ["All"]
        office_filter = st.selectbox("Office (اختياري)", office_vals)
    with f3:
        conf_filter = st.selectbox("Confidence (اختياري)", ["All", "High", "Medium", "Low"], index=0)

    filtered = cases_table.copy()
    if meter_search.strip():
        filtered = filtered[filtered["Meter Number"].astype(str).str.contains(meter_search.strip(), na=False)]
    if office_filter != "All" and len(filtered):
        filtered = filtered[filtered["Office"].astype(str) == str(office_filter)]
    if conf_filter != "All" and len(filtered):
        filtered = filtered[filtered["confidence"] == conf_filter]

    preferred_cols = [
        "case_type","Meter Number","Office",
        "risk_final_%","risk_point_%","risk_dist_%","risk_temp_%","paths_hit",
        "confidence","reason",
        "repeat_count","total_duration_min","longest_duration_min","points_total",
        "start_time","end_time","suggested_visit_time",
        "V1_peak","V2_peak","V3_peak","A1_peak","A2_peak","A3_peak",
        "V_mean_avg","V_mean_peak","A_mean_avg","A_mean_peak",
        "V_imb_%","A_imb_%","Amax_Amin_ratio",
        "S_proxy_avg","S_proxy_peak",
    ]
    cols = [c for c in preferred_cols if c in filtered.columns] + [c for c in filtered.columns if c not in preferred_cols]
    st.dataframe(filtered[cols], use_container_width=True)

with tab2:
    st.subheader("Meters Explorer – كل العدادات (للاستكشاف)")
    st.caption("هنا يظهر جميع العدادات. جدول الحالات أعلاه يظهر فقط العدادات الشاذة (حسب حد الشذوذ).")

    m1, m2 = st.columns([2, 1])
    with m1:
        meter_search2 = st.text_input("بحث برقم العداد (اختياري)", value="", key="meter_search2")
    with m2:
        office_vals2 = ["All"] + sorted(meters_ranked["Office"].dropna().astype(str).unique().tolist())
        office_filter2 = st.selectbox("Office (اختياري)", office_vals2, index=0)

    mdf = meters_ranked.copy()
    if meter_search2.strip():
        mdf = mdf[mdf["Meter Number"].astype(str).str.contains(meter_search2.strip(), na=False)]
    if office_filter2 != "All":
        mdf = mdf[mdf["Office"].astype(str) == str(office_filter2)]

    st.dataframe(mdf, use_container_width=True)

with tab3:
    st.subheader("Meter Profile – ملف العداد (أرقام فقط)")
    meter_list = meters_ranked["Meter Number"].astype(str).tolist()
    chosen = st.selectbox("اختر العداد", meter_list, index=0)

    mrow = meters_ranked[meters_ranked["Meter Number"].astype(str) == chosen].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Risk Max", f"{mrow['risk_max']:.2f}%")
    k2.metric("Risk Mean", f"{mrow['risk_mean']:.2f}%")
    k3.metric("عدد القراءات", f"{int(mrow['n_points'])}")
    k4.metric("Office", str(mrow["Office"]))

    st.markdown("### الحالة المجمعة لهذا العداد (إن كانت شاذة)")
    meter_case = cases_table[cases_table["Meter Number"].astype(str) == chosen].copy()
    st.dataframe(meter_case, use_container_width=True)

    if compute_periods:
        st.markdown("### الفترات الخام (تحليل داخلي – للشفافية فقط)")
        meter_periods = raw_periods_table[raw_periods_table["Meter Number"].astype(str) == chosen].copy()
        st.dataframe(meter_periods, use_container_width=True)

    st.markdown("### أعلى 15 قراءة (حسب final_score) لهذا العداد")
    meter_rows = rows_scored[rows_scored["Meter Number"].astype(str) == chosen].copy()
    top15 = meter_rows.sort_values("final_score", ascending=False).head(15)[
        ["Meter Datetime","final_score","V1","V2","V3","A1","A2","A3","risk_point_comp","risk_dist_comp","risk_temp_comp"]
    ]
    st.dataframe(top15, use_container_width=True)

# Export
st.subheader("تصدير النتائج (Excel)")
excel_bytes = build_excel_bytes(cases_table, meters_ranked, raw_periods_table, rows_top_export)
st.download_button(
    label="تحميل النتائج كملف Excel",
    data=excel_bytes,
    file_name="ntl_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Sheets: Cases_Consolidated (حالة واحدة لكل عداد) + Cases_Periods_Raw (فترات داخلية للشفافية) + Meters + TopRows.")
st.markdown("👨‍💻 Developed by: Mashhour Alabbas | 2026")
