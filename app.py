
import re
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="TES Radar from Excel", layout="wide")

st.title("Thermal Energy Storage (TES) — Radar from Excel")

st.markdown(
    "Upload your workbook (.xlsx) with **DATA** and **RANK** sheets. "
    "The app applies your **Both** logic to select materials by temperature and draws a radar with your 1–5 criteria.\n\n"
    "**Both logic**\n"
    "- Interval `a–b` → selected if `a ≤ T ≤ b` (open bounds allowed).\n"
    "- `≥ a` → selected if `T ≥ a`.\n"
    "- Single value `x` (phase change) → selected if `T > x`."
)

uploaded = st.file_uploader("Upload your Excel (.xlsx)", type=["xlsx"])

def parse_range(cell):
    \"\"\"Parse strings in 'Faixa_T_°C' into (expr_type, tmin, tmax).
    Supports: 'a-b', '>=a', '≤b', 'a' (single), commas/degree symbols, en-dash.
    Returns (kind, tmin, tmax) where kind in {'interval','ge','le','single','empty'}.
    \"\"\"
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return ("empty", np.nan, np.nan)
    s = str(cell).strip().lower()
    s = s.replace("ºc","").replace("°c","").replace(" c","").replace("c","")
    s = s.replace("–","-").replace("—","-").replace(" ", "")
    s = s.replace(",", ".")
    if not s:
        return ("empty", np.nan, np.nan)
    # >= or ≥
    if s.startswith(">=") or s.startswith("≥"):
        try:
            val = float(s[2:]) if s.startswith(">=") else float(s[1:])
            return ("ge", val, np.nan)
        except:
            return ("empty", np.nan, np.nan)
    # <= or ≤
    if s.startswith("<=") or s.startswith("≤"):
        try:
            val = float(s[2:]) if s.startswith("<=") else float(s[1:])
            return ("le", np.nan, val)
        except:
            return ("empty", np.nan, np.nan)
    # interval a-b
    if "-" in s:
        left, right = s.split("-", 1)
        left = left.strip(); right = right.strip()
        tmin = float(left) if left not in ("", None) else np.nan
        tmax = float(right) if right not in ("", None) else np.nan
        return ("interval", tmin, tmax)
    # single value
    try:
        val = float(s)
        return ("single", val, val)
    except:
        return ("empty", np.nan, np.nan)

def match_both(kind, tmin, tmax, T):
    if kind == "interval":
        ok_low  = (np.isnan(tmin) or T >= tmin)
        ok_high = (np.isnan(tmax) or T <= tmax)
        return bool(ok_low and ok_high)
    if kind == "ge":
        return bool(T >= tmin)
    if kind == "le":
        return bool(T <= tmax)
    if kind == "single":
        # strictly greater than the single 'phase' temperature
        return bool(T > (tmin if not np.isnan(tmin) else tmax))
    return False

if uploaded is not None:
    try:
        xls = pd.ExcelFile(uploaded)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        st.stop()

    needed = {"DATA","RANK"}
    if not needed.issubset(set(xls.sheet_names)):
        st.error(f"Your file must include sheets: {sorted(needed)}. Found: {xls.sheet_names}")
        st.stop()

    df_data = pd.read_excel(xls, sheet_name="DATA")
    df_rank = pd.read_excel(xls, sheet_name="RANK")

    # Normalize column names
    df_data.columns = [str(c).strip() for c in df_data.columns]
    df_rank.columns = [str(c).strip() for c in df_rank.columns]

    # Basic checks
    required_data_cols = {"Material","Tipo","Faixa_T_°C"}
    if not required_data_cols.issubset(df_data.columns):
        st.error(f"DATA sheet needs columns at least: {sorted(required_data_cols)}")
        st.stop()
    if "Material" not in df_rank.columns or "Avg rank" not in df_rank.columns:
        st.error("RANK sheet needs columns: Material, Avg rank (+ the 1–5 criteria columns).")
        st.stop()

    # Sidebar / controls
    with st.sidebar:
        T = st.number_input("Target temperature (°C)", value=82.0, step=1.0, format="%.0f")
        topN = st.number_input("Top N to chart", min_value=1, max_value=10, value=5)

    # Parse ranges
    parsed = df_data["Faixa_T_°C"].map(parse_range)
    df_data["ExprType"] = [k for (k,_,_) in parsed]
    df_data["T_min"]    = [tmin for (_,tmin,_) in parsed]
    df_data["T_max"]    = [tmax for (_,_,tmax) in parsed]
    df_data["Match_BOTH"] = [match_both(k,tmin,tmax,T) for (k,tmin,tmax) in parsed]

    # Join with rank
    df = df_data.merge(df_rank, on="Material", how="left", suffixes=("",""))
    dff = df[df["Match_BOTH"]].copy()
    if dff.empty:
        st.warning("No materials match the temperature with the Both logic.")
        st.stop()

    # Order by Avg rank (desc) and pick TopN
    dff = dff.sort_values(by="Avg rank", ascending=False).head(int(topN))

    st.subheader("Selected materials")
    st.dataframe(dff[["Material","Tipo","Faixa_T_°C","ExprType","T_min","T_max","Avg rank"]], use_container_width=True)

    # Criteria columns: everything on RANK except Material and Avg rank
    criteria_cols = [c for c in df_rank.columns if c not in ("Material","Avg rank")]
    if not criteria_cols:
        st.error("No criteria columns found on RANK (besides Material/Avg rank).")
        st.stop()

    # Radar chart
    categories = criteria_cols
    fig = go.Figure()
    for _, row in dff.iterrows():
        r = df_rank.loc[df_rank["Material"]==row["Material"], criteria_cols]
        if r.empty: 
            continue
        vals = r.iloc[0].fillna(0).tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=row["Material"],
            opacity=0.5
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,5], tick0=0, dtick=1)),
        showlegend=True,
        height=650
    )
    st.subheader("Radar (1–5)")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Parsing: interval a–b | ≥a | ≤b | single x. Selection: interval → a ≤ T ≤ b | ≥a → T ≥ a | single → T > x.")
else:
    st.info("Upload your workbook to get started. Use the Excel you already have for this project.")
