# app.py — revised, dataset-aware ONS fetcher + full UI
import os
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yaml
from dateutil import parser as dtp

# Optional (only for local env if you want the chatbot)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ------------------------------------------------------------
# App config
# ------------------------------------------------------------
st.set_page_config(
    page_title="UK Inflation & Price Indicators (ONS) — Defra Microservice",
    layout="wide",
)

# ------------------------------------------------------------
# Helpers: configuration & LLM (optional)
# ------------------------------------------------------------
def _env(name: str, default=None):
    # Prefer Streamlit secrets; fall back to env vars
    return st.secrets.get(name, os.getenv(name, default))

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Provider-agnostic call: supports OpenAI-compatible or local Ollama.
    Configure via Streamlit secrets or environment variables:
      LLM_PROVIDER=openai|ollama
      OPENAI_API_KEY=...
      OPENAI_BASE_URL=https://api.openai.com/v1
      OPENAI_MODEL=gpt-4o-mini
      OLLAMA_BASE_URL=http://localhost:11434
      OLLAMA_MODEL=llama3.1:8b-instruct
    """
    provider = str(_env("LLM_PROVIDER", "openai")).lower()

    if provider == "ollama":
        base = str(_env("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        model = str(_env("OLLAMA_MODEL", "llama3.1:8b-instruct"))
        url = f"{base}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            return f"Ollama error: {e}"

    # OpenAI-compatible default
    base = str(_env("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
    model = str(_env("OPENAI_MODEL", "gpt-4o-mini"))
    api_key = _env("OPENAI_API_KEY", "")
    if not api_key:
        return "No OPENAI_API_KEY configured. Add it to Streamlit secrets."

    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"LLM error: {e}"

# ------------------------------------------------------------
# YAML loader (returns base_url, groups, dataset_map, endpoint_override)
# ------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60)
def load_catalog() -> Tuple[str, Dict[str, List[Dict]], Dict[str, str], Dict[str, str]]:
    """
    Load indicators.yml allowing 1+ YAML documents.
    Later docs override earlier keys; 'groups' are shallow-merged.
    Also strips UTF-8 BOM and accidental Markdown code fences.
    Returns: (ons_api_base, groups, dataset_map, endpoint_override)
    """
    import io as _io
    with open("indicators.yml", "rb") as f:
        raw = f.read()

    # Strip BOM if present
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]

    txt = raw.decode("utf-8", errors="replace")
    txt = txt.replace("```yaml", "").replace("```", "")

    docs = [d for d in yaml.safe_load_all(_io.StringIO(txt)) if isinstance(d, dict)]
    if not docs:
        raise ValueError("indicators.yml is empty or invalid YAML.")

    cfg: Dict = {}
    for d in docs:
        for k, v in d.items():
            if k == "groups" and isinstance(v, dict):
                cfg.setdefault("groups", {}).update(v)
            else:
                cfg[k] = v

    if "ons_api_base" not in cfg or "groups" not in cfg:
        raise ValueError("indicators.yml must define 'ons_api_base' and 'groups'.")

    dataset_map = cfg.get("dataset_map", {}) or {}
    endpoint_override = cfg.get("endpoint_override", {}) or {}
    return cfg["ons_api_base"], cfg["groups"], dataset_map, endpoint_override

# ------------------------------------------------------------
# ONS JSON flattening
# ------------------------------------------------------------
def _flatten_timeseries_json(js: dict) -> pd.DataFrame:
    """
    ONS time series API returns a 'years' array with optional 'months' or 'quarters'.
    Flatten to monthly-like time index when possible.
    """
    rows = []
    for y in js.get("years", []):
        year = y["year"]
        months = y.get("months")
        quarters = y.get("quarters")
        if months:
            for m in months:
                v = m.get("value")
                if v in (None, ""):
                    continue
                mm = f"{int(m['month']):02d}" if str(m.get("month", "")).isdigit() else m["month"]
                dt_obj = dtp.parse(f"{year}-{mm}-01")
                rows.append({"date": dt_obj, "value": float(v)})
        elif quarters:
            map_q = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}
            for q in quarters:
                v = q.get("value")
                if v in (None, ""):
                    continue
                mm = map_q.get(q["quarter"], "01")
                dt_obj = dtp.parse(f"{year}-{mm}-01")
                rows.append({"date": dt_obj, "value": float(v)})
        else:
            v = y.get("value")
            if v not in (None, ""):
                dt_obj = dtp.parse(f"{year}-01-01")
                rows.append({"date": dt_obj, "value": float(v)})

    if not rows:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

# ------------------------------------------------------------
# Robust ONS fetcher: classic → dataset_map → endpoint_override
# ------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_ons_series(
    series_code: str,
    base_url: str,
    dataset_map: Dict[str, str],
    endpoint_override: Dict[str, str],
) -> Tuple[pd.DataFrame, dict]:
    """
    Tries multiple ONS endpoints in a robust order:
    - Classic path
    - Dataset-qualified (two permutations)
    - Same again on beta domain
    - Optional full override from YAML
    Records every attempt in meta['debug'].
    """
    debug: List[str] = []

    def _try(url: str):
        try:
            r = requests.get(url, timeout=30)
            debug.append(f"GET {url} -> {r.status_code}")
            if r.status_code == 200:
                try:
                    js = r.json()
                except Exception as e:
                    debug.append(f"JSON error: {e}")
                    return None
                df = _flatten_timeseries_json(js)
                return {
                    "ok": True,
                    "status": 200,
                    "url": url,
                    "description": js.get("description"),
                    "dataset_id": js.get("datasetId"),
                    "label": js.get("label"),
                    "source": "ONS",
                }, df
            else:
                # keep short body for troubleshooting
                try:
                    body = r.text[:240]
                    debug.append(f"Body: {body}")
                except Exception:
                    pass
                return None
        except Exception as e:
            debug.append(f"Request error: {e}")
            return None

    # 1) Classic
    first = _try(base_url.format(code=series_code))
    if first:
        meta, df = first
        meta["debug"] = debug
        return df, meta

    # 2) Dataset-qualified (if known)
    ds = dataset_map.get(series_code)
    candidates: List[str] = []
    if ds:
        candidates.extend([
            f"https://api.ons.gov.uk/timeseries/{series_code}/dataset/{ds}/data",
            f"https://api.ons.gov.uk/dataset/{ds}/timeseries/{series_code}/data",
            f"https://api.beta.ons.gov.uk/timeseries/{series_code}/dataset/{ds}/data",
            f"https://api.beta.ons.gov.uk/dataset/{ds}/timeseries/{series_code}/data",
        ])

    # 3) Optional full override (put it last so explicit mapping wins if present)
    if series_code in endpoint_override:
        candidates.append(endpoint_override[series_code])

    for url in candidates:
        tried = _try(url)
        if tried:
            meta, df = tried
            meta["debug"] = debug
            return df, meta

    # 4) All attempts failed
    return pd.DataFrame(columns=["date", "value"]), {
        "ok": False,
        "status": "not_found_or_changed",
        "url": base_url.format(code=series_code),
        "source": "ONS",
        "debug": debug,
    }

# ------------------------------------------------------------
# Transforms / plotting
# ------------------------------------------------------------
def pct_change_12m(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["yoy_%"] = out["value"].pct_change(12) * 100
    return out

def pct_change_mom(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mom_%"] = out["value"].pct_change(1) * 100
    return out

def plot_line(df: pd.DataFrame, col: str, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["date"], df[col])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def download_csv(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download CSV — {label}",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )

def format_series_context(series_name: str, df: pd.DataFrame) -> dict:
    if df.empty or "date" not in df or "value" not in df:
        return {"series": series_name, "status": "no_data"}
    latest = df.dropna(subset=["value"]).iloc[-1]
    yoy_df = pct_change_12m(df)
    mom_df = pct_change_mom(df)
    yoy = float(yoy_df["yoy_%"].dropna().iloc[-1]) if not yoy_df["yoy_%"].dropna().empty else None
    mom = float(mom_df["mom_%"].dropna().iloc[-1]) if not mom_df["mom_%"].dropna().empty else None
    return {
        "series": series_name,
        "latest_date": str(latest["date"].date()),
        "latest_value": float(latest["value"]),
        "yoy_percent": yoy,
        "mom_percent": mom
    }

def build_context_blob(selected_labels: List[str], label_to_code: Dict[str, str], base_url: str,
                       dataset_map: Dict[str, str], endpoint_override: Dict[str, str]) -> dict:
    out = {"selected": []}
    for lab in selected_labels[:5]:
        code = label_to_code[lab]
        df, meta = fetch_ons_series(code, base_url, dataset_map, endpoint_override)
        out["selected"].append({
            "name": lab,
            "code": code,
            "api": meta.get("url") if meta else None,
            "summary": format_series_context(lab, df)
        })
    return out

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
try:
    base_url, groups, dataset_map, endpoint_override = load_catalog()
except Exception as e:
    st.error(f"Failed to read indicators.yml — {e}")
    st.stop()

st.title("UK Inflation & Price Indicators (ONS)")
st.caption(
    "Experimental Microservice: CPI/CPIH/RPI + sectoral, producer & trade indices. "
    "Live from ONS API (no keys).  •  Created by **Bandhu Das**"
)

# About pane
with st.expander("ℹ️ About this App", expanded=False):
    st.markdown(
        """
### 🎯 Purpose
This dashboard is a **microservice** that provides live UK inflation and price indicators
directly from the **Office for National Statistics (ONS) Open API** — no spreadsheets or manual downloads.
It supports economists, analysts, and policy colleagues working on food, agriculture, environment, and
infrastructure topics where price and cost trends are critical.

---

### 🧮 Data Coverage
- **Consumer Prices:** CPI, CPIH, RPI and sectoral components (food, energy, water, transport, etc.)  
- **Producer & Trade Prices:** Input/output PPI, import/export, agricultural, and construction indices  
- **Macro & Labour Indicators:** GDP Deflator, Average Weekly Earnings (AWE), Unit Labour Costs (ULC)

All data are sourced from **ONS Open Data APIs** under the Open Government Licence v3.0.

---

### 🧩 Key Widgets
| Area | Widget | Description |
|------|--------|-------------|
| **Sidebar** | Group selector | Navigate between indicator families (Consumer, Producer, etc.) |
|  | Series multiselect | Choose one or multiple indicators to visualise |
|  | Checkboxes | Toggle 12-month and month-on-month change views |
| **Main Tabs** | Index / YoY / MoM tabs | Switch between different time-series views |
|  | Download buttons | Export each view to CSV |
| **Compare Tab** | Overlay chart | Plot up to 5 series with rebasing option |
| **Footer** | Report issue link | Opens GitHub page to request new indicators or improvements |
"""
    )

# Data Release Calendar
with st.expander("📅 Data Release Calendar (UK)", expanded=False):
    st.markdown(
        """
Typical release timings for key UK price indicators (**exact dates vary**; confirm before briefings).

| Domain | Indicator | Publisher | Frequency | Usual Release Window (UK time) |
|---|---|---|---|---|
| Consumer Prices | **CPI / CPIH** | ONS | Monthly | ~ **3rd Wednesday**, **07:00** |
| Consumer Prices | **RPI** | ONS | Monthly | Alongside CPI/CPIH, **07:00** |
| Producer Prices | **PPI Input / Output** | ONS | Monthly | Often same day as CPI, **07:00** |
| Trade Prices | **Import / Export Price Index** | ONS | Monthly | Mid-month, **07:00** |
| Construction | **Construction Materials Price Index** | ONS | Monthly | Mid-to-late month, **07:00** |
| Construction | **Construction Output Price Index (OPI)** | ONS | Monthly/Quarterly | Monthly series at **07:00** |
| Housing | **UK House Price Index (HPI)** | ONS / HM Land Registry | Monthly (lag) | Mid-month, **09:30** |
| Labour | **Average Weekly Earnings (AWE)** | ONS | Monthly | Labour market release, **07:00** |
| Labour | **Unit Labour Costs (ULC)** | ONS | Quarterly | With National Accounts, **07:00** |
| Macro | **GDP Deflator** | ONS / HMT | Quarterly | With GDP / QNA, **07:00** |
| Expectations | **Inflation Attitudes Survey** | Bank of England | Quarterly | Per BoE schedule |
| Market | **Breakeven Inflation** | BoE / Markets | Daily | Market-driven, continuous |
"""
    )

# Sidebar controls
st.sidebar.header("Select indicators")
group_names = list(groups.keys())
sel_group = st.sidebar.selectbox("Group", group_names, index=0)

choices = groups[sel_group]
label_to_code = {c["name"]: c["code"] for c in choices}

sel_labels = st.sidebar.multiselect(
    "Series (select one or more)",
    list(label_to_code.keys()),
    default=list(label_to_code.keys())[:1],
)

st.sidebar.markdown("---")
show_yoy = st.sidebar.checkbox("Show 12-month % change", value=True)
show_mom = st.sidebar.checkbox("Show month-on-month % change", value=False)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: use the Compare tab to overlay multiple series.")

# Main tabs
tab_overview, tab_compare = st.tabs(["Single Series", "Compare"])

# ---------------- Single Series ----------------
with tab_overview:
    if not sel_labels:
        st.info("Pick at least one series from the sidebar.")
    else:
        label = sel_labels[0]
        code = label_to_code[label]
        st.subheader(label)

        with st.spinner(f"Fetching {label} ({code}) …"):
            df, meta = fetch_ons_series(code, base_url, dataset_map, endpoint_override)

        if not meta.get("ok") or df.empty:
            st.error(
                f"Could not load '{label}' (code {code}). Status: {meta.get('status')}."
            )
            st.caption(f"Endpoint attempted: {meta.get('url')}")
        else:
            info_col, btn_col = st.columns([3, 1])
            with info_col:
                st.markdown(
                    f"**Code:** `{code}`  •  **Source:** ONS  •  **Points:** {len(df)}  \n"
                    f"**API:** {meta.get('url')}"
                )
                if meta.get("label"):
                    st.caption(meta.get("label"))
            with btn_col:
                download_csv(df, f"{code}_index.csv", "Index")

            # Build tab list based on sidebar toggles
            tab_labels = ["Index level"]
            if show_yoy:
                tab_labels.append("12-month % change")
            if show_mom:
                tab_labels.append("Month-on-month % change")
            tabs = st.tabs(tab_labels)

            # Index tab
            with tabs[0]:
                plot_line(df, "value", f"{label} — Index", "Index")

            # YoY tab
            idx = 1
            if show_yoy:
                with tabs[idx]:
                    yoy = pct_change_12m(df)
                    plot_line(yoy, "yoy_%", f"{label} — YoY %", "%")
                    download_csv(yoy[["date", "yoy_%"]], f"{code}_yoy.csv", "YoY")
                idx += 1

            # MoM tab
            if show_mom:
                with tabs[idx]:
                    mom = pct_change_mom(df)
                    plot_line(mom, "mom_%", f"{label} — MoM %", "%")
                    download_csv(mom[["date", "mom_%"]], f"{code}_mom.csv", "MoM")

        # 🔧 Debug panel: shows all URL attempts/status codes
        with st.expander("🔧 Debug (fetch details)"):
            st.write(meta.get("debug"))

# ---------------- Compare ----------------
with tab_compare:
    st.subheader("Overlay up to 5 series")
    compare_labels = st.multiselect(
        "Pick series to overlay",
        options=list(label_to_code.keys()),
        default=sel_labels[: min(3, len(sel_labels))],
    )

    norm = st.checkbox("Normalise to 100 at first common date", value=True)
    calc = st.selectbox("Metric", ["Index", "12-month %", "Month-on-month %"], index=0)

    if compare_labels:
        fig, ax = plt.subplots(figsize=(11, 5))
        legend = []
        merged = None
        ylabel = "Index"

        for lab in compare_labels[:5]:
            code = label_to_code[lab]
            series_df, meta_c = fetch_ons_series(code, base_url, dataset_map, endpoint_override)
            if series_df.empty:
                st.warning(f"Skipped: {lab} ({code}) — no data returned.")
                continue

            tmp = series_df.copy()
            if calc == "12-month %":
                tmp = pct_change_12m(tmp).dropna(subset=["yoy_%"])
                ycol = "yoy_%"
                ylabel = "%"
            elif calc == "Month-on-month %":
                tmp = pct_change_mom(tmp).dropna(subset=["mom_%"])
                ycol = "mom_%"
                ylabel = "%"
            else:
                ycol = "value"
                ylabel = "Index"

            if calc == "Index" and norm:
                if merged is None:
                    merged = tmp[["date"]].copy()
                merged = merged.merge(
                    tmp[["date", ycol]].rename(columns={ycol: f"{lab}"}),
                    on="date",
                    how="outer",
                )

            ax.plot(tmp["date"], tmp[ycol])
            legend.append(lab)

        if calc == "Index" and norm and merged is not None:
            merged = merged.dropna()
            if not merged.empty:
                ax.clear()
                for lab in legend:
                    base_val = merged[lab].iloc[0]
                    ax.plot(merged["date"], merged[lab] / base_val * 100)
                ylabel = "Index (rebased=100)"

        ax.set_title(
            f"Compare — {calc} {'(rebased)' if (calc=='Index' and norm) else ''}"
        )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.grid(True, alpha=0.3)
        if legend:
            ax.legend(legend, loc="best")
        st.pyplot(fig)
    else:
        st.info("Pick at least one series to compare.")

# ---------------- Reasoning Chatbot (beta) ----------------
st.divider()
with st.expander("🧠 Reasoning Chatbot (beta)", expanded=False):
    st.caption("Asks an LLM to explain current trends using ONLY the structured context shown here.")
    mode = st.radio(
        "Task",
        ["Explain latest trends", "Compare selected series", "Create briefing bullets", "Custom question"],
        horizontal=True,
    )

    context_blob = build_context_blob(
        sel_labels or list(label_to_code.keys())[:1],
        label_to_code, base_url, dataset_map, endpoint_override
    )

    if st.checkbox("Show data context sent to the model", value=False):
        st.json(context_blob)

    if mode == "Custom question":
        user_q = st.text_area(
            "Ask a question about these indicators:",
            placeholder="e.g., Are energy prices still decelerating relative to food?"
        )
    elif mode == "Explain latest trends":
        user_q = "Explain the latest trends concisely for a policy brief, using the provided series context only."
    elif mode == "Compare selected series":
        user_q = "Compare the selected series: highlight divergences, accelerations/decelerations, and any notable crossovers."
    else:
        user_q = "Create 3–5 briefing bullets with numbers and dates, suitable for a ministerial note."

    system_prompt = (
        "You are a government-style analytical assistant. Use ONLY the structured context provided. "
        "Cite series names and dates explicitly. Do not invent data. If unsure, say so briefly. "
        "Keep answers concise, neutral, and policy-friendly."
    )

    if st.button("Generate analysis"):
        with st.spinner("Reasoning…"):
            import json as _json
            packed = _json.dumps(context_blob, ensure_ascii=False)
            full_user_prompt = f"DATA_CONTEXT_JSON:\n{packed}\n\nQUESTION:\n{user_q}"
            reply = call_llm(system_prompt, full_user_prompt)
        st.markdown("#### Result")
        st.write(reply)
        st.caption("Model output is experimental and for educational use.")

# ------------------------------------------------------------
# Footer: provenance, issues link, disclaimer, credit
# ------------------------------------------------------------
st.divider()

c1, c2 = st.columns([3, 1])
with c1:
    st.caption(
        "Data © Office for National Statistics (ONS) • Open Government Licence v3.0 • "
        "This application queries public endpoints only and processes no personal data."
    )
with c2:
    st.link_button(
        "🪲 Report an issue / Request a new series",
        "https://github.com/your-org/uk-inflation-microservice/issues",
        help="Open a GitHub issue to suggest improvements or request additional indicators.",
    )

st.markdown("---")
st.markdown(
    """
<div style="background-color:#f5f5f5;border-left:5px solid #2E7D32;padding:1em 1.2em;margin-top:1em;">
<b>Disclaimer:</b><br>
This dashboard is an <i>experimental prototype</i> created as part of an educational and analytical project.
It is <b>not an official government product or service</b> and does not represent the views or outputs of Defra or any UK Government department.
Data and charts are provided for learning and demonstration purposes only.
</div>
<p style="margin-top:0.6em;"><i>Created by <b>Bandhu Das</b></i></p>
""",
    unsafe_allow_html=True,
)
