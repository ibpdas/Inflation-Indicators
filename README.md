# UK Inflation & Price Indicators — Defra Microservice (Streamlit)

A lightweight **Streamlit** app that fetches official UK price and inflation indicators from the **ONS Open API** —
ready for **economists and analysts**. Includes CPI/CPIH/RPI, sectoral CPI components (food, energy, water),
producer & trade price indices (PPI input/output, import/export), and macro/labour indicators (AWE, ULC, GDP deflator).

- **Live data** (no API keys) via ONS time-series endpoints
- **12-month (YoY)** and **month-on-month** calculations built in
- **CSV export** for Power BI, R, Python, and dashboards
- **Compare view** with optional rebasing to 100
- **Reasoning Chatbot (beta)** that summarises/compares trends using only on-screen data

---

## Quickstart

```bash
git clone https://github.com/your-org/uk-inflation-microservice
cd uk-inflation-microservice
pip install -r requirements.txt
streamlit run app.py

Created by Bandhu Das
