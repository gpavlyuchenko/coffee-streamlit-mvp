import os, re, io, json
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
import streamlit as st

# ---- PDF (мягко) -----------------------------------------------------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ---- Page ------------------------------------------------------------------
st.set_page_config(page_title="Coffee Landed Cost — MVP", page_icon="☕", layout="wide")
st.title("☕ Coffee Landed Cost — MVP")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ---- Secrets / API keys ----------------------------------------------------
ALPHA_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", os.getenv("ALPHAVANTAGE_API_KEY"))

# ---- Small utils -----------------------------------------------------------
def fmt_ts(ts: Optional[int]) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M UTC") if ts else "n/a"

def load_json(filename: str, default: dict) -> dict:
    """Безопасное чтение JSON из data/ с дефолтом."""
    p = DATA_DIR / filename
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# ---- Conversions -----------------------------------------------------------
def arabica_centlb_to_usd_per_kg(cents_per_lb: float) -> float:
    # 1 lb = 0.45359237 kg; price in ¢/lb => $/kg
    return (cents_per_lb / 100.0) / 0.45359237

def robusta_usd_per_tonne_to_usd_per_kg(usd_per_tonne: float) -> float:
    return usd_per_tonne / 1000.0

# ---- Market fetchers -------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_stooq_csv(symbol: str, interval: str = "d") -> pd.DataFrame:
    """CSV со Stooq (два домена, UA). Поддерживает kc.f / rm.f / RMU25.F и т.п."""
    headers = {"User-Agent": "Mozilla/5.0"}
    for base in ("https://stooq.com", "https://stooq.pl"):
        url = f"{base}/q/d/l/?s={symbol}&i={interval}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.ok and r.text.strip():
            first = r.text.splitlines()[0].lower()
            if "date" in first and "close" in first:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip().lower() for c in df.columns]
                return df.dropna()
    raise RuntimeError(f"Stooq CSV not available for {symbol}")

def stooq_front_series_symbol(base_symbol: str = "rm.f") -> Optional[str]:
    """Пытаемся обнаружить актуальную серию робусты на странице Stooq (например RMU25.F)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://stooq.com/q/f/?s={base_symbol}"
    r = requests.get(url, headers=headers, timeout=10)
    if not r.ok or not r.text:
        return None
    m = re.search(r"RM[A-Z]\d{2}\.F", r.text, re.IGNORECASE)
    return m.group(0).upper() if m else None

def fetch_yahoo_intraday_last(symbol: str, interval: str = "1m", range_: str = "1d") -> Tuple[float, Optional[int]]:
    """Последняя непустая свеча Yahoo (интрадей). Возвращает (price, unix_ts_utc)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": range_, "interval": interval}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    closes = j["indicators"]["quote"][0]["close"] or []
    ts = j.get("timestamp") or []
    last = None; last_ts = None
    for i in range(len(closes) - 1, -1, -1):
        if closes[i] is not None:
            last = float(closes[i]); last_ts = ts[i] if i < len(ts) else None
            break
    if last is None:
        last = float(j["meta"]["regularMarketPrice"])
        last_ts = j["meta"].get("currentTradingPeriod", {}).get("regular", {}).get("end", None)
    return last, last_ts

@st.cache_data(ttl=3600)
def fetch_alpha_coffee_monthly(api_key: str, interval: str = "monthly") -> Tuple[float, str]:
    """
    Alpha Vantage: Global price of Coffee (IMF, Other Mild Arabica), ежемесячно.
    Возвращает (usd_per_kg, iso_date).
    """
    if not api_key:
        raise RuntimeError("Alpha Vantage API key is missing")
    url = "https://www.alphavantage.co/query"
    params = {"function": "COFFEE", "interval": interval, "apikey": api_key}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    data = j.get("data") or []
    if not data:
        raise RuntimeError(j.get("note") or "Alpha Vantage: empty data")
    rec = max(data, key=lambda x: x.get("date", ""))
    usd_per_kg = float(rec["value"])
    return usd_per_kg, rec["date"]

@st.cache_data(ttl=900)  # 15 минут
def get_live_prices(seed: int = 0,
                    kc_yahoo: str = "KC=F",
                    rm_yahoo_candidates: Tuple[str, ...] = ("RC=F", "RM=F"),
                    prefer_alpha_for_arabica: bool = False) -> dict:
    """
    Возвращает словарь:
      KC.F: {last_raw (¢/lb), usdkg, unit, source, asof}
      RM.F: {last_raw (USD/t), usdkg, unit, source, asof}
    """
    data = {}

    # ---- Arabica (¢/lb) ----
    def pack_arabica_from_usdkg(usdkg: float, source: str, asof=None):
        cents_per_lb = usdkg * 0.45359237 * 100.0
        return {"last_raw": cents_per_lb, "unit": "¢/lb", "usdkg": float(usdkg),
                "source": source, "asof": asof}

    if prefer_alpha_for_arabica and ALPHA_KEY:
        try:
            usdkg, d = fetch_alpha_coffee_monthly(ALPHA_KEY, "monthly")
            data["KC.F"] = pack_arabica_from_usdkg(usdkg, "AlphaVantage (IMF monthly)", d)
        except Exception:
            pass

    if "KC.F" not in data:
        try:
            kc = fetch_stooq_csv("kc.f")
            last_kc = float(kc.iloc[-1]["close"])  # ¢/lb
            data["KC.F"] = {"last_raw": last_kc, "unit": "¢/lb",
                            "usdkg": arabica_centlb_to_usd_per_kg(last_kc),
                            "source": "Stooq", "asof": None}
        except Exception:
            try:
                last_kc, last_ts = fetch_yahoo_intraday_last(kc_yahoo, "1m", "1d")
                data["KC.F"] = {"last_raw": last_kc, "unit": "¢/lb",
                                "usdkg": arabica_centlb_to_usd_per_kg(last_kc),
                                "source": f"Yahoo 1m ({kc_yahoo})", "asof": last_ts}
            except Exception:
                if ALPHA_KEY:
                    try:
                        usdkg, d = fetch_alpha_coffee_monthly(ALPHA_KEY, "monthly")
                        data["KC.F"] = pack_arabica_from_usdkg(usdkg, "AlphaVantage (IMF monthly)", d)
                    except Exception as e:
                        data["KC.F"] = {"error": f"Arabica: {e}"}
                else:
                    data["KC.F"] = {"error": "Arabica: no free sources responded"}

    # ---- Robusta (USD/tonne) ----
    try:
        rm = fetch_stooq_csv("rm.f")
        last_rm = float(rm.iloc[-1]["close"])
        data["RM.F"] = {"last_raw": last_rm, "unit": "USD/t",
                        "usdkg": robusta_usd_per_tonne_to_usd_per_kg(last_rm),
                        "source": "Stooq", "asof": None}
    except Exception:
        try:
            front = stooq_front_series_symbol("rm.f")
            if front:
                rm_series = fetch_stooq_csv(front)
                last_rm = float(rm_series.iloc[-1]["close"])
                data["RM.F"] = {"last_raw": last_rm, "unit": "USD/t",
                                "usdkg": robusta_usd_per_tonne_to_usd_per_kg(last_rm),
                                "source": f"Stooq ({front})", "asof": None}
            else:
                raise RuntimeError("No Stooq front month found")
        except Exception:
            last_rm = None; last_ts = None; picked = None; last_err = None
            for ysym in rm_yahoo_candidates:
                try:
                    yprice, yts = fetch_yahoo_intraday_last(ysym, "1m", "1d")
                    last_rm, last_ts, picked = float(yprice), yts, ysym
                    break
                except Exception as e:
                    last_err = e
            if last_rm is not None:
                data["RM.F"] = {"last_raw": last_rm, "unit": "USD/t",
                                "usdkg": robusta_usd_per_tonne_to_usd_per_kg(last_rm),
                                "source": f"Yahoo 1m ({picked})", "asof": last_ts}
            else:
                data["RM.F"] = {"error": f"Robusta: {last_err}"}

    data["ts"] = datetime.now(timezone.utc).isoformat()
    return data

# ---- Core calc -------------------------------------------------------------
def compute_customs_value(incoterm: str, goods_value: float, freight: float, insurance: float) -> float:
    inc = incoterm.upper()
    if inc in {"FOB", "EXW"}:
        return goods_value + freight + insurance
    if inc == "CFR":
        return goods_value + insurance
    # CIF: уже включает фрахт+страховку
    return goods_value

def compute_quote(usd_per_kg, weight_kg, incoterm, freight, insurance,
                  duty_rate, duty_sp_perkg, vat_rate, fees):
    goods_value = usd_per_kg * weight_kg
    cv = compute_customs_value(incoterm, goods_value, freight, insurance)
    duty_ad = cv * duty_rate
    duty_sp = duty_sp_perkg * weight_kg
    duty_total = duty_ad + duty_sp

    def fee_amt(f):
        if f["kind"] == "fixed":
            return float(f.get("amount", 0))
        base = f.get("base", "CV")
        base_val = cv if base == "CV" else goods_value if base == "Goods" else cv + duty_total
        return float(f.get("rate", 0)) * base_val

    fees_list = [{"name": f.get("name", "Fee"),
                  "amount": fee_amt(f),
                  "vat_base": bool(f.get("vat_base", True))} for f in fees]
    fees_total = sum(f["amount"] for f in fees_list)
    vat_base = cv + duty_total + sum(f["amount"] for f in fees_list if f["vat_base"])
    vat = vat_base * vat_rate
    total = cv + duty_total + fees_total + vat
    per_kg = total / max(1.0, weight_kg)
    return {"goods_value": goods_value, "customs_value": cv,
            "duty_ad": duty_ad, "duty_sp": duty_sp, "duty_total": duty_total,
            "fees": fees_list, "fees_total": fees_total,
            "vat_base": vat_base, "vat": vat, "total": total, "per_kg": per_kg}

def make_result_df(b, currency="USD"):
    rows = [
        ["Goods value", b["goods_value"], currency],
        ["Customs value (CV)", b["customs_value"], currency],
        ["Duty (ad val.)", b["duty_ad"], currency],
        ["Duty (specific)", b["duty_sp"], currency],
        ["Duty total", b["duty_total"], currency],
        ["Fees total", b["fees_total"], currency],
        ["VAT base", b["vat_base"], currency],
        ["VAT", b["vat"], currency],
        ["Landed total", b["total"], currency],
        ["Per kg", b["per_kg"], f"{currency}/kg"],
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])

def export_excel(b, calc_params):
    df_main = make_result_df(b)
    df_fees = pd.DataFrame([{"Fee": f["name"], "Amount": f["amount"], "In VAT base": f["vat_base"]} for f in b["fees"]])
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame([calc_params]).to_excel(writer, index=False, sheet_name="Input")
        df_main.to_excel(writer, index=False, sheet_name="Result")
        df_fees.to_excel(writer, index=False, sheet_name="Fees")
    buf.seek(0); return buf

def export_pdf(b, calc_params):
    if not REPORTLAB_OK:
        return None
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4; y = h - 2*cm
    c.setFont("Helvetica-Bold", 14); c.drawString(2*cm, y, "Coffee Landed Cost — Summary"); y -= 1*cm
    c.setFont("Helvetica", 10)
    for k, v in calc_params.items():
        c.drawString(2*cm, y, f"{k}: {v}"); y -= 0.5*cm
        if y < 2*cm: c.showPage(); y = h - 2*cm; c.setFont("Helvetica", 10)
    y -= 0.5*cm
    items = [("Customs value (CV)", b["customs_value"]), ("Duty ad val.", b["duty_ad"]),
             ("Duty specific", b["duty_sp"]), ("Duty total", b["duty_total"]),
             ("Fees total", b["fees_total"]), ("VAT base", b["vat_base"]),
             ("VAT", b["vat"]), ("Landed total", b["total"]), ("Per kg", b["per_kg"])]
    c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, "Result"); y -= 0.7*cm; c.setFont("Helvetica", 10)
    for name, val in items:
        c.drawString(2*cm, y, f"{name}: {val:,.4f} USD"); y -= 0.5*cm
        if y < 2*cm: c.showPage(); y = h - 2*cm; c.setFont("Helvetica", 10)
    c.showPage(); c.save(); buf.seek(0); return buf

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.subheader("Источник котировок")
    kc_yahoo = st.text_input("Arabica — Yahoo symbol", value="KC=F",
                             help="Непрерывный контракт. Для месяца: KCU25=F, KCH26=F (на странице Yahoo это KCU25.NYB).")
    rm_main = st.text_input("Robusta — Yahoo symbol (1)", value="RC=F")
    rm_alt  = st.text_input("Robusta — Yahoo symbol (2, запасной)", value="RM=F")
    prefer_alpha = st.checkbox("Arabica: предпочитать Alpha Vantage (IMF monthly)", value=False)
    st.caption("Yahoo отдаёт интрадей (1m) с задержкой; Alpha — ежемесячный индекс IMF.")

    st.divider()
    if "refresh_seed" not in st.session_state: st.session_state["refresh_seed"] = 0
    if st.button("↻ Обновить котировки"): st.session_state["refresh_seed"] += 1

    data = get_live_prices(
        st.session_state["refresh_seed"],
        kc_yahoo=kc_yahoo.strip() or "KC=F",
        rm_yahoo_candidates=tuple(s for s in [rm_main.strip(), rm_alt.strip()] if s),
        prefer_alpha_for_arabica=prefer_alpha
    )
    kc = data.get("KC.F", {}); rm = data.get("RM.F", {})

    colA, colB = st.columns(2)
    with colA:
        if "error" in kc:
            st.error(kc["error"])
        else:
            st.metric("Arabica (KC)", f"{kc['last_raw']:.2f} {kc.get('unit','')}")
            st.caption(f"≈ {kc['usdkg']:.3f} $/кг • {kc.get('source','?')} • as of {fmt_ts(kc.get('asof'))}")
    with colB:
        if "error" in rm:
            st.error(rm["error"])
        else:
            st.metric("Robusta (RM)", f"{rm['last_raw']:.2f} {rm.get('unit','')}")
            st.caption(f"≈ {rm['usdkg']:.3f} $/кг • {rm.get('source','?')} • as of {fmt_ts(rm.get('asof'))}")
    st.caption("Котировки ознакомительные. Для сделок сверяйте с брокером/биржей.")

# ---- Main form -------------------------------------------------------------
st.header("Калькулятор")

src = st.radio("Источник цены", ["Онлайн фьючерс (Stooq/Yahoo)", "Введу вручную"], horizontal=True)

col1, col2, col3 = st.columns(3)
with col1:
    instrument = st.selectbox("Инструмент", ["Arabica (KC.F)", "Robusta (RM.F)"])
with col2:
    if src == "Онлайн фьючерс (Stooq/Yahoo)":
        market_usdkg = (data.get("KC.F", {}).get("usdkg") if instrument.startswith("Arabica")
                        else data.get("RM.F", {}).get("usdkg"))
        if market_usdkg is None:
            st.warning("Нет данных рынка — переключитесь на ввод вручную.")
            base_usdkg = st.number_input("Базовая цена $/кг", min_value=0.0, value=3.000, step=0.001)
        else:
            st.text_input("Базовая цена $/кг (из рынка)", value=f"{market_usdkg:.4f}", disabled=True)
            base_usdkg = float(market_usdkg)
    else:
        base_usdkg = st.number_input("Базовая цена $/кг", min_value=0.0, value=3.000, step=0.001)
with col3:
    diff = st.number_input("Дифференциал $/кг (±)", value=0.000, step=0.010, help="Добавка/скидка к базовой цене")
effective_usdkg = base_usdkg + diff
st.caption(f"Эффективная цена: **{effective_usdkg:.4f} $/кг**")

# Containers
st.markdown("**Контейнер / вес**")
colw1, colw2, colw3 = st.columns(3)
with colw1:
    container = st.selectbox("Контейнер", ["20' (~19.2 т)", "40' (~26.0 т)", "Пользовательский"])
with colw2:
    if container.startswith("20"):
        weight_kg = st.number_input("Вес (кг)", min_value=1.0, value=19200.0, step=100.0)
    elif container.startswith("40"):
        weight_kg = st.number_input("Вес (кг)", min_value=1.0, value=26000.0, step=100.0)
    else:
        weight_kg = st.number_input("Вес (кг)", min_value=1.0, value=1000.0, step=100.0)
with colw3:
    incoterm = st.selectbox("Incoterm", ["FOB", "CFR", "CIF"], help="Упрощённая логика")

# Jurisdiction presets
st.markdown("**Юрисдикция (НДС/пошлина пресет)**")
jur_defaults = {
    "EAEU - Belarus": {"vat_rate": 0.20, "duty_rate": 0.00},
    "EAEU - Russia": {"vat_rate": 0.20, "duty_rate": 0.00},
    "UAE": {"vat_rate": 0.05, "duty_rate": 0.00},
}
jur_data = load_json("jurisdictions.json", jur_defaults)
colsj1, colsj2 = st.columns(2)
with colsj1:
    jname = st.selectbox("Страна/регион", list(jur_data.keys()))
with colsj2:
    j = jur_data[jname]
    vat_rate = st.number_input("Ставка НДС (0..1)", min_value=0.0, max_value=1.0,
                               value=float(j.get("vat_rate", 0.0)), step=0.01)
duty_rate = st.number_input("Пошлина (адвал., 0..1)", min_value=0.0, max_value=1.0,
                            value=float(j.get("duty_rate", 0.0)), step=0.01)
duty_sp_perkg = st.number_input("Пошлина (специф., $/кг)", min_value=0.0, value=0.0, step=0.01)

# Route presets (for CFR/CIF)
st.markdown("**Маршрут (для CFR/CIF пресеты фрахта/страховки)**")
routes_defaults = {
    "Santos → Riga": {"incoterms": ["CFR", "CIF"], "freight": 1800, "insurance": 80},
    "Santos → Dubai": {"incoterms": ["CFR", "CIF"], "freight": 1600, "insurance": 70},
    "Jebel Ali → Riyadh": {"incoterms": ["CFR", "CIF"], "freight": 600, "insurance": 40},
}
routes = load_json("routes.json", routes_defaults)
route_names = ["(не использ.)"] + list(routes.keys())
colr1, colr2, colr3 = st.columns(3)
with colr1:
    rsel = st.selectbox("Маршрут", route_names)
with colr2:
    if rsel != "(не использ.)" and incoterm in routes[rsel]["incoterms"]:
        freight = st.number_input("Фрахт (USD)", min_value=0.0, value=float(routes[rsel]["freight"]), step=10.0)
    else:
        freight = st.number_input("Фрахт (USD)", min_value=0.0, value=1800.0, step=10.0)
with colr3:
    if rsel != "(не использ.)" and incoterm in routes[rsel]["incoterms"]:
        insurance = st.number_input("Страховка (USD)", min_value=0.0, value=float(routes[rsel]["insurance"]), step=5.0)
    else:
        insurance = st.number_input("Страховка (USD)", min_value=0.0, value=80.0, step=5.0)

# Fees (single line starter)
st.markdown("**Локальные сборы (стартер)**")
feec1, feec2, feec3, feec4, _ = st.columns([2, 1, 1, 1, 1])
with feec1:
    fee_name = st.text_input("Название", value="Customs processing")
with feec2:
    fee_kind = st.selectbox("Тип", ["fixed", "percent"])
with feec3:
    if fee_kind == "fixed":
        fee_amount = st.number_input("Сумма ($)", min_value=0.0, value=25.0, step=1.0)
        fee_rate = 0.0; fee_base = "CV"
    else:
        fee_rate = st.number_input("Ставка (%)", min_value=0.0, value=0.0, step=0.1) / 100.0
        fee_base = st.selectbox("База", ["CV", "Goods", "CVPlusDuty"])
        fee_amount = 0.0
with feec4:
    fee_vb = st.checkbox("Включать в базу НДС", value=True)
fees = [{"name": fee_name, "kind": fee_kind, "amount": fee_amount,
         "rate": fee_rate, "base": fee_base, "vat_base": fee_vb}]

st.divider()

if st.button("Рассчитать", type="primary"):
    b = compute_quote(
        usd_per_kg=float(effective_usdkg),
        weight_kg=float(weight_kg),
        incoterm=incoterm,
        freight=float(freight),
        insurance=float(insurance),
        duty_rate=float(duty_rate),
        duty_sp_perkg=float(duty_sp_perkg),
        vat_rate=float(vat_rate),
        fees=fees,
    )

    colL, colR = st.columns([1, 1])
    with colL:
        st.subheader("Итог")
        st.metric("Landed total (USD)", f"{b['total']:.2f}")
        st.metric("$/кг", f"{b['per_kg']:.4f}")
        st.write(f"Incoterm: {incoterm} • Вес: {weight_kg:,.0f} кг • Регион: {jname}")
        st.write(f"Фрахт: {freight:.2f} • Страховка: {insurance:.2f} • Дифференциал: {diff:+.3f} $/кг")

    with colR:
        st.subheader("Разбор")
        st.dataframe(make_result_df(b), hide_index=True, use_container_width=True)
        st.write("Сборы:")
        st.dataframe(pd.DataFrame([{"Сбор": f["name"], "Сумма": f["amount"], "В базу НДС": f["vat_base"]} for f in b["fees"]]),
                     hide_index=True, use_container_width=True)

    # Export
    calc_params = {
        "Instrument": instrument, "Base USD/kg": f"{base_usdkg:.4f}",
        "Differential USD/kg": f"{diff:+.4f}", "Effective USD/kg": f"{effective_usdkg:.4f}",
        "Weight kg": weight_kg, "Incoterm": incoterm, "Freight USD": freight, "Insurance USD": insurance,
        "VAT rate": vat_rate, "Duty ad val.": duty_rate, "Duty specific $/kg": duty_sp_perkg,
        "Jurisdiction": jname, "Route": rsel,
    }
    excel_buf = export_excel(b, calc_params)
    st.download_button("⬇️ Скачать Excel", data=excel_buf, file_name="landed_cost.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    csv = make_result_df(b).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Скачать CSV (результат)", data=csv, file_name="result.csv", mime="text/csv")

    pdf_buf = export_pdf(b, calc_params)
    if pdf_buf:
        st.download_button("⬇️ Скачать PDF (сводка)", data=pdf_buf, file_name="summary.pdf", mime="application/pdf")
    else:
        st.info("PDF-экспорт недоступен (ReportLab не установлен).")

    st.subheader("Скопировать расчёт (текст)")
    txt = f"""Landed cost: {b['total']:.2f} USD ({b['per_kg']:.4f} USD/kg)
Instrument: {instrument}, Base: {base_usdkg:.4f}, Diff: {diff:+.4f}, Effective: {effective_usdkg:.4f}
Weight: {weight_kg:,.0f} kg, Incoterm: {incoterm}, Freight: {freight:.2f}, Insurance: {insurance:.2f}
VAT rate: {vat_rate}, Duty ad val.: {duty_rate}, Duty specific: {duty_sp_perkg}
Jurisdiction: {jname}, Route: {rsel}
CV: {b['customs_value']:.2f}, Duty total: {b['duty_total']:.2f}, Fees: {b['fees_total']:.2f}, VAT: {b['vat']:.2f}
"""
    st.text_area("Текст:", value=txt, height=160)
    st.info("Выделите текст и нажмите ⌘+C / Ctrl+C для копирования.")
else:
    st.info("Заполните поля и нажмите «Рассчитать».")
