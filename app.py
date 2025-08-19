import io, re, json, os
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict

import requests
import pandas as pd
import streamlit as st

# ---------- PDF (опционально) ----------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ---------- PAGE ----------
st.set_page_config(page_title="Coffee Landed Cost — MVP", page_icon="☕", layout="wide")
st.title("☕ Coffee Landed Cost — MVP")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ---------- UTILS ----------
def fmt_ts(ts: Optional[int]) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M UTC") if ts else "n/a"

def load_json(filename: str, default: dict) -> dict:
    p = DATA_DIR / filename
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# ---------- CONVERSIONS ----------
def arabica_centlb_to_usd_per_kg(cents_per_lb: float) -> float:
    return (float(cents_per_lb) / 100.0) / 0.45359237

def robusta_usd_per_tonne_to_usd_per_kg(usd_per_tonne: float) -> float:
    return float(usd_per_tonne) / 1000.0

# ---------- HTTP with retries ----------
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

def _retry_obj():
    try:
        return Retry(total=3, backoff_factor=0.6,
                     status_forcelist=[429, 500, 502, 503, 504],
                     allowed_methods=frozenset(["GET"]), raise_on_status=False)
    except TypeError:  # urllib3 v1
        return Retry(total=3, backoff_factor=0.6,
                     status_forcelist=[429, 500, 502, 503, 504],
                     method_whitelist=frozenset(["GET"]), raise_on_status=False)

def _http_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=_retry_obj()))
    s.headers.update(HEADERS)
    return s

def _set_yahoo_state(ok: bool, msg: str = ""):
    st.session_state["last_yahoo_ok"] = ok
    if not ok:
        st.session_state["last_yahoo_err"] = msg

# ---------- STOOQ (CSV) ----------
@st.cache_data(ttl=1800)
def fetch_stooq_csv(symbol: str, interval: str = "d") -> pd.DataFrame:
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
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://stooq.com/q/f/?s={base_symbol}"
    r = requests.get(url, headers=headers, timeout=10)
    if not r.ok or not r.text:
        return None
    m = re.search(r"RM[A-Z]\d{2}\.F", r.text, re.IGNORECASE)
    return m.group(0).upper() if m else None

# ---------- YAHOO (chart & quote) ----------
@st.cache_data(ttl=900)
def fetch_yahoo_intraday_last(symbol: str, interval: str = "1m", range_: str = "1d") -> Tuple[float, Optional[int]]:
    sess = _http_session()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = sess.get(url, params={"range": range_, "interval": interval}, timeout=10)
    if r.status_code >= 400:
        _set_yahoo_state(False, f"chart {r.status_code} for {symbol}")
        r.raise_for_status()
    _set_yahoo_state(True, "")
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
        last_ts = j["meta"].get("regularMarketTime")
    return last, last_ts

@st.cache_data(ttl=900)
def fetch_yahoo_quote_single(symbol: str) -> Optional[Dict]:
    """v7/quote для одного символа (упрощаем, чтобы не бить пачками)."""
    sess = _http_session()
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    r = sess.get(url, params={"symbols": symbol}, timeout=10)
    if r.status_code >= 400:
        _set_yahoo_state(False, f"quote {r.status_code} for {symbol}")
        r.raise_for_status()
    _set_yahoo_state(True, "")
    res = r.json().get("quoteResponse", {}).get("result", [])
    if not res:
        return None
    row = res[0]
    return {
        "price": row.get("regularMarketPrice"),
        "ts": row.get("regularMarketTime"),
        "symbol": row.get("symbol")
    }

# ---------- KC (Arabica) ----------
MONTHS = [(3, "H"), (5, "K"), (7, "N"), (9, "U"), (12, "Z")]  # Mar/May/Jul/Sep/Dec

def next_kc_contracts(n: int = 8, now: Optional[datetime] = None) -> List[str]:
    now = now or datetime.now(timezone.utc)
    y, m = now.year, now.month
    start_idx = None
    for i, (mm, _) in enumerate(MONTHS):
        if m <= mm:
            start_idx = i; break
    if start_idx is None:
        start_idx = 0
        y += 1
    out, year, idx = [], y, start_idx
    for _ in range(n):
        mm, code = MONTHS[idx]
        out.append(f"KC{code}{year % 100:02d}")
        idx += 1
        if idx >= len(MONTHS):
            idx = 0; year += 1
    return out

@st.cache_data(ttl=900)
def kc_contract_price(contract: str, seed: int = 0) -> Tuple[Optional[float], Optional[int], str]:
    """
    Возвращает (¢/lb, ts, source) для KC-контракта (напр. 'KCU25').
    Каскад: Yahoo quote -> Yahoo chart -> Stooq series (KCU25.F).
    """
    sym_quote = f"{contract}=F"
    try:
        q = fetch_yahoo_quote_single(sym_quote)
        if q and q.get("price") is not None:
            return float(q["price"]), int(q.get("ts") or 0) or None, f"Yahoo quote ({sym_quote})"
    except Exception:
        pass
    try:
        p, ts = fetch_yahoo_intraday_last(sym_quote, interval="1m", range_="1d")
        if p is not None:
            return float(p), ts, f"Yahoo chart ({sym_quote})"
    except Exception:
        pass
    try:
        stooq_sym = f"{contract}.F"  # пример: KCU25.F
        df = fetch_stooq_csv(stooq_sym)
        price = float(df.iloc[-1]["close"])  # у KC на Stooq — тоже ¢/lb
        return price, None, f"Stooq ({stooq_sym})"
    except Exception:
        pass
    return None, None, "N/A"

# ---------- ROBUSTA ----------
@st.cache_data(ttl=900)
def get_robusta_price(symbol_order: Tuple[str, ...], seed: int = 0) -> dict:
    """Каскад: Yahoo (RC=F → RM=F) → Stooq rm.f → Stooq(front)."""
    # 1) Yahoo
    for sym in symbol_order:
        s = sym.strip()
        if not s:
            continue
        try:
            p, ts = fetch_yahoo_intraday_last(s, "1m", "1d")
            return {
                "last_raw": float(p), "unit": "USD/t",
                "usdkg": robusta_usd_per_tonne_to_usd_per_kg(p),
                "source": f"Yahoo 1m ({s})", "asof": ts
            }
        except Exception:
            continue
    # 2) Stooq continuous rm.f
    try:
        rm = fetch_stooq_csv("rm.f")
        last = float(rm.iloc[-1]["close"])
        return {"last_raw": last, "unit": "USD/t",
                "usdkg": robusta_usd_per_tonne_to_usd_per_kg(last),
                "source": "Stooq (rm.f)", "asof": None}
    except Exception:
        pass
    # 3) Stooq front-month series (напр., RMU25.F)
    try:
        front = stooq_front_series_symbol("rm.f")
        if front:
            rm_series = fetch_stooq_csv(front)
            last = float(rm_series.iloc[-1]["close"])
            return {"last_raw": last, "unit": "USD/t",
                    "usdkg": robusta_usd_per_tonne_to_usd_per_kg(last),
                    "source": f"Stooq ({front})", "asof": None}
    except Exception:
        pass
    return {"error": "Robusta: no data from Yahoo/Stooq"}

# ---------- CALC CORE ----------
def compute_customs_value(incoterm: str, goods_value: float, freight: float, insurance: float) -> float:
    inc = incoterm.upper()
    if inc in {"FOB", "EXW"}: return goods_value + freight + insurance
    if inc == "CFR":          return goods_value + insurance
    return goods_value  # CIF

def compute_quote(usd_per_kg, weight_kg, incoterm, freight, insurance,
                  duty_rate, duty_sp_perkg, vat_rate, fees):
    goods_value = usd_per_kg * weight_kg
    cv = compute_customs_value(incoterm, goods_value, freight, insurance)
    duty_ad = cv * duty_rate
    duty_sp = duty_sp_perkg * weight_kg
    duty_total = duty_ad + duty_sp

    def fee_amt(f):
        if f["kind"] == "fixed": return float(f.get("amount", 0))
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
    if not REPORTLAB_OK: return None
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

# ---------- SIDEBAR ----------
with st.sidebar:
    st.subheader("Arabica — контракт (Yahoo/Stooq)")
    if "refresh_seed" not in st.session_state: st.session_state["refresh_seed"] = 0
    if st.button("↻ Обновить котировки"): st.session_state["refresh_seed"] += 1

    # Выбираем ТОЛЬКО один контракт — меньше запросов
    options = next_kc_contracts(8)
    contract = st.selectbox("Контракт KC", options, index=0, key="kc_pick")

    price_cents, kc_ts, kc_src = kc_contract_price(contract, seed=st.session_state["refresh_seed"])
    if price_cents is not None:
        kc_usdkg = arabica_centlb_to_usd_per_kg(price_cents)
        st.metric("Arabica (KC)", f"{price_cents:.2f} ¢/lb")
        st.caption(f"≈ {kc_usdkg:.3f} $/кг • {kc_src} • as of {fmt_ts(kc_ts)}")
    else:
        kc_usdkg = None
        st.warning("Арабика: источник временно недоступен (ни Yahoo, ни Stooq).")

    st.divider()
    st.subheader("Robusta — котировки")
    rm_main = st.text_input("Yahoo symbol (1)", value="RC=F")
    rm_alt  = st.text_input("Yahoo symbol (2, запасной)", value="RM=F")
    rm_data = get_robusta_price((rm_main, rm_alt), seed=st.session_state["refresh_seed"])

    if "error" in rm_data:
        st.warning("Robusta: временно нет онлайна. Попробуйте обновить.")
    else:
        st.metric("Robusta (RM)", f"{rm_data['last_raw']:.2f} {rm_data.get('unit','')}")
        st.caption(f"≈ {rm_data['usdkg']:.3f} $/кг • {rm_data.get('source','?')} • as of {fmt_ts(rm_data.get('asof'))}")

    # компактный лог последней ошибки Yahoo
    if not st.session_state.get("last_yahoo_ok", True):
        st.caption(f"Yahoo warn: {st.session_state.get('last_yahoo_err','')}")

    st.caption("Котировки ознакомительные. Для сделок сверяйте с брокером/биржей.")

# ---------- MAIN FORM ----------
st.header("Калькулятор")

src = st.radio("Источник цены", ["Онлайн фьючерс (контракт KC/RC)", "Введу вручную"], horizontal=True)

col1, col2, col3 = st.columns(3)
with col1:
    instrument = st.selectbox("Инструмент", ["Arabica (KC.F)", "Robusta (RM.F)"])

with col2:
    if src == "Онлайн фьючерс (контракт KC/RC)":
        if instrument.startswith("Arabica"):
            if kc_usdkg is not None:
                st.text_input("Базовая цена $/кг (из выбранного контракта)",
                              value=f"{kc_usdkg:.4f}", disabled=True)
                base_usdkg = float(kc_usdkg)
            else:
                st.warning("Нет данных по арабике — введите вручную.")
                base_usdkg = st.number_input("Базовая цена $/кг", min_value=0.0, value=3.000, step=0.001)
        else:
            rmk = rm_data.get("usdkg") if "error" not in rm_data else None
            if rmk is None:
                st.warning("Нет данных по робусте — введите вручную.")
                base_usdkg = st.number_input("Базовая цена $/кг", min_value=0.0, value=3.000, step=0.001)
            else:
                st.text_input("Базовая цена $/кг (из рынка)", value=f"{rmk:.4f}", disabled=True)
                base_usdkg = float(rmk)
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
    "EAEU - Russia":  {"vat_rate": 0.20, "duty_rate": 0.00},
    "UAE":            {"vat_rate": 0.05, "duty_rate": 0.00},
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

# Route presets
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
else:
    st.info("Заполните поля и нажмите «Рассчитать».")
