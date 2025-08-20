# app.py — BeanRoute (EN/RU), safe boot, clean UI

import io, csv, time, random, json
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Tuple

import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

# ============================ BRAND / THEME =============================
APP_NAME   = "BeanRoute"
TAGLINE_EN = "Coffee imports, made clear."
TAGLINE_RU = "Импорт кофе — прозрачно и просто."
LOGO_PATH  = Path("logo_light.svg")  # если нет файла — покажем текстовый логотип

PRIMARY  = "#0FB5A8"
GRAPHITE = "#0F172A"

# ============================ GLOBAL SETTINGS ===========================
SAFE_BOOT        = True         # не ходим в сеть на холодном старте
STOOQ_CACHE_TTL  = 900          # 15 минут
REQ_TIMEOUT      = 2.5          # сек. таймаут HTTP-запросов
UA               = {"User-Agent": "Mozilla/5.0"}
STOOQ_DOMAINS    = ("https://stooq.com", "https://stooq.pl")

ARABICA_SYMBOL   = "KC.F"       # ¢/lb
ROBUSTA_SYMBOL   = "RM.F"       # USD/tonne

# ============================ PAGE CONFIG ===============================
st.set_page_config(page_title=f"{APP_NAME}", page_icon="☕", layout="wide")

# ============================== I18N ===================================
def T(key: str) -> str:
    """Simple i18n with safe fallback to EN/keys."""
    lang = st.session_state.get("lang", "en")
    L = {
        "en": {
            "brand": APP_NAME,
            "tagline": TAGLINE_EN,
            "language": "Language",
            "english": "English",
            "russian": "Русский",
            "market_title": "Market — Stooq (safe boot)",
            "refresh": "Refresh quotes",
            "last_check": "Last check",
            "arabica": "Arabica (KC.F)",
            "robusta": "Robusta (RM.F)",
            "no_quotes": "Quotes are not available yet — try to refresh.",
            "approx": "≈",
            "usdkg": "$/kg",
            "source": "Source",
            "as_of": "as of",
            "calc": "Calculator",
            "price_source": "Price source",
            "online": "Online (Stooq: KC.F / RM.F)",
            "manual": "Manual input",
            "instrument": "Instrument",
            "base_price": "Base price $/kg",
            "base_from_kc": "Base price $/kg (from KC.F)",
            "base_from_rm": "Base price $/kg (from RM.F)",
            "diff": "Differential $/kg (±)",
            "effective_price": "Effective price",
            "container_weight": "Container / weight",
            "container": "Container",
            "weight": "Weight (kg)",
            "incoterm": "Incoterm",
            "jurisdiction": "Jurisdiction (VAT/duty preset)",
            "country_region": "Country/region",
            "vat_rate": "VAT rate (0..1)",
            "duty_ad": "Duty (ad val., 0..1)",
            "duty_sp": "Duty (specific, $/kg)",
            "route": "Route (CFR/CIF freight & insurance)",
            "route_name": "Route",
            "freight": "Freight (USD)",
            "insurance": "Insurance (USD)",
            "fees": "Local fees (starter)",
            "fee_name": "Name",
            "fee_type": "Type",
            "fee_amount": "Amount ($)",
            "fee_rate": "Rate (%)",
            "fee_base": "Base",
            "fee_in_vat": "Include in VAT base",
            "btn_calc": "Calculate",
            "result": "Result",
            "landed_total": "Landed total (USD)",
            "per_kg": "$/kg",
            "summary": "Breakdown",
            "fill_and_calc": "Fill the fields and press “Calculate”.",
            "auto_refresh": "Auto refresh quotes (15 min)",
        },
        "ru": {
            "brand": APP_NAME,
            "tagline": TAGLINE_RU,
            "language": "Язык",
            "english": "English",
            "russian": "Русский",
            "market_title": "Рынок — Stooq (безопасный запуск)",
            "refresh": "Обновить котировки",
            "last_check": "Последняя проверка",
            "arabica": "Арабика (KC.F)",
            "robusta": "Робуста (RM.F)",
            "no_quotes": "Котировки пока недоступны — попробуйте обновить.",
            "approx": "≈",
            "usdkg": "$/кг",
            "source": "Источник",
            "as_of": "на",
            "calc": "Калькулятор",
            "price_source": "Источник цены",
            "online": "Онлайн (Stooq: KC.F / RM.F)",
            "manual": "Введу вручную",
            "instrument": "Инструмент",
            "base_price": "Базовая цена $/кг",
            "base_from_kc": "Базовая цена $/кг (из KC.F)",
            "base_from_rm": "Базовая цена $/кг (из RM.F)",
            "diff": "Дифференциал $/кг (±)",
            "effective_price": "Эффективная цена",
            "container_weight": "Контейнер / вес",
            "container": "Контейнер",
            "weight": "Вес (кг)",
            "incoterm": "Incoterm",
            "jurisdiction": "Юрисдикция (пресет НДС/пошлина)",
            "country_region": "Страна/регион",
            "vat_rate": "Ставка НДС (0..1)",
            "duty_ad": "Пошлина (адвал., 0..1)",
            "duty_sp": "Пошлина (специф., $/кг)",
            "route": "Маршрут (CFR/CIF фрахт и страховка)",
            "route_name": "Маршрут",
            "freight": "Фрахт (USD)",
            "insurance": "Страховка (USD)",
            "fees": "Локальные сборы (стартер)",
            "fee_name": "Название",
            "fee_type": "Тип",
            "fee_amount": "Сумма ($)",
            "fee_rate": "Ставка (%)",
            "fee_base": "База",
            "fee_in_vat": "Включать в базу НДС",
            "btn_calc": "Рассчитать",
            "result": "Итог",
            "landed_total": "Landed total (USD)",
            "per_kg": "$/кг",
            "summary": "Разбор",
            "fill_and_calc": "Заполните поля и нажмите «Рассчитать».",
            "auto_refresh": "Авто-обновление котировок (15 мин)",
        },
    }
    return L.get(lang, {}).get(key) or L["en"].get(key) or key


# ============================ UTILITIES ================================
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M UTC")

def kc_centslb_to_usdkg(x: float) -> float:
    """¢/lb -> $/kg"""
    return (float(x) / 100.0) / 0.45359237

def rm_usdt_to_usdkg(x: float) -> float:
    """USD/tonne -> $/kg"""
    return float(x) / 1000.0

def _http_get_text(url: str, params: dict | None = None, timeout: float = REQ_TIMEOUT) -> Tuple[Optional[str], Dict]:
    meta = {"url": url, "params": params, "status": None, "ctype": None, "err": None}
    try:
        r = requests.get(url, headers=UA, params=params, timeout=timeout)
        meta["status"] = r.status_code
        meta["ctype"] = r.headers.get("Content-Type", "")
        if not r.ok:
            return None, meta
        text = r.text
        # фильтр на случай HTML/капчи
        if not text or "<html" in text.lower():
            return None, meta
        return text, meta
    except Exception as e:
        meta["err"] = f"{type(e).__name__}: {e}"
        return None, meta

def _try_domains(path: str, params: dict, retries: int = 3) -> Tuple[Optional[str], Dict]:
    last_meta: Dict = {}
    for base in STOOQ_DOMAINS:
        for attempt in range(retries):
            text, meta = _http_get_text(base + path, params=params)
            if text:
                meta["domain"] = base
                return text, meta
            last_meta = {**meta, "domain": base, "attempt": attempt + 1}
            time.sleep(0.2 + random.random() * 0.5)
    return None, last_meta

def _parse_snapshot(text: str, expect_symbol: str) -> Optional[Dict]:
    """Parse /q/l CSV: Symbol,Date,Time,Open,High,Low,Close,Volume."""
    reader = csv.DictReader(io.StringIO(text))
    row = next(reader, None)
    if not row: return None
    symbol = (row.get("Symbol") or "").strip().upper()
    if symbol != expect_symbol.upper():
        return None
    try:
        close = float(row["Close"])
    except Exception:
        return None
    date = (row.get("Date") or "").strip()
    time_ = (row.get("Time") or "").strip()
    if symbol.startswith("KC"):
        unit, usdkg = "¢/lb", kc_centslb_to_usdkg(close)
    elif symbol.startswith("RM"):
        unit, usdkg = "USD/t", rm_usdt_to_usdkg(close)
    else:
        return None
    return {"last_raw": close, "unit": unit, "usdkg": usdkg,
            "asof": f"{date} {time_} (snapshot)", "source": "Stooq /q/l"}

def _parse_eod(text: str, expect_symbol: str) -> Optional[Dict]:
    """Parse /q/d/l CSV: Date,Open,High,Low,Close,Volume."""
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows: return None
    last = rows[-1]
    try:
        close = float(last["Close"])
    except Exception:
        return None
    date = (last.get("Date") or "").strip()
    if expect_symbol.upper().startswith("KC"):
        unit, usdkg = "¢/lb", kc_centslb_to_usdkg(close)
    elif expect_symbol.upper().startswith("RM"):
        unit, usdkg = "USD/t", rm_usdt_to_usdkg(close)
    else:
        return None
    return {"last_raw": close, "unit": unit, "usdkg": usdkg,
            "asof": f"{date} (EOD)", "source": "Stooq /q/d/l"}

@st.cache_data(ttl=STOOQ_CACHE_TTL)
def stooq_latest(symbol: str, seed: int = 0, debug: bool = False) -> Dict:
    """Snapshot first, fallback to EOD."""
    sym = symbol.upper()
    snap_txt, snap_meta = _try_domains("/q/l/", params={"s": sym.lower(), "f": "sd2t2ohlcv", "h": "", "e": "csv"})
    if snap_txt:
        parsed = _parse_snapshot(snap_txt, sym)
        if parsed:
            if debug: parsed["_debug"] = {"endpoint": "snapshot", **snap_meta}
            return parsed

    eod_txt, eod_meta = _try_domains("/q/d/l/", params={"s": sym.lower(), "i": "d"})
    if eod_txt:
        parsed = _parse_eod(eod_txt, sym)
        if parsed:
            if debug: parsed["_debug"] = {"endpoint": "eod", **eod_meta}
            return parsed

    meta = snap_meta if snap_meta else eod_meta
    raise RuntimeError(f"Stooq not available for {sym} (status={meta.get('status')}, ctype={meta.get('ctype')}, err={meta.get('err')})")

# ======================== QUOTE RENDER HELPERS =========================
def metric_block(title: str, data: dict):
    """Draws metric or friendly message if no data."""
    val = data.get("last_raw")
    unit = data.get("unit", "")
    if val is not None:
        st.metric(title, f"{val:.2f} {unit}")
        usdkg = data.get("usdkg")
        src = data.get("source", "")
        asof = data.get("asof", "")
        if usdkg is not None:
            st.caption(f"{T('approx')} {usdkg:.3f} {T('usdkg')} • {T('source')}: {src} • {T('as_of')} {asof}")
        else:
            st.caption(f"{T('source')}: {src} • {T('as_of')} {asof}")
    else:
        st.error(T("no_quotes"))

# ============================ CALCULATOR ================================
def compute_customs_value(incoterm: str, goods_value: float, freight: float, insurance: float) -> float:
    inc = incoterm.upper()
    if inc in {"FOB", "EXW"}: return goods_value + freight + insurance
    if inc == "CFR":          return goods_value + insurance
    return goods_value  # CIF включает фрахт и страховку

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
    return pd.DataFrame(rows, columns=["Metric","Value","Unit"])

def export_excel(b, calc_params):
    df_main = make_result_df(b)
    df_fees = pd.DataFrame([{"Fee": f["name"], "Amount": f["amount"], "In VAT base": f["vat_base"]} for f in b["fees"]])
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame([calc_params]).to_excel(writer, index=False, sheet_name="Input")
        df_main.to_excel(writer, index=False, sheet_name="Result")
        df_fees.to_excel(writer, index=False, sheet_name="Fees")
    buf.seek(0)
    return buf

def export_pdf(b, calc_params):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 2*cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, y, f"{APP_NAME} — Summary")
    y -= 1*cm
    c.setFont("Helvetica", 10)
    for k,v in calc_params.items():
        c.drawString(2*cm, y, f"{k}: {v}")
        y -= 0.5*cm
        if y < 2*cm: c.showPage(); y = h - 2*cm; c.setFont("Helvetica", 10)
    y -= 0.5*cm
    lines = [
        ("Customs value (CV)", b["customs_value"]),
        ("Duty ad val.", b["duty_ad"]),
        ("Duty specific", b["duty_sp"]),
        ("Duty total", b["duty_total"]),
        ("Fees total", b["fees_total"]),
        ("VAT base", b["vat_base"]),
        ("VAT", b["vat"]),
        ("Landed total", b["total"]),
        ("Per kg", b["per_kg"]),
    ]
    c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, "Result"); y -= 0.7*cm; c.setFont("Helvetica", 10)
    for name,val in lines:
        c.drawString(2*cm, y, f"{name}: {val:,.4f} USD"); y -= 0.5*cm
        if y < 2*cm: c.showPage(); y = h - 2*cm; c.setFont("Helvetica", 10)
    c.showPage(); c.save(); buf.seek(0)
    return buf

# ============================== HEADER =================================
# init session
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "refresh_seed" not in st.session_state:
    st.session_state["refresh_seed"] = 0
if "net_ok" not in st.session_state:
    st.session_state["net_ok"] = not SAFE_BOOT  # если SAFE_BOOT=True, ждём нажатия кнопки
if "last_check" not in st.session_state:
    st.session_state["last_check"] = None
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = False

# top bar
col_l, col_c, col_r = st.columns([1,1,1])
with col_l:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)
    else:
        st.markdown(
            f"<div style='font-weight:800;font-size:28px;color:{GRAPHITE}'>{APP_NAME}</div>",
            unsafe_allow_html=True
        )
with col_c:
    st.markdown(
        f"<div style='text-align:center;font-weight:800;font-size:34px;color:{GRAPHITE}'>{APP_NAME}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='text-align:center;color:#64748B'>{T('tagline')}</div>",
        unsafe_allow_html=True
    )
with col_r:
    st.caption(f"🌐 {T('language')}")
    lang_choice = st.radio("", ["en","ru"],
                           index=0 if st.session_state["lang"]=="en" else 1,
                           horizontal=True, label_visibility="collapsed",
                           format_func=lambda x: T("english") if x=="en" else T("russian"))
    st.session_state["lang"] = lang_choice

st.markdown("<hr style='margin-top:6px;margin-bottom:6px'/>", unsafe_allow_html=True)

# =============================== SIDEBAR ================================
with st.sidebar:
    st.subheader(T("market_title"))
    if st.button(f"↻ {T('refresh')}"):
        st.session_state["refresh_seed"] += 1
        st.session_state["net_ok"] = True
        st.session_state["last_check"] = utc_now_str()

    st.toggle(T("auto_refresh"), key="auto_refresh")
    if st.session_state["auto_refresh"]:
        # каждые 15 минут (900_000 мс)
        st.experimental_rerun = st.autorefresh(interval=900_000, key="auto_r")

    last = st.session_state.get("last_check")
    st.caption(f"{T('last_check')}: {last or '—'}")

    # котировки только если разрешена сеть (safe boot)
    kc, rm = {}, {}
    if st.session_state["net_ok"]:
        try:
            kc = stooq_latest(ARABICA_SYMBOL, seed=st.session_state["refresh_seed"], debug=False)
        except Exception:
            kc = {}
        try:
            rm = stooq_latest(ROBUSTA_SYMBOL, seed=st.session_state["refresh_seed"], debug=False)
        except Exception:
            rm = {}
    metric_block(T("arabica"), kc)
    metric_block(T("robusta"), rm)
    st.caption("Snapshot у Stooq может быть с задержкой; если недоступен — берём EOD.")

# ============================== CALCULATOR UI ===========================
st.header(T("calc"))

src = st.radio(T("price_source"), [T("online"), T("manual")], horizontal=True)

col1, col2, col3 = st.columns(3)
with col1:
    instrument = st.selectbox(T("instrument"), [T("arabica"), T("robusta")])

with col2:
    if src == T("online"):
        if instrument.startswith("Arabica") or instrument.startswith("Арабика"):
            market = kc.get("usdkg")
            if market is None:
                st.warning(T("no_quotes"))
                base_usdkg = st.number_input(T("base_price"), min_value=0.0, value=3.000, step=0.001)
            else:
                st.text_input(T("base_from_kc"), value=f"{market:.4f}", disabled=True)
                base_usdkg = float(market)
        else:
            market = rm.get("usdkg")
            if market is None:
                st.warning(T("no_quotes"))
                base_usdkg = st.number_input(T("base_price"), min_value=0.0, value=3.000, step=0.001)
            else:
                st.text_input(T("base_from_rm"), value=f"{market:.4f}", disabled=True)
                base_usdkg = float(market)
    else:
        base_usdkg = st.number_input(T("base_price"), min_value=0.0, value=3.000, step=0.001)

with col3:
    diff = st.number_input(T("diff"), value=0.000, step=0.010)

effective_usdkg = base_usdkg + diff
st.caption(f"{T('effective_price')}: **{effective_usdkg:.4f} {T('usdkg')}**")

# контейнер / вес
st.markdown(f"**{T('container_weight')}**")
colw1, colw2, colw3 = st.columns(3)
with colw1:
    container = st.selectbox(T("container"), ["20' (~19.2 t)", "40' (~26.0 t)", "Custom"])
with colw2:
    if container.startswith("20"):
        weight_kg = st.number_input(T("weight"), min_value=1.0, value=19200.0, step=100.0)
    elif container.startswith("40"):
        weight_kg = st.number_input(T("weight"), min_value=1.0, value=26000.0, step=100.0)
    else:
        weight_kg = st.number_input(T("weight"), min_value=1.0, value=1000.0, step=100.0)
with colw3:
    incoterm = st.selectbox(T("incoterm"), ["FOB","CFR","CIF"])

# юрисдикции (простые пресеты)
st.markdown(f"**{T('jurisdiction')}**")
jur_presets = {
    "EAEU - Belarus": {"vat_rate": 0.20, "duty_rate": 0.00},
    "EAEU - Russia":  {"vat_rate": 0.20, "duty_rate": 0.00},
    "UAE":            {"vat_rate": 0.05, "duty_rate": 0.00},
}
colsj1, colsj2 = st.columns(2)
with colsj1:
    jname = st.selectbox(T("country_region"), list(jur_presets.keys()))
with colsj2:
    j = jur_presets[jname]
    vat_rate = st.number_input(T("vat_rate"), min_value=0.0, max_value=1.0, value=float(j.get("vat_rate", 0.0)), step=0.01)
duty_rate = st.number_input(T("duty_ad"), min_value=0.0, max_value=1.0, value=float(j.get("duty_rate", 0.0)), step=0.01)
duty_sp_perkg = st.number_input(T("duty_sp"), min_value=0.0, value=0.0, step=0.01)

# маршруты (пресеты для CFR/CIF)
st.markdown(f"**{T('route')}**")
route_presets = {
    "Santos → Riga":     {"incoterms": ["CFR","CIF"], "freight": 1800, "insurance": 80},
    "Santos → Dubai":    {"incoterms": ["CFR","CIF"], "freight": 1600, "insurance": 70},
    "Jebel Ali → Riyadh":{"incoterms": ["CFR","CIF"], "freight": 600,  "insurance": 40},
}
route_names = ["(none)"] + list(route_presets.keys())
colr1, colr2, colr3 = st.columns(3)
with colr1:
    rsel = st.selectbox(T("route_name"), route_names)
with colr2:
    if rsel != "(none)" and incoterm in route_presets[rsel]["incoterms"]:
        freight = st.number_input(T("freight"), min_value=0.0, value=float(route_presets[rsel]["freight"]), step=10.0)
    else:
        freight = st.number_input(T("freight"), min_value=0.0, value=1800.0, step=10.0)
with colr3:
    if rsel != "(none)" and incoterm in route_presets[rsel]["incoterms"]:
        insurance = st.number_input(T("insurance"), min_value=0.0, value=float(route_presets[rsel]["insurance"]), step=5.0)
    else:
        insurance = st.number_input(T("insurance"), min_value=0.0, value=80.0, step=5.0)

# fees (одна строка — стартер)
st.markdown(f"**{T('fees')}**")
feec1, feec2, feec3, feec4, _ = st.columns([2,1,1,1,1])
with feec1:
    fee_name = st.text_input(T("fee_name"), value="Customs processing")
with feec2:
    fee_kind = st.selectbox(T("fee_type"), ["fixed","percent"])
with feec3:
    if fee_kind == "fixed":
        fee_amount = st.number_input(T("fee_amount"), min_value=0.0, value=25.0, step=1.0)
        fee_rate = 0.0; fee_base = "CV"
    else:
        fee_rate = st.number_input(T("fee_rate"), min_value=0.0, value=0.0, step=0.1) / 100.0
        fee_base = st.selectbox(T("fee_base"), ["CV","Goods","CVPlusDuty"])
        fee_amount = 0.0
with feec4:
    fee_vb = st.checkbox(T("fee_in_vat"), value=True)
fees = [{"name": fee_name, "kind": fee_kind, "amount": fee_amount,
         "rate": fee_rate, "base": fee_base, "vat_base": fee_vb}]

st.divider()

if st.button(T("btn_calc"), type="primary"):
    b = compute_quote(
        usd_per_kg=float(effective_usdkg),
        weight_kg=float(weight_kg),
        incoterm=incoterm,
        freight=float(freight),
        insurance=float(insurance),
        duty_rate=float(duty_rate),
        duty_sp_perkg=float(duty_sp_perkg),
        vat_rate=float(vat_rate),
        fees=fees
    )
    colL, colR = st.columns([1,1])
    with colL:
        st.subheader(T("result"))
        st.metric(T("landed_total"), f"{b['total']:.2f}")
        st.metric(T("per_kg"), f"{b['per_kg']:.4f}")
        st.write(f"Incoterm: {incoterm} • {T('weight')}: {weight_kg:,.0f} kg • {T('country_region')}: {jname}")
        st.write(f"{T('freight')}: {freight:.2f} • {T('insurance')}: {insurance:.2f} • {T('diff')}: {diff:+.3f} {T('usdkg')}")
    with colR:
        st.subheader(T("summary"))
        st.dataframe(make_result_df(b), hide_index=True, use_container_width=True)

    # экспорт
    calc_params = {
        "Instrument": instrument,
        "Base USD/kg": f"{base_usdkg:.4f}",
        "Differential USD/kg": f"{diff:+.4f}",
        "Effective USD/kg": f"{effective_usdkg:.4f}",
        "Weight kg": weight_kg,
        "Incoterm": incoterm,
        "Freight USD": freight,
        "Insurance USD": insurance,
        "VAT rate": vat_rate,
        "Duty ad val.": duty_rate,
        "Duty specific $/kg": duty_sp_perkg,
        "Jurisdiction": jname,
        "Route": rsel,
    }
    excel_buf = export_excel(b, calc_params)
    st.download_button("⬇️ Excel", data=excel_buf, file_name="landed_cost.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    csv = make_result_df(b).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV", data=csv, file_name="result.csv", mime="text/csv")
    pdf_buf = export_pdf(b, calc_params)
    st.download_button("⬇️ PDF", data=pdf_buf, file_name="summary.pdf", mime="application/pdf")
else:
    st.info(T("fill_and_calc"))
