# app.py — BeanRoute (EN/RU), no-sidebar UI, safe auto-refresh, Stooq KC.F/RM.F

# ---------- Imports ----------
import io, csv, time, random, json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timezone

import requests
import pandas as pd
import streamlit as st

# ---------- Brand / Theme ----------
APP_NAME = "BeanRoute"
TAGLINE_EN = "Coffee imports, made clear."
TAGLINE_RU = "Импорт кофе — прозрачно и просто."

LOGO_PATH = Path("logo_light.svg")  # если файла нет — покажем текст
PRIMARY = "#0F85A8"
GRAPHITE = "#0F172A"

# ---------- Global settings ----------
SAFE_BOOT = True                   # не ходим в сеть на холодном старте
STOOQ_CACHE_TTL = 900              # кэш 15 минут
REQ_TIMEOUT = 12.0                 # секунды таймаут на HTTP
UA = {"User-Agent": "Mozilla/5.0"}
STOOQ_DOMAINS = ("https://stooq.com", "https://stooq.pl")

ARABICA_SYMBOL = "KC.F"            # ¢/lb (continuous)
ROBUSTA_SYMBOL = "RM.F"            # USD/tonne (continuous)

# ---------- i18n ----------
st.set_page_config(page_title=f"{APP_NAME}", page_icon="☕", layout="wide")

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

I18N = {
    "en": {
        "lang_label": "Language",
        "tagline": TAGLINE_EN,
        "refresh": "Refresh quotes",
        "auto_refresh": "Auto refresh quotes (15 min)",
        "last_check": "Last check",
        "market_title": "Market — Stooq (safe boot)",
        "arabica": "Arabica (KC.F)",
        "robusta": "Robusta (RM.F)",
        "approx": "≈",
        "usdkg": "$/kg",
        "source": "Source",
        "asof": "as of",
        "no_quotes_try_refresh": "Quotes are not available yet — try to refresh.",
        "calculator": "Calculator",
        "price_source": "Price source",
        "online_stooq": "Online (Stooq: KC.F / RM.F)",
        "manual_input": "Manual input",
        "instrument": "Instrument",
        "base_price": "Base price $/kg",
        "base_price_from_market": "Base price $/kg (from market)",
        "diff_per_kg": "Differential $/kg (±)",
        "diff_help": "Positive adds, negative subtracts from base price",
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
        "route": "Route (CFR/CIF freight/insurance presets)",
        "freight": "Freight (USD)",
        "insurance": "Insurance (USD)",
        "local_fees": "Local fees (starter)",
        "fee_name": "Name",
        "fee_type": "Type",
        "fee_fixed": "fixed",
        "fee_percent": "percent",
        "fee_amount": "Amount ($)",
        "fee_rate": "Rate (%)",
        "fee_base": "Base",
        "fee_vat_base": "Include in VAT base",
        "compute": "Compute",
        "result": "Result",
        "landed_total": "Landed total (USD)",
        "usd_per_kg": "$/kg",
        "fill_and_compute": "Fill the fields and press “Compute”.",
    },
    "ru": {
        "lang_label": "Язык",
        "tagline": TAGLINE_RU,
        "refresh": "Обновить котировки",
        "auto_refresh": "Авто-обновление котировок (15 мин)",
        "last_check": "Последняя проверка",
        "market_title": "Рынок — Stooq (безопасный запуск)",
        "arabica": "Арабика (KC.F)",
        "robusta": "Робуста (RM.F)",
        "approx": "≈",
        "usdkg": "$/кг",
        "source": "Источник",
        "asof": "на",
        "no_quotes_try_refresh": "Котировки сейчас недоступны — попробуйте обновить.",
        "calculator": "Калькулятор",
        "price_source": "Источник цены",
        "online_stooq": "Онлайн (Stooq: KC.F / RM.F)",
        "manual_input": "Введу вручную",
        "instrument": "Инструмент",
        "base_price": "Базовая цена $/кг",
        "base_price_from_market": "Базовая цена $/кг (из рынка)",
        "diff_per_kg": "Дифференциал $/кг (±)",
        "diff_help": "Плюс — надбавка, минус — скидка к базовой цене",
        "effective_price": "Эффективная цена",
        "container_weight": "Контейнер / вес",
        "container": "Контейнер",
        "weight": "Вес (кг)",
        "incoterm": "Incoterm",
        "jurisdiction": "Юрисдикция (пресет НДС/пошлины)",
        "country_region": "Страна/регион",
        "vat_rate": "Ставка НДС (0..1)",
        "duty_ad": "Пошлина (адвал., 0..1)",
        "duty_sp": "Пошлина (специф., $/кг)",
        "route": "Маршрут (пресеты фрахта/страховки для CFR/CIF)",
        "freight": "Фрахт (USD)",
        "insurance": "Страховка (USD)",
        "local_fees": "Локальные сборы (стартер)",
        "fee_name": "Название",
        "fee_type": "Тип",
        "fee_fixed": "фикс.",
        "fee_percent": "проц.",
        "fee_amount": "Сумма ($)",
        "fee_rate": "Ставка (%)",
        "fee_base": "База",
        "fee_vat_base": "Включать в базу НДС",
        "compute": "Рассчитать",
        "result": "Разбор",
        "landed_total": "Итого (USD)",
        "usd_per_kg": "$/кг",
        "fill_and_compute": "Заполните поля и нажмите «Рассчитать».",
    },
}

def T(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return I18N.get(lang, I18N["en"]).get(key, key)

# ---------- Language switch (top) ----------
lang_prev = st.session_state["lang"]
lang_choice = st.radio(
    "🌐 " + T("lang_label"),
    ["en", "ru"],
    horizontal=True,
    index=0 if lang_prev == "en" else 1,
    format_func=lambda x: "English" if x == "en" else "Русский",
)
if lang_choice != lang_prev:
    st.session_state["lang"] = lang_choice
    st.rerun()

# ---------- Header (logo + title + tagline) ----------
col_l, col_c = st.columns([1, 2])
with col_l:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)
    else:
        st.markdown(
            f"<div style='font-weight:800;font-size:28px;color:{GRAPHITE}'>{APP_NAME}</div>",
            unsafe_allow_html=True,
        )
with col_c:
    st.markdown(
        f"<div style='text-align:center;font-weight:800;font-size:34px;color:{GRAPHITE}'>{APP_NAME}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='text-align:center;color:#64748B'>{T('tagline')}</div>",
        unsafe_allow_html=True,
    )
st.markdown("<hr style='margin-top:6px;margin-bottom:10px'/>", unsafe_allow_html=True)

# ---------- Helpers: Stooq fetch ----------
def kc_centslb_to_usdkg(x: float) -> float:
    return (float(x) / 100.0) / 0.45359237

def rm_usdt_to_usdkg(x: float) -> float:
    return float(x) / 1000.0

def _http_get_text(url: str, params: dict | None = None, timeout: float = REQ_TIMEOUT) -> tuple[Optional[str], Dict]:
    meta = {"url": url, "params": params, "status": None, "ctype": None, "err": None}
    try:
        r = requests.get(url, headers=UA, params=params, timeout=timeout)
        meta["status"] = r.status_code
        meta["ctype"] = r.headers.get("Content-Type", "")
        if not r.ok:
            return None, meta
        text = r.text
        if not text or "<html" in text.lower():
            return None, meta
        return text, meta
    except Exception as e:
        meta["err"] = f"{type(e).__name__}: {e}"
        return None, meta

def _try_domains(path: str, params: dict, retries: int = 3) -> tuple[Optional[str], Dict]:
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
    reader = csv.DictReader(io.StringIO(text))
    row = next(reader, None)
    if not row:
        return None
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
            "asof": f"{date} {time_} (Stooq snapshot)", "source": "Stooq /q/l"}

def _parse_eod(text: str, expect_symbol: str) -> Optional[Dict]:
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return None
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
def stooq_latest(symbol: str, seed: int = 0) -> Dict:
    """Try Stooq snapshot, then EOD CSV. Cached."""
    sym = symbol.upper()
    # snapshot
    snap_txt, _ = _try_domains("/q/l/", params={"s": sym.lower(), "f": "sd2t2ohlcv", "h": "", "e": "csv"})
    if snap_txt:
        parsed = _parse_snapshot(snap_txt, sym)
        if parsed:
            return parsed
    # eod
    eod_txt, _ = _try_domains("/q/d/l/", params={"s": sym.lower(), "i": "d"})
    if eod_txt:
        parsed = _parse_eod(eod_txt, sym)
        if parsed:
            return parsed
    raise RuntimeError(f"Stooq not available for {sym}")

# ---------- Controls row (no sidebar) ----------
for key, default in [("refresh_seed", 0), ("auto_refresh", False), ("last_check", None), ("last_fetch_ts", None)]:
    st.session_state.setdefault(key, default)

ctl_l, ctl_r = st.columns([2, 1])
with ctl_l:
    clicked = st.button("↻ " + T("refresh"))
    st.session_state["auto_refresh"] = st.toggle(T("auto_refresh"), value=st.session_state["auto_refresh"])
    st.caption(f"{T('last_check')}: {st.session_state['last_check'] or '—'}")

    if clicked:
        st.session_state["refresh_seed"] += 1

    # Safe timer (no st.autorefresh)
    if st.session_state["auto_refresh"]:
        last_ts = st.session_state.get("last_fetch_ts")
        due = (last_ts is None) or ((datetime.now(timezone.utc) - last_ts).total_seconds() >= 900)
        if due:
            st.session_state["refresh_seed"] += 1
            st.session_state["last_fetch_ts"] = datetime.now(timezone.utc)
            st.experimental_rerun()

with ctl_r:
    st.markdown(
        "<div style='text-align:right;color:#94A3B8'>Market — Stooq (safe boot)</div>",
        unsafe_allow_html=True,
    )

# ---------- Fetch quotes once for the screen ----------
kc, rm = {}, {}
try:
    kc = stooq_latest(ARABICA_SYMBOL, seed=st.session_state["refresh_seed"])
except Exception:
    kc = {}
try:
    rm = stooq_latest(ROBUSTA_SYMBOL, seed=st.session_state["refresh_seed"])
except Exception:
    rm = {}

now_utc = datetime.now(timezone.utc)
st.session_state["last_check"] = now_utc.strftime("%H:%M UTC")
st.session_state["last_fetch_ts"] = now_utc

# ---------- Top market strip ----------
st.markdown(
    f"<div style='margin-top:-4px;margin-bottom:8px;color:#64748B'>{T('market_title')}</div>",
    unsafe_allow_html=True,
)
a, b = st.columns(2)
with a:
    if kc.get("last_raw") is not None:
        st.metric(T("arabica"), f"{kc['last_raw']:.2f} {kc.get('unit','')}")
        st.caption(f"{T('approx')} {kc.get('usdkg',0):.3f} {T('usdkg')} • {T('source')}: {kc.get('source','Stooq')} • {T('asof')} {kc.get('asof','')}")
    else:
        st.warning(T("no_quotes_try_refresh"))
with b:
    if rm.get("last_raw") is not None:
        st.metric(T("robusta"), f"{rm['last_raw']:.2f} {rm.get('unit','')}")
        st.caption(f"{T('approx')} {rm.get('usdkg',0):.3f} {T('usdkg')} • {T('source')}: {rm.get('source','Stooq')} • {T('asof')} {rm.get('asof','')}")
    else:
        st.warning(T("no_quotes_try_refresh"))

st.markdown("<hr style='margin-top:8px;margin-bottom:14px'/>", unsafe_allow_html=True)

# ---------- Calculator helpers ----------
def compute_customs_value(incoterm: str, goods_value: float, freight: float, insurance: float) -> float:
    inc = incoterm.upper()
    if inc in {"FOB", "EXW"}:
        return goods_value + freight + insurance
    if inc == "CFR":
        return goods_value + insurance
    return goods_value  # CIF включает фрахт+страховку

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

# ---------- Calculator UI ----------
st.header(T("calculator"))

src = st.radio(T("price_source"), [T("online_stooq"), T("manual_input")], horizontal=True)

col1, col2, col3 = st.columns(3)
with col1:
    instrument = st.selectbox(T("instrument"), ["Arabica (KC.F)", "Robusta (RM.F)"])

with col2:
    if src == T("online_stooq"):
        market_usdkg = kc.get("usdkg") if instrument.startswith("Arabica") else rm.get("usdkg")
        if market_usdkg is None:
            st.warning(T("no_quotes_try_refresh"))
            base_usdkg = st.number_input(T("base_price"), min_value=0.0, value=3.000, step=0.001)
        else:
            st.text_input(T("base_price_from_market"), value=f"{market_usdkg:.4f}", disabled=True)
            base_usdkg = float(market_usdkg)
    else:
        base_usdkg = st.number_input(T("base_price"), min_value=0.0, value=3.000, step=0.001)

with col3:
    diff = st.number_input(T("diff_per_kg"), value=0.000, step=0.010, help=T("diff_help"))

effective_usdkg = base_usdkg + diff
st.caption(f"{T('effective_price')}: **{effective_usdkg:.4f} {T('usdkg')}**")

# Container / weight
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
    incoterm = st.selectbox(T("incoterm"), ["FOB", "CFR", "CIF"])

# Jurisdiction presets
st.markdown(f"**{T('jurisdiction')}**")
jur_presets = {
    "EAEU - Belarus": {"vat_rate": 0.20, "duty_rate": 0.00},
    "EAEU - Russia": {"vat_rate": 0.20, "duty_rate": 0.00},
    "UAE": {"vat_rate": 0.05, "duty_rate": 0.00},
}
colsj1, colsj2 = st.columns(2)
with colsj1:
    jname = st.selectbox(T("country_region"), list(jur_presets.keys()))
with colsj2:
    j = jur_presets[jname]
    vat_rate = st.number_input(T("vat_rate"), min_value=0.0, max_value=1.0,
                               value=float(j.get("vat_rate", 0.0)), step=0.01)
duty_rate = st.number_input(T("duty_ad"), min_value=0.0, max_value=1.0,
                            value=float(j.get("duty_rate", 0.0)), step=0.01)
duty_sp_perkg = st.number_input(T("duty_sp"), min_value=0.0, value=0.0, step=0.01)

# Route presets
st.markdown(f"**{T('route')}**")
route_presets = {
    "Santos → Riga": {"incoterms": ["CFR", "CIF"], "freight": 1800, "insurance": 80},
    "Santos → Dubai": {"incoterms": ["CFR", "CIF"], "freight": 1600, "insurance": 70},
    "Jebel Ali → Riyadh": {"incoterms": ["CFR", "CIF"], "freight": 600, "insurance": 40},
}
route_names = ["(none)"] + list(route_presets.keys())
colr1, colr2, colr3 = st.columns(3)
with colr1:
    rsel = st.selectbox("Route", route_names)
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

# Local fees (starter)
st.markdown(f"**{T('local_fees')}**")
feec1, feec2, feec3, feec4, _ = st.columns([2, 1, 1, 1, 1])
with feec1:
    fee_name = st.text_input(T("fee_name"), value="Customs processing")
with feec2:
    fee_kind = st.selectbox(T("fee_type"), [T("fee_fixed"), T("fee_percent")])
    fk_internal = "fixed" if fee_kind.startswith(("fixed", "фикс")) else "percent"
with feec3:
    if fk_internal == "fixed":
        fee_amount = st.number_input(T("fee_amount"), min_value=0.0, value=25.0, step=1.0)
        fee_rate = 0.0
        fee_base = "CV"
    else:
        fee_rate = st.number_input(T("fee_rate"), min_value=0.0, value=0.0, step=0.1) / 100.0
        fee_base = st.selectbox(T("fee_base"), ["CV", "Goods", "CVPlusDuty"])
        fee_amount = 0.0
with feec4:
    fee_vb = st.checkbox(T("fee_vat_base"), value=True)

fees = [{
    "name": fee_name, "kind": fk_internal, "amount": fee_amount,
    "rate": fee_rate, "base": fee_base, "vat_base": fee_vb
}]

st.divider()

if st.button(T("compute"), type="primary"):
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

    colL, colR = st.columns([1, 1])
    with colL:
        st.subheader(T("result"))
        st.metric(T("landed_total"), f"{b['total']:.2f}")
        st.metric(T("usd_per_kg"), f"{b['per_kg']:.4f}")
        st.write(f"Incoterm: {incoterm} • {T('weight')}: {weight_kg:,.0f} kg • {T('country_region')}: {jname}")
        st.write(f"{T('freight')}: {freight:.2f} • {T('insurance')}: {insurance:.2f} • Δ: {diff:+.3f} $/kg")

    with colR:
        st.dataframe(make_result_df(b), hide_index=True, use_container_width=True)
else:
    st.info(T("fill_and_compute"))
