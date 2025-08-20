# app.py — BeanRoute (EN/RU), safe boot, cleaner UI

import io, csv, time, random, json
from pathlib import Path
from typing import Optional, Dict, Tuple

import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

# ========================= BRAND / THEME =========================
APP_NAME = "BeanRoute"
TAGLINE_EN = "Coffee imports, made clear."
TAGLINE_RU = "Импорт кофе — прозрачно и просто."

LOGO_PATH = Path("logo_light.svg")  # необязательно; если файла нет — покажем текст
PRIMARY = "#0FB5A8"
GRAPHITE = "#0F172A"

# ========================= GLOBAL SETTINGS ======================
SAFE_BOOT = True              # не ходим в сеть на холодном старте
STOOQ_CACHE_TTL = 900         # кэш 15 минут
REQ_TIMEOUT = 2.5             # секунд на HTTP-запрос
UA = {"User-Agent": "Mozilla/5.0"}
STOOQ_DOMAINS = ("https://stooq.com", "https://stooq.pl")

ARABICA_SYMBOL = "KC.F"       # ¢/lb
ROBUSTA_SYMBOL = "RM.F"       # USD/t

st.set_page_config(page_title=f"{APP_NAME} — Coffee pricing", page_icon="☕", layout="wide")

# ========================= I18N =========================
def init_lang():
    if "lang" not in st.session_state:
        st.session_state.lang = "en"

def T(key: str) -> str:
    """Simple dictionary-based i18n."""
    d = {
        # hero
        "hero_caption_en": "KC.F / RM.F • routes • duties & VAT • clear $/kg in seconds",
        "hero_caption_ru": "KC.F / RM.F • маршруты • пошлины и НДС • понятная цена $/кг за секунды",

        # sidebar quotes
        "market_title_en": "Market — Stooq (safe boot)",
        "market_title_ru": "Рынок — Stooq (безопасный старт)",
        "refresh_en": "↻ Refresh quotes",
        "refresh_ru": "↻ Обновить котировки",
        "last_check_en": "Last check",
        "last_check_ru": "Последняя проверка",
        "delayed_note_en": "Quotes are delayed (snapshot/EOD). Verify with your broker/exchange.",
        "delayed_note_ru": "Котировки с задержкой (снапшот/EOD). Для сделок сверяйтесь с брокером/биржей.",
        "arabica_en": "Arabica KC.F",
        "arabica_ru": "Арабика KC.F",
        "robusta_en": "Robusta RM.F",
        "robusta_ru": "Робуста RM.F",

        # calculator
        "calc_title_en": "Calculator",
        "calc_title_ru": "Калькулятор",
        "src_en": "Price source",
        "src_ru": "Источник цены",
        "src_online_en": "Online (Stooq: KC.F / RM.F)",
        "src_online_ru": "Онлайн (Stooq: KC.F / RM.F)",
        "src_manual_en": "Manual input",
        "src_manual_ru": "Введу вручную",
        "instrument_en": "Instrument",
        "instrument_ru": "Инструмент",
        "base_from_kc_en": "Base $/kg (from KC.F)",
        "base_from_kc_ru": "Базовая цена $/кг (из KC.F)",
        "base_from_rm_en": "Base $/kg (from RM.F)",
        "base_from_rm_ru": "Базовая цена $/кг (из RM.F)",
        "base_manual_en": "Base $/kg",
        "base_manual_ru": "Базовая цена $/кг",
        "no_kc_en": "No KC.F data — enter manually.",
        "no_kc_ru": "Нет данных по KC.F — введите вручную.",
        "no_rm_en": "No RM.F data — enter manually.",
        "no_rm_ru": "Нет данных по RM.F — введите вручную.",
        "diff_en": "Differential $/kg (±)",
        "diff_ru": "Дифференциал $/кг (±)",
        "eff_price_en": "Effective price",
        "eff_price_ru": "Эффективная цена",

        "container_title_en": "Container / weight",
        "container_title_ru": "Контейнер / вес",
        "container_en": "Container",
        "container_ru": "Контейнер",
        "weight_en": "Weight (kg)",
        "weight_ru": "Вес (кг)",
        "incoterm_en": "Incoterm",
        "incoterm_ru": "Incoterm",

        "jur_title_en": "Jurisdiction (VAT/duty presets)",
        "jur_title_ru": "Юрисдикция (пресеты НДС/пошлины)",
        "region_en": "Country/region",
        "region_ru": "Страна/регион",
        "vat_rate_en": "VAT rate (0..1)",
        "vat_rate_ru": "Ставка НДС (0..1)",
        "duty_ad_en": "Duty (ad val., 0..1)",
        "duty_ad_ru": "Пошлина (адвал., 0..1)",
        "duty_sp_en": "Duty (specific, $/kg)",
        "duty_sp_ru": "Пошлина (специф., $/кг)",

        "route_title_en": "Route (CFR/CIF freight & insurance presets)",
        "route_title_ru": "Маршрут (пресеты фрахта/страховки для CFR/CIF)",
        "route_en": "Route",
        "route_ru": "Маршрут",
        "freight_en": "Freight (USD)",
        "freight_ru": "Фрахт (USD)",
        "ins_en": "Insurance (USD)",
        "ins_ru": "Страховка (USD)",

        "fees_title_en": "Local fees (starter)",
        "fees_title_ru": "Локальные сборы (стартер)",
        "fee_name_en": "Name",
        "fee_name_ru": "Название",
        "fee_type_en": "Type",
        "fee_type_ru": "Тип",
        "fee_fixed_en": "fixed",
        "fee_fixed_ru": "fixed",
        "fee_percent_en": "percent",
        "fee_percent_ru": "percent",
        "fee_amt_en": "Amount ($)",
        "fee_amt_ru": "Сумма ($)",
        "fee_rate_en": "Rate (%)",
        "fee_rate_ru": "Ставка (%)",
        "fee_base_en": "Base",
        "fee_base_ru": "База",
        "fee_vat_en": "Include in VAT base",
        "fee_vat_ru": "Включать в базу НДС",

        "calc_btn_en": "Calculate",
        "calc_btn_ru": "Рассчитать",
        "fill_hint_en": "Fill the fields and click “Calculate”.",
        "fill_hint_ru": "Заполните поля и нажмите «Рассчитать».",

        "result_en": "Result",
        "result_ru": "Итог",
        "breakdown_en": "Breakdown",
        "breakdown_ru": "Разбор",
    }
    lang = st.session_state.lang
    return d.get(f"{key}_{lang}", d.get(f"{key}_en", key))

# ========================= UTILS =========================
def kc_centslb_to_usdkg(x: float) -> float:
    return (float(x) / 100.0) / 0.45359237

def rm_usdt_to_usdkg(x: float) -> float:
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
        if not text or "<html" in text.lower():
            return None, meta
        return text, meta
    except Exception as e:
        meta["err"] = f"{type(e).__name__}: {e}"
        return None, meta

def _try_domains(path: str, params: dict, retries: int = 2) -> Tuple[Optional[str], Dict]:
    last_meta: Dict = {}
    for base in STOOQ_DOMAINS:
        for attempt in range(retries):
            text, meta = _http_get_text(base + path, params=params)
            if text:
                meta["domain"] = base
                return text, meta
            last_meta = {**meta, "domain": base, "attempt": attempt + 1}
            time.sleep(0.15 + random.random()*0.35)
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
def stooq_latest(symbol: str, seed: int = 0, debug: bool = False) -> Dict:
    """KC.F / RM.F — сперва снапшот /q/l, затем EOD /q/d/l."""
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

# ========================= PRICING ENGINE =========================
def compute_customs_value(incoterm: str, goods_value: float, freight: float, insurance: float) -> float:
    inc = incoterm.upper()
    if inc in {"FOB", "EXW"}: return goods_value + freight + insurance
    if inc == "CFR":          return goods_value + insurance
    return goods_value  # CIF включает фрахт+страховку

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
        ["Total", b["total"], currency],
        ["Per kg", b["per_kg"], f"{currency}/kg"],
    ]
    return pd.DataFrame(rows, columns=["Metric","Value","Unit"])

# ========================= LANGUAGE INIT + CSS =========================
init_lang()
st.markdown(
    f"""
    <style>
      .bean-hero h1 {{ margin-bottom: 0.25rem; }}
      .bean-hero p  {{ color: {GRAPHITE}CC; margin-top: 0.25rem; }}
      .stMetric label p {{ font-weight: 600; }}
      .bean-card {{ padding: 1rem 1rem; border: 1px solid #eaeaea; border-radius: 12px; background: #fff; }}
      .bean-subtle {{ background: #F6F7F9; border-color: #eef1f4; }}
      .bean-cta button {{ width: 100%; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ========================= HEADER (logo + language) =========================
header = st.container()
with header:
    c1, c2, c3 = st.columns([1.5, 6, 1.5], vertical_alignment="center")
    with c1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=False)
        else:
            st.markdown(f"### **{APP_NAME}**")
    with c2:
        st.markdown(
            f"""
            <div class="bean-hero">
              <h1>{APP_NAME}</h1>
              <p>{TAGLINE_EN if st.session_state.lang=='en' else TAGLINE_RU}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown("**🌐 Language**")
        lang = st.radio(
            label="",
            options=["en","ru"],
            format_func=lambda x: "English" if x=="en" else "Русский",
            horizontal=True,
            key="lang",
        )

# ========================= SIDEBAR: QUOTES (safe boot) =========================
with st.sidebar:
    st.subheader(T("market_title"))
    if "quotes" not in st.session_state:
        # на первом рендере не грузим сеть
        st.session_state.quotes = {
            "KC.F": {"error": "not fetched yet"},
            "RM.F": {"error": "not fetched yet"},
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    colbtn, coltime = st.columns([1, 1])
    with colbtn:
        if st.button(T("refresh"), use_container_width=True):
            with st.spinner("Fetching..."):
                # быстрая попытка получить снапшоты
                st.session_state.quotes = {
                    "KC.F": {},
                    "RM.F": {},
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    st.session_state.quotes["KC.F"] = stooq_latest(ARABICA_SYMBOL, seed=int(time.time()))
                except Exception as e:
                    st.session_state.quotes["KC.F"] = {"error": str(e)}
                try:
                    st.session_state.quotes["RM.F"] = stooq_latest(ROBUSTA_SYMBOL, seed=int(time.time()))
                except Exception as e:
                    st.session_state.quotes["RM.F"] = {"error": str(e)}
    with coltime:
        st.caption(f"{T('last_check')}: {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

    data = st.session_state.quotes
    kc = data.get("KC.F", {})
    rm = data.get("RM.F", {})

    box = st.container()
    with box:
        cA, cB = st.columns(2)
        with cA:
            if "error" in kc:
                st.error(f"{T('arabica')}: {kc['error']}")
            else:
                st.metric(T("arabica"), f"{kc['last_raw']:.2f} {kc.get('unit','')}")
                st.caption(f"≈ {kc.get('usdkg', 0):.3f} $/kg • {kc.get('source','?')} • {kc.get('asof','')}")
        with cB:
            if "error" in rm:
                st.error(f"{T('robusta')}: {rm['error']}")
            else:
                st.metric(T("robusta"), f"{rm['last_raw']:.2f} {rm.get('unit','')}")
                st.caption(f"≈ {rm.get('usdkg', 0):.3f} $/kg • {rm.get('source','?')} • {rm.get('asof','')}")

    st.caption(T("delayed_note"))

# ========================= MAIN: CALCULATOR =========================
st.markdown(f"### {T('calc_title')}")

src = st.radio(T("src"), [T("src_online"), T("src_manual")], horizontal=True, key="price_src")

col1, col2, col3 = st.columns(3)
with col1:
    instrument = st.selectbox(
        T("instrument"),
        ["Arabica (KC.F)", "Robusta (RM.F)"],
        index=0
    )

with col2:
    quotes = st.session_state.quotes
    kc = quotes.get("KC.F", {})
    rm = quotes.get("RM.F", {})
    if st.session_state.price_src == T("src_online"):
        if instrument.startswith("Arabica"):
            if kc.get("usdkg") is not None:
                st.text_input(T("base_from_kc"), value=f"{kc['usdkg']:.4f}", disabled=True)
                base_usdkg = float(kc["usdkg"])
            else:
                st.warning(T("no_kc"))
                base_usdkg = st.number_input(T("base_manual"), min_value=0.0, value=3.000, step=0.001)
        else:
            if rm.get("usdkg") is not None:
                st.text_input(T("base_from_rm"), value=f"{rm['usdkg']:.4f}", disabled=True)
                base_usdkg = float(rm["usdkg"])
            else:
                st.warning(T("no_rm"))
                base_usdkg = st.number_input(T("base_manual"), min_value=0.0, value=3.000, step=0.001)
    else:
        base_usdkg = st.number_input(T("base_manual"), min_value=0.0, value=3.000, step=0.001)

with col3:
    diff = st.number_input(T("diff"), value=0.000, step=0.010, help=None)

effective_usdkg = base_usdkg + diff
st.caption(f"{T('eff_price')}: **{effective_usdkg:.4f} $/kg**")

# --- Container / weight / incoterm ---
st.markdown(f"**{T('container_title')}**")
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

# --- Jurisdiction presets ---
st.markdown(f"**{T('jur_title')}**")
jur_presets = {
    "UAE":            {"vat_rate": 0.05, "duty_rate": 0.00},
    "EAEU - Belarus": {"vat_rate": 0.20, "duty_rate": 0.00},
    "EAEU - Russia":  {"vat_rate": 0.20, "duty_rate": 0.00},
}
colsj1, colsj2 = st.columns(2)
with colsj1:
    jname = st.selectbox(T("region"), list(jur_presets.keys()))
with colsj2:
    j = jur_presets[jname]
    vat_rate = st.number_input(T("vat_rate"), min_value=0.0, max_value=1.0,
                               value=float(j.get("vat_rate", 0.0)), step=0.01)
duty_rate = st.number_input(T("duty_ad"), min_value=0.0, max_value=1.0,
                            value=float(j.get("duty_rate", 0.0)), step=0.01)
duty_sp_perkg = st.number_input(T("duty_sp"), min_value=0.0, value=0.0, step=0.01)

# --- Route presets ---
st.markdown(f"**{T('route_title')}**")
route_presets = {
    "Santos → Riga":      {"incoterms": ["CFR","CIF"], "freight": 1800, "insurance": 80},
    "Santos → Dubai":     {"incoterms": ["CFR","CIF"], "freight": 1600, "insurance": 70},
    "Jebel Ali → Riyadh": {"incoterms": ["CFR","CIF"], "freight": 600,  "insurance": 40},
}
route_names = ["(none)"] + list(route_presets.keys())
colr1, colr2, colr3 = st.columns(3)
with colr1:
    rsel = st.selectbox(T("route"), route_names)
with colr2:
    if rsel != "(none)" and incoterm in route_presets[rsel]["incoterms"]:
        freight = st.number_input(T("freight"), min_value=0.0, value=float(route_presets[rsel]["freight"]), step=10.0)
    else:
        freight = st.number_input(T("freight"), min_value=0.0, value=1800.0, step=10.0)
with colr3:
    if rsel != "(none)" and incoterm in route_presets[rsel]["incoterms"]:
        insurance = st.number_input(T("ins"), min_value=0.0, value=float(route_presets[rsel]["insurance"]), step=5.0)
    else:
        insurance = st.number_input(T("ins"), min_value=0.0, value=80.0, step=5.0)

# --- Local fees (single row editor) ---
st.markdown(f"**{T('fees_title')}**")
feec1, feec2, feec3, feec4, _ = st.columns([2,1,1,1,1])
with feec1:
    fee_name = st.text_input(T("fee_name"), value="Customs processing")
with feec2:
    fee_kind = st.selectbox(T("fee_type"), [T("fee_fixed"), T("fee_percent")])
with feec3:
    if fee_kind == T("fee_fixed"):
        fee_amount = st.number_input(T("fee_amt"), min_value=0.0, value=25.0, step=1.0)
        fee_rate = 0.0; fee_base = "CV"
    else:
        fee_rate = st.number_input(T("fee_rate"), min_value=0.0, value=0.0, step=0.1) / 100.0
        fee_base = st.selectbox(T("fee_base"), ["CV","Goods","CVPlusDuty"])
        fee_amount = 0.0
with feec4:
    fee_vb = st.checkbox(T("fee_vat"), value=True)
fees = [{"name": fee_name, "kind": "fixed" if fee_kind==T("fee_fixed") else "percent",
         "amount": fee_amount, "rate": fee_rate, "base": fee_base, "vat_base": fee_vb}]

st.divider()

# --- Calculate ---
if st.button(T("calc_btn"), type="primary"):
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
        st.metric("Total (USD)", f"{b['total']:.2f}")
        st.metric("$/kg", f"{b['per_kg']:.4f}")
        st.caption(f"Incoterm: {incoterm} • Weight: {weight_kg:,.0f} kg • Region: {jname}")
        st.caption(f"Freight: {freight:.2f} • Insurance: {insurance:.2f} • Diff: {diff:+.3f} $/kg")
    with colR:
        st.subheader(T("breakdown"))
        st.dataframe(make_result_df(b), hide_index=True, use_container_width=True)
else:
    st.info(T("fill_hint"))
