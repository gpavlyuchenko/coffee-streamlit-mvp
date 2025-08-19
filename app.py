import io, json
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict

import requests
import pandas as pd
import streamlit as st

# ====== КОНСТАНТЫ ===========================================================
ARABICA_SERIES = "KCZ25.F"   # Arabica (ICE US) — только эта серия
ROBUSTA_SERIES = "RMU25.F"   # Robusta (ICE Europe) — только эта серия
CACHE_TTL_SEC  = 900         # 15 минут
UA_HEADERS     = {"User-Agent": "Mozilla/5.0"}

st.set_page_config(page_title="Coffee Landed Cost — MVP", page_icon="☕", layout="wide")
st.title("☕ Coffee Landed Cost — MVP (Stooq snapshot + EOD fallback)")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ====== УТИЛИТЫ =============================================================
def fmt_asof(asof: Optional[str]) -> str:
    return asof if asof else "n/a"

def arabica_centlb_to_usd_per_kg(cents_per_lb: float) -> float:
    return (float(cents_per_lb) / 100.0) / 0.45359237

def robusta_usd_per_tonne_to_usd_per_kg(usd_per_tonne: float) -> float:
    return float(usd_per_tonne) / 1000.0

def load_json(filename: str, default: dict) -> dict:
    p = DATA_DIR / filename
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# ====== STOOQ: SNAPSHOT (/q/l) + EOD CSV (/q/d/l) ===========================
def _stooq_snapshot(symbol: str) -> Optional[dict]:
    """
    Интрадей-снапшот: https://stooq.com/q/l/?s=<symbol>&f=sd2t2ohlcv&h&e=csv
    Возвращает {'last_raw', 'unit', 'usdkg', 'asof', 'source'} или None.
    """
    url = "https://stooq.com/q/l/"
    params = {"s": symbol.lower(), "f": "sd2t2ohlcv", "h": "", "e": "csv"}
    r = requests.get(url, headers=UA_HEADERS, params=params, timeout=12)
    r.raise_for_status()

    # Пример CSV: Symbol,Date,Time,Open,High,Low,Close,Volume
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        return None
    cols = [c.strip().lower() for c in df.columns]
    row = df.iloc[0]
    symbol_csv = str(row[cols[0]]).upper()
    if symbol_csv != symbol.upper():
        return None

    close_val = float(row["close" if "close" in cols else cols[-2]])
    date_val  = str(row["date"  if "date"  in cols else cols[1]])
    time_val  = str(row["time"  if "time"  in cols else cols[2]])

    if symbol.upper().startswith("KC"):
        unit = "¢/lb"
        usdkg = arabica_centlb_to_usd_per_kg(close_val)
    elif symbol.upper().startswith("RM"):
        unit = "USD/t"
        usdkg = robusta_usd_per_tonne_to_usd_per_kg(close_val)
    else:
        unit, usdkg = "", None

    return {
        "last_raw": close_val,
        "unit": unit,
        "usdkg": usdkg,
        "asof": f"{date_val} {time_val} (Stooq snapshot)",
        "source": "Stooq /q/l"
    }

def _stooq_eod(symbol: str) -> dict:
    """
    Дневной CSV (EOD): https://stooq.pl/q/d/l/?s=<symbol>&i=d
    Возвращает {'last_raw','unit','usdkg','asof','source'}.
    """
    for base in ("https://stooq.pl", "https://stooq.com"):
        url = f"{base}/q/d/l/?s={symbol.lower()}&i=d"
        r = requests.get(url, headers=UA_HEADERS, timeout=12)
        if not r.ok or not r.text.strip():
            continue
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty or "Close" not in df.columns:
            continue
        last = df.iloc[-1]
        close_val = float(last["Close"])
        date_val  = str(last["Date"])
        if symbol.upper().startswith("KC"):
            unit = "¢/lb"
            usdkg = arabica_centlb_to_usd_per_kg(close_val)
        elif symbol.upper().startswith("RM"):
            unit = "USD/t"
            usdkg = robusta_usd_per_tonne_to_usd_per_kg(close_val)
        else:
            unit, usdkg = "", None
        return {
            "last_raw": close_val,
            "unit": unit,
            "usdkg": usdkg,
            "asof": f"{date_val} (EOD)",
            "source": "Stooq /q/d/l"
        }
    raise RuntimeError(f"Stooq EOD CSV not available for {symbol}")

@st.cache_data(ttl=CACHE_TTL_SEC)
def stooq_series_last(symbol: str, seed: int = 0) -> dict:
    """
    1) Пытаемся взять интрадей-снапшот (/q/l)
    2) Если нет — EOD CSV (/q/d/l)
    seed — просто число для ручного сброса кэша кнопкой.
    """
    snap = None
    try:
        snap = _stooq_snapshot(symbol)
    except Exception:
        snap = None

    if snap and snap.get("usdkg") is not None:
        return snap

    # fallback на EOD
    return _stooq_eod(symbol)

# ====== КАЛЬКУЛЯТОР =========================================================
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
        ["Landed total", b["total"], currency],
        ["Per kg", b["per_kg"], f"{currency}/kg"],
    ]
    return pd.DataFrame(rows, columns=["Metric","Value","Unit"])

# ====== САЙДБАР =============================================================
with st.sidebar:
    st.subheader("Источники (Stooq)")
    if "refresh_seed" not in st.session_state: st.session_state["refresh_seed"] = 0
    if st.button("↻ Обновить котировки"): st.session_state["refresh_seed"] += 1

    try:
        arab = stooq_series_last(ARABICA_SERIES, seed=st.session_state["refresh_seed"])
        st.metric(f"Arabica {ARABICA_SERIES}", f"{arab['last_raw']:.2f} {arab['unit']}")
        st.caption(f"≈ {arab['usdkg']:.3f} $/кг • {arab['source']} • as of {fmt_asof(arab['asof'])}")
    except Exception as e:
        arab = {"usdkg": None}
        st.error(f"Arabica {ARABICA_SERIES}: нет данных ({e})")

    try:
        robu = stooq_series_last(ROBUSTA_SERIES, seed=st.session_state["refresh_seed"])
        st.metric(f"Robusta {ROBUSTA_SERIES}", f"{robu['last_raw']:.2f} {robu['unit']}")
        st.caption(f"≈ {robu['usdkg']:.3f} $/кг • {robu['source']} • as of {fmt_asof(robu['asof'])}")
    except Exception as e:
        robu = {"usdkg": None}
        st.error(f"Robusta {ROBUSTA_SERIES}: нет данных ({e})")

    st.caption("Примечание: snapshot у Stooq имеет задержку; если недоступен, берём EOD.")

# ====== ОСНОВНАЯ ФОРМА ======================================================
st.header("Калькулятор")

src = st.radio("Источник цены", ["Онлайн (Stooq: KCZ25.F / RMU25.F)", "Введу вручную"], horizontal=True)

col1, col2, col3 = st.columns(3)
with col1:
    instrument = st.selectbox("Инструмент", ["Arabica (KCZ25.F)", "Robusta (RMU25.F)"])

with col2:
    if src == "Онлайн (Stooq: KCZ25.F / RMU25.F)":
        if instrument.startswith("Arabica"):
            if arab.get("usdkg") is not None:
                st.text_input("Базовая цена $/кг (из KCZ25.F)", value=f"{arab['usdkg']:.4f}", disabled=True)
                base_usdkg = float(arab["usdkg"])
            else:
                st.warning("Нет данных по KCZ25.F — введите вручную.")
                base_usdkg = st.number_input("Базовая цена $/кг", min_value=0.0, value=3.000, step=0.001)
        else:
            if robu.get("usdkg") is not None:
                st.text_input("Базовая цена $/кг (из RMU25.F)", value=f"{robu['usdkg']:.4f}", disabled=True)
                base_usdkg = float(robu["usdkg"])
            else:
                st.warning("Нет данных по RMU25.F — введите вручную.")
                base_usdkg = st.number_input("Базовая цена $/кг", min_value=0.0, value=3.000, step=0.001)
    else:
        base_usdkg = st.number_input("Базовая цена $/кг", min_value=0.0, value=3.000, step=0.001)

with col3:
    diff = st.number_input("Дифференциал $/кг (±)", value=0.000, step=0.010, help="Добавка/скидка к базовой цене")

effective_usdkg = base_usdkg + diff
st.caption(f"Эффективная цена: **{effective_usdkg:.4f} $/кг**")

# Контейнер / вес
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
    incoterm = st.selectbox("Incoterm", ["FOB","CFR","CIF"], help="Упрощённая логика")

# Юрисдикция
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

# Маршрут
st.markdown("**Маршрут (для CFR/CIF пресеты фрахта/страховки)**")
routes_defaults = {
    "Santos → Riga":     {"incoterms": ["CFR","CIF"], "freight": 1800, "insurance": 80},
    "Santos → Dubai":    {"incoterms": ["CFR","CIF"], "freight": 1600, "insurance": 70},
    "Jebel Ali → Riyadh":{"incoterms": ["CFR","CIF"], "freight": 600,  "insurance": 40},
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

# Сборы
st.markdown("**Локальные сборы (стартер)**")
feec1, feec2, feec3, feec4, _ = st.columns([2,1,1,1,1])
with feec1:
    fee_name = st.text_input("Название", value="Customs processing")
with feec2:
    fee_kind = st.selectbox("Тип", ["fixed","percent"])
with feec3:
    if fee_kind == "fixed":
        fee_amount = st.number_input("Сумма ($)", min_value=0.0, value=25.0, step=1.0)
        fee_rate = 0.0; fee_base = "CV"
    else:
        fee_rate = st.number_input("Ставка (%)", min_value=0.0, value=0.0, step=0.1) / 100.0
        fee_base = st.selectbox("База", ["CV","Goods","CVPlusDuty"])
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
        fees=fees
    )

    colL, colR = st.columns([1,1])
    with colL:
        st.subheader("Итог")
        st.metric("Landed total (USD)", f"{b['total']:.2f}")
        st.metric("$/кг", f"{b['per_kg']:.4f}")
        st.write(f"Incoterm: {incoterm} • Вес: {weight_kg:,.0f} кг • Регион: {jname}")
        st.write(f"Фрахт: {freight:.2f} • Страховка: {insurance:.2f} • Дифференциал: {diff:+.3f} $/кг")

    with colR:
        st.subheader("Разбор")
        st.dataframe(make_result_df(b), hide_index=True, use_container_width=True)
else:
    st.info("Заполните поля и нажмите «Рассчитать».")
