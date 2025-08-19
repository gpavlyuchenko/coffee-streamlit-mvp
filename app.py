import io, csv, time, random, json
from pathlib import Path
from typing import Optional, Dict

import requests
import pandas as pd
import streamlit as st

# ========================= КОНСТАНТЫ =========================
STOOQ_CACHE_TTL = 900  # кэш 15 минут
UA = {"User-Agent": "Mozilla/5.0"}
STOOQ_DOMAINS = ("https://stooq.com", "https://stooq.pl")  # пробуем оба
ARABICA_SYMBOL = "KC.F"   # Arabica continuous (¢/lb)
ROBUSTA_SYMBOL = "RM.F"   # Robusta continuous (USD/t)

st.set_page_config(page_title="Coffee Landed Cost — MVP", page_icon="☕", layout="wide")
st.title("☕ Coffee Landed Cost — MVP (Stooq: KC.F / RM.F)")

# ======================= ВСПОМОГАТЕЛЬНОЕ =====================
def kc_centslb_to_usdkg(x: float) -> float:
    """¢/lb -> $/kg"""
    return (float(x) / 100.0) / 0.45359237

def rm_usdt_to_usdkg(x: float) -> float:
    """USD/tonne -> $/kg"""
    return float(x) / 1000.0

def _http_get_text(url: str, params: dict | None = None, timeout: float = 14.0) -> tuple[Optional[str], Dict]:
    """GET с фильтром HTML; вернёт (text|None, meta)."""
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
    """Перебираем stooq.com и stooq.pl, по каждому — до N попыток с джиттером."""
    last_meta: Dict = {}
    for base in STOOQ_DOMAINS:
        for attempt in range(retries):
            text, meta = _http_get_text(base + path, params=params)
            if text:
                meta["domain"] = base
                return text, meta
            last_meta = {**meta, "domain": base, "attempt": attempt + 1}
            time.sleep(0.2 + random.random()*0.5)
    return None, last_meta

def _parse_snapshot(text: str, expect_symbol: str) -> Optional[Dict]:
    """Парсим /q/l CSV: Symbol,Date,Time,Open,High,Low,Close,Volume."""
    reader = csv.DictReader(io.StringIO(text))
    row = next(reader, None)
    if not row:
        return None
    symbol = (row.get("Symbol") or "").strip().upper()
    if symbol != expect_symbol.upper():  # иногда Stooq подсовывает другой символ
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
    """Парсим /q/d/l CSV: Date,Open,High,Low,Close,Volume."""
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
    """
    Возвращает котировку Stooq по непрерывному символу ('KC.F' или 'RM.F').
    1) /q/l снапшот -> 2) /q/d/l дневной -> иначе RuntimeError.
    seed — просто число для ручного сброса кэша (кнопка).
    """
    sym = symbol.upper()

    # 1) snapshot
    snap_txt, snap_meta = _try_domains("/q/l/", params={"s": sym.lower(), "f": "sd2t2ohlcv", "h": "", "e": "csv"})
    if snap_txt:
        parsed = _parse_snapshot(snap_txt, sym)
        if parsed:
            if debug: parsed["_debug"] = {"endpoint": "snapshot", **snap_meta}
            return parsed

    # 2) EOD fallback
    eod_txt, eod_meta = _try_domains("/q/d/l/", params={"s": sym.lower(), "i": "d"})
    if eod_txt:
        parsed = _parse_eod(eod_txt, sym)
        if parsed:
            if debug: parsed["_debug"] = {"endpoint": "eod", **eod_meta}
            return parsed

    meta = snap_meta if snap_meta else eod_meta
    raise RuntimeError(f"Stooq not available for {sym} (status={meta.get('status')}, ctype={meta.get('ctype')}, err={meta.get('err')})")

# ========================== КАЛЬКУЛЯТОР ======================
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

# ============================ UI: САЙДБАР ====================
with st.sidebar:
    st.subheader("Источники (Stooq: KC.F / RM.F)")
    if "refresh_seed" not in st.session_state: st.session_state["refresh_seed"] = 0
    if st.button("↻ Обновить котировки"): st.session_state["refresh_seed"] += 1
    debug = st.checkbox("Показать отладку", value=False)

    try:
        kc = stooq_latest(ARABICA_SYMBOL, seed=st.session_state["refresh_seed"], debug=debug)
        st.metric("Arabica KC.F", f"{kc['last_raw']:.2f} {kc['unit']}")
        st.caption(f"≈ {kc['usdkg']:.3f} $/кг • {kc['source']} • as of {kc['asof']}")
        if debug and "_debug" in kc: st.code(json.dumps(kc["_debug"], ensure_ascii=False, indent=2), language="json")
    except Exception as e:
        kc = {"usdkg": None}
        st.error(f"Arabica KC.F: {e}")

    try:
        rm = stooq_latest(ROBUSTA_SYMBOL, seed=st.session_state["refresh_seed"], debug=debug)
        st.metric("Robusta RM.F", f"{rm['last_raw']:.2f} {rm['unit']}")
        st.caption(f"≈ {rm['usdkg']:.3f} $/кг • {rm['source']} • as of {rm['asof']}")
        if debug and "_debug" in rm: st.code(json.dumps(rm["_debug"], ensure_ascii=False, indent=2), language="json")
    except Exception as e:
        rm = {"usdkg": None}
        st.error(f"Robusta RM.F: {e}")

    st.caption("Snapshot у Stooq может быть с задержкой; если недоступен — берём EOD.")

# ============================ UI: ФОРМА ======================
st.header("Калькулятор")

src = st.radio("Источник цены", ["Онлайн (Stooq: KC.F / RM.F)", "Введу вручную"], horizontal=True)

col1, col2, col3 = st.columns(3)
with col1:
    instrument = st.selectbox("Инструмент", ["Arabica (KC.F)", "Robusta (RM.F)"])

with col2:
    if src == "Онлайн (Stooq: KC.F / RM.F)":
        if instrument.startswith("Arabica"):
            if kc.get("usdkg") is not None:
                st.text_input("Базовая цена $/кг (из KC.F)", value=f"{kc['usdkg']:.4f}", disabled=True)
                base_usdkg = float(kc["usdkg"])
            else:
                st.warning("Нет данных по KC.F — введите вручную.")
                base_usdkg = st.number_input("Базовая цена $/кг", min_value=0.0, value=3.000, step=0.001)
        else:
            if rm.get("usdkg") is not None:
                st.text_input("Базовая цена $/кг (из RM.F)", value=f"{rm['usdkg']:.4f}", disabled=True)
                base_usdkg = float(rm["usdkg"])
            else:
                st.warning("Нет данных по RM.F — введите вручную.")
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

# Юрисдикция (простые пресеты)
st.markdown("**Юрисдикция (НДС/пошлина пресет)**")
jur_presets = {
    "EAEU - Belarus": {"vat_rate": 0.20, "duty_rate": 0.00},
    "EAEU - Russia":  {"vat_rate": 0.20, "duty_rate": 0.00},
    "UAE":            {"vat_rate": 0.05, "duty_rate": 0.00},
}
colsj1, colsj2 = st.columns(2)
with colsj1:
    jname = st.selectbox("Страна/регион", list(jur_presets.keys()))
with colsj2:
    j = jur_presets[jname]
    vat_rate = st.number_input("Ставка НДС (0..1)", min_value=0.0, max_value=1.0,
                               value=float(j.get("vat_rate", 0.0)), step=0.01)
duty_rate = st.number_input("Пошлина (адвал., 0..1)", min_value=0.0, max_value=1.0,
                            value=float(j.get("duty_rate", 0.0)), step=0.01)
duty_sp_perkg = st.number_input("Пошлина (специф., $/кг)", min_value=0.0, value=0.0, step=0.01)

# Маршрут (простые пресеты для CFR/CIF)
st.markdown("**Маршрут (для CFR/CIF пресеты фрахта/страховки)**")
route_presets = {
    "Santos → Riga":     {"incoterms": ["CFR","CIF"], "freight": 1800, "insurance": 80},
    "Santos → Dubai":    {"incoterms": ["CFR","CIF"], "freight": 1600, "insurance": 70},
    "Jebel Ali → Riyadh":{"incoterms": ["CFR","CIF"], "freight": 600,  "insurance": 40},
}
route_names = ["(не использ.)"] + list(route_presets.keys())
colr1, colr2, colr3 = st.columns(3)
with colr1:
    rsel = st.selectbox("Маршрут", route_names)
with colr2:
    if rsel != "(не использ.)" and incoterm in route_presets[rsel]["incoterms"]:
        freight = st.number_input("Фрахт (USD)", min_value=0.0, value=float(route_presets[rsel]["freight"]), step=10.0)
    else:
        freight = st.number_input("Фрахт (USD)", min_value=0.0, value=1800.0, step=10.0)
with colr3:
    if rsel != "(не использ.)" and incoterm in route_presets[rsel]["incoterms"]:
        insurance = st.number_input("Страховка (USD)", min_value=0.0, value=float(route_presets[rsel]["insurance"]), step=5.0)
    else:
        insurance = st.number_input("Страховка (USD)", min_value=0.0, value=80.0, step=5.0)

# Локальные сборы (минимальный редактор)
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
