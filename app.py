import io
import os
import re
import json
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

# ---------- Page ----------
st.set_page_config(page_title="Coffee Landed Cost — MVP", page_icon="☕", layout="wide")
st.title("☕ Coffee Landed Cost — MVP")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ---------- Small utils ----------
def fmt_ts(ts: Optional[int]) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M UTC") if ts else "n/a"

def load_json(filename: str, default: dict) -> dict:
    p = DATA_DIR / filename
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# ---------- Units / conversions ----------
def arabica_centlb_to_usd_per_kg(cents_per_lb: float) -> float:
    return (float(cents_per_lb) / 100.0) / 0.45359237

def robusta_usd_per_tonne_to_usd_per_kg(usd_per_tonne: float) -> float:
    return float(usd_per_tonne) / 1000.0

# ---------- HTTP session with retries ----------
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

def _retry_obj():
    # совместимость с urllib3 v1/v2
    try:
        return Retry(
            total=3,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
    except TypeError:  # urllib3 v1.x
        return Retry(
            total=3,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=frozenset(["GET"]),
            raise_on_status=False,
        )

def _http_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=_retry_obj()))
    return s

# ---------- Stooq fetchers (для Robusta и бэкапов) ----------
@st.cache_data(ttl=600)
def fetch_stooq_csv(symbol: str, interval: str = "d") -> pd.DataFrame:
    """CSV из Stooq (kc.f / rm.f / RMU25.F)."""
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
    """Находим актуальную серию робусты (например RMU25.F)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://stooq.com/q/f/?s={base_symbol}"
    r = requests.get(url, headers=headers, timeout=10)
    if not r.ok or not r.text:
        return None
    m = re.search(r"RM[A-Z]\d{2}\.F", r.text, re.IGNORECASE)
    return m.group(0).upper() if m else None

# ---------- Yahoo fetchers ----------
def fetch_yahoo_intraday_last(symbol: str, interval: str = "1m", range_: str = "1d") -> Tuple[float, Optional[int]]:
    """
    Последняя непустая свеча Yahoo (chart v8). Возвращает (price, unix_ts_utc).
    """
    sess = _http_session()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": range_, "interval": interval}
    r = sess.get(url, headers=HEADERS, params=params, timeout=10)
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
        last_ts = j["meta"].get("regularMarketTime")
    return last, last_ts

@st.cache_data(ttl=60)
def yahoo_quote_multi(symbols: List[str]) -> Dict[str, Dict]:
    """
    Yahoo quote v7 (не HTML). Возвращает {symbol: {price, ts}}.
    Делает:
      • чанки по 8 символов,
      • ретраи с backoff,
      • фолбэк на chart v8 по каждому символу при ошибке.
    """
    out: Dict[str, Dict] = {}
    if not symbols:
        return out

    sess = _http_session()

    def fetch_chunk(chunk: List[str]) -> None:
        url = "https://query1.finance.yahoo.com/v7/finance/quote"
        try:
            r = sess.get(url, headers=HEADERS, params={"symbols": ",".join(chunk)}, timeout=10)
            r.raise_for_status()
            res = r.json().get("quoteResponse", {}).get("result", [])
            for row in res:
                sym = row.get("symbol")
                price = row.get("regularMarketPrice")
                ts = row.get("regularMarketTime")
                if sym and price is not None:
                    out[sym] = {"price": float(price), "ts": int(ts) if ts else None}
        except Exception:
            # фолбэк на v8 по каждому
            for sym in chunk:
                try:
                    price, ts = fetch_yahoo_intraday_last(sym, interval="1m", range_="1d")
                    out[sym] = {"price": float(price), "ts": ts}
                except Exception:
                    pass

    CHUNK = 8
    for i in range(0, len(symbols), CHUNK):
        fetch_chunk(symbols[i:i + CHUNK])

    return out

# ---------- KC futures board (Arabica) ----------
MONTHS = [(3, "H"), (5, "K"), (7, "N"), (9, "U"), (12, "Z")]  # Mar/May/Jul/Sep/Dec

def next_kc_contracts(n: int = 8, now: Optional[datetime] = None) -> List[Dict]:
    """Список ближайших KC: [{'contract':'KCU25','yahoo':'KCU25=F'}, ...]"""
    now = now or datetime.now(timezone.utc)
    y, m = now.year, now.month
    # стартовый индекс
    start_idx = None
    for i, (mm, _) in enumerate(MONTHS):
        if m <= mm:
            start_idx = i; break
    if start_idx is None:
        start_idx = 0
        y += 1
    out = []
    year = y
    idx = start_idx
    for _ in range(n):
        mm, code = MONTHS[idx]
        contract = f"KC{code}{year % 100:02d}"
        out.append({"contract": contract, "yahoo": f"{contract}=F"})
        idx += 1
        if idx >= len(MONTHS):
            idx = 0
            year += 1
    return out

@st.cache_data(ttl=60)
def get_kc_futures_board(n: int = 8, seed: int = 0) -> List[Dict]:
    """
    Возвращает [{'contract','yahoo','price_cents_lb','usdkg','asof','is_front'}].
    Не бросает исключений — максимум вернёт пустые/частичные цены.
    """
    try:
        rows = next_kc_contracts(n=n)
    except Exception:
        rows = []

    quotes = {}
    try:
        quotes = yahoo_quote_multi([r["yahoo"] for r in rows])
    except Exception:
        quotes = {}

    out = []
    for i, r in enumerate(rows):
        q = quotes.get(r["yahoo"], {})
        price = q.get("price")  # ¢/lb
        asof = q.get("ts")
        out.append({
            "contract": r["contract"],
            "yahoo": r["yahoo"],
            "price_cents_lb": price,
            "usdkg": arabica_centlb_to_usd_per_kg(price) if price is not None else None,
            "asof": asof,
            "is_front": i == 0,
        })
    return out

# ---------- Robusta: Stooq -> Stooq(front) -> Yahoo 1m ----------
@st.cache_data(ttl=900)
def get_robusta_prices(seed: int = 0,
                       rm_yahoo_candidates: Tuple[str, ...] = ("RC=F", "RM=F")) -> dict:
    data = {}
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

# ---------- Calculator core ----------
def compute_customs_value(incoterm: str, goods_value: float, freight: float, insurance: float) -> float:
    inc = incoterm.upper()
    if inc in {"FOB", "EXW"}: return goods_value + freight + insurance
    if inc == "CFR":          return goods_value + insurance
    return goods_value  # CIF: уже включает фрахт+страховку

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

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Arabica — контракты (Yahoo futures)")
    if "refresh_seed" not in st.session_state: st.session_state["refresh_seed"] = 0
    if st.button("↻ Обновить котировки"): st.session_state["refresh_seed"] += 1

    try:
        board = get_kc_futures_board(n=8, seed=st.session_state["refresh_seed"])
    except Exception:
        board = []

    if not board:
        st.error("Не удалось собрать список контрактов KC.")
        kc_selected_row = None
    else:
        options = [row["contract"] for row in board]
        sel = st.selectbox("Выберите контракт", options, index=0, key="kc_pick")
        kc_selected_row = next((r for r in board if r["contract"] == sel), None)

        # Front-month метрика (даже если селектор про другое)
        if board[0]["price_cents_lb"] is not None:
            st.metric("Текущий контракт (front)", board[0]["contract"])
            st.caption(
                f"{board[0]['contract']} • {board[0]['price_cents_lb']:.2f} ¢/lb • "
                f"≈ {board[0]['usdkg']:.3f} $/кг • as of {fmt_ts(board[0]['asof'])}"
            )
        else:
            st.warning("Котировки сейчас недоступны — попробуйте обновить.")

    st.divider()
    st.subheader("Robusta — котировки")
    rm_main = st.text_input("Yahoo symbol (1)", value="RC=F")
    rm_alt  = st.text_input("Yahoo symbol (2, запасной)", value="RM=F")
    robusta_data = get_robusta_prices(st.session_state["refresh_seed"],
                                      rm_yahoo_candidates=tuple(s for s in [rm_main.strip(), rm_alt.strip()] if s))
    rm = robusta_data.get("RM.F", {})

    colA, colB = st.columns(2)
    with colA:
        # Arabica (по выбранному контракту)
        if kc_selected_row and kc_selected_row["usdkg"] is not None:
            st.metric("Arabica (KC)", f"{kc_selected_row['price_cents_lb']:.2f} ¢/lb")
            st.caption(f"≈ {kc_selected_row['usdkg']:.3f} $/кг • Yahoo ({kc_selected_row['yahoo']}) • as of {fmt_ts(kc_selected_row['asof'])}")
        else:
            st.error("Arabica: нет цены.")
    with colB:
        # Robusta
        if "error" in rm:
            st.error(rm["error"])
        else:
            st.metric("Robusta (RM)", f"{rm.get('last_raw', 0):.2f} {rm.get('unit','')}")
            st.caption(f"≈ {rm.get('usdkg',0):.3f} $/кг • {rm.get('source','?')} • as of {fmt_ts(rm.get('asof'))}")

    st.caption("Котировки ознакомительные. Для сделок сверяйте с брокером/биржей.")

# ---------- Main form ----------
st.header("Калькулятор")

src = st.radio("Источник цены", ["Онлайн фьючерс (Yahoo, контракт KC)", "Введу вручную"], horizontal=True)

col1, col2, col3 = st.columns(3)
with col1:
    instrument = st.selectbox("Инструмент", ["Arabica (KC.F)", "Robusta (RM.F)"])

with col2:
    # базовая цена $/кг из рынка/ручная
    if src == "Онлайн фьючерс (Yahoo, контракт KC)":
        if instrument.startswith("Arabica"):
            if kc_selected_row and kc_selected_row["usdkg"] is not None:
                st.text_input("Базовая цена $/кг (из выбранного контракта)",
                              value=f"{kc_selected_row['usdkg']:.4f}", disabled=True)
                base_usdkg = float(kc_selected_row["usdkg"])
            else:
                st.warning("Нет данных по выбранному контракту — введите вручную.")
                base_usdkg = st.number_input("Базовая цена $/кг", min_value=0.0, value=3.000, step=0.001)
        else:
            market_usdkg = rm.get("usdkg")
            if market_usdkg is None:
                st.warning("Нет данных Robusta — введите вручную.")
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

# Route presets (для CFR/CIF)
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
else:
    st.info("Заполните поля и нажмите «Рассчитать».")
