# =================================================
# STANDARD LIBRARY IMPORTS
# =================================================
import os
import sys
from pathlib import Path
import importlib.util
import yaml

# =================================================
# THIRD-PARTY IMPORTS
# =============================================3====
import streamlit as st
import pandas as pd
import requests
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv

# =================================================
# ENV + PATH SETUP
# =================================================
load_dotenv()  # loads DATABASE_URL from .env

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_PATH))
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

if not CONFIG_PATH.exists():
    raise RuntimeError(f"config.yaml not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# =================================================
# APP IMPORTS
# =================================================
from src.utils.db_utils.inventory_snapshot import get_latest as get_inventory_latest
from src.utils.db_utils.inventory_snapshot import get_oldest as get_inventory_oldest
from src.utils.db_utils.l1_quotes import get_latest as get_l1_latest
from src.adapters.ibkr_client import server_status
from src.tests.test_open_orders import test_get_open_orders
from src.utils.db_utils.fills import get_latest_n_orders_by_side
print("Get latest order fills ",get_latest_n_orders_by_side(10,'B'))
# print("Get inventory latest function:", get_inventory_latest("AAPL"))

# result = test_get_open_orders()
# print("DEBUG result:", result, type(result))
# print("Open buys  orders processed:", result.get("buys"))
# print("Open sells orders processed:", result.get("sells"))
# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Tadawul Market Making Dashboard",
    layout="wide",
)
st_autorefresh(interval=10000, key="data_refresher")
# -------------------------------------------------
# Backend config
# -------------------------------------------------
API_BASE_URL = "http://localhost:8014"
TRIGGER_ENDPOINT = f"{API_BASE_URL}/trigger"


# -------------------------------------------------
# FETCH MARKET QUALITY METRICS
# -------------------------------------------------
@st.cache_data(ttl=10)
def fetch_market_quality_metrics():
    try:
        resp = requests.get(
            f"{API_BASE_URL}/market-quality",
            timeout=20
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.warning(f"Market quality metrics unavailable: {e}")

    return {}

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown(
    """
    <style>
    .app-header {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .app-subheader {
        color: #6c757d;
        margin-bottom: 24px;
    }
    .bids-box, .asks-box {
        border-radius: 10px;
        padding: 0;
        overflow: hidden;
    }
    .side-header {
        font-weight: 700;
        font-size: 15px;
        padding: 8px 12px;
    }
    .bids-box {
        border: 2px solid #2ecc71;
        background-color: #ecfdf3;
    }
    .bids-header {
        background-color: #2ecc71;
        color: white;
    }
    .asks-box {
        border: 2px solid #e74c3c;
        background-color: #fff1f0;
    }
    .asks-header {
        background-color: #e74c3c;
        color: white;
    }
    .sticky-actions {
        position: sticky;
        bottom: 0;
        background: white;
        padding-top: 12px;
        border-top: 1px solid #eee;
        z-index: 100;
    }
    .server-status {
        padding: 10px;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 12px;
        color: white;
    }
    .server-up {
        background-color: #2ecc71;
    }
    .server-down {
        background-color: #e74c3c;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* Remove default Streamlit top padding */
    .block-container {
        padding-top: 0rem !important;
    }

    /* Remove extra padding above the first element */
    header {
        margin-bottom: 0rem !important;
    }

    /* Optional: remove top margin from first markdown */
    div[data-testid="stMarkdown"] {
        margin-top: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# SERVER STATUS BAR (AUTO REFRESH EVERY 10s)
# -------------------------------------------------
try:
    is_server_up = server_status()
except Exception:
    is_server_up = False

if is_server_up:
    st.markdown(
        '<div class="server-status server-up">🟢 Server Connected</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="server-status server-down">🔴 Server Disconnected</div>',
        unsafe_allow_html=True,
    )
    st.link_button(
        "Login",
        "https://localhost:5000",
        type="primary",
    )
SYMBOL = CONFIG.get("trading_asset", {}).get("symbol")
print("Trading symbol:", SYMBOL)
# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<div class="app-header"> Tadawul Market Making Dashboard</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="app-subheader">Symbol: {SYMBOL} </div>',
#     unsafe_allow_html=True,
# )
if st.button(f"Symbol: {SYMBOL}", use_container_width=False):
    st.info(f"Currently selected symbol: {SYMBOL}")


# -------------------------------------------------
# Force-load db module for legacy imports
# -------------------------------------------------
db_path = SRC_PATH / "core" / "db.py"

if db_path.exists() and "db" not in sys.modules:
    spec = importlib.util.spec_from_file_location("db", db_path)
    db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db)
    sys.modules["db"] = db



# =================================================
# FETCH INVENTORY + L1 DATA
# =================================================
SYMBOL = CONFIG.get("trading_asset", {}).get("symbol")

if not SYMBOL:
    raise RuntimeError("trading_asset.symbol missing in config.yaml")


from src.utils.db_utils.inventory_snapshot import get_oldest as get_inventory_oldest

try:
    # ===============================
    # INVENTORY (LATEST + OLDEST) — STRICT
    # ===============================
    latest = get_inventory_latest(SYMBOL)
    oldest = get_inventory_oldest(SYMBOL)

    if not isinstance(latest, dict) or not isinstance(oldest, dict):
        st.warning(
            f"No inventory snapshot found for symbol {SYMBOL}. "
            "Please wait for backend data."
        )
        st.stop()

    # ----- REQUIRED FIELDS -----
    position_qty = latest.get("position_qty")
    cash_latest = latest.get("cash_value")
    cash_oldest = oldest.get("cash_value")
    unrealized_pnl = latest.get("unrealized_pnl")

    if position_qty is None or cash_latest is None or cash_oldest is None:
        st.warning(
            f"Incomplete inventory data for symbol {SYMBOL}. "
            "Waiting for backend sync."
        )
        st.stop()

    # ===============================
    # L1 DATA — SOFT FAIL (THIS IS THE KEY FIX)
    # ===============================
    l1 = get_l1_latest(SYMBOL)

    if not l1:
        st.warning(
            f"No L1 data found for this symbol {SYMBOL}, "
            "please wait for a while or till markets open"
        )
        bid_px = None
        ask_px = None
    else:
        bid_px = l1.get("bid_price") or l1.get("bid_px") or l1.get("best_bid")
        ask_px = l1.get("ask_price") or l1.get("ask_px") or l1.get("best_ask")

    # ===============================
    # PRICE DERIVATIONS
    # ===============================
    if bid_px is None or ask_px is None:
        live_mid_price = None
        live_spread = None
    else:
        live_mid_price = (bid_px + ask_px) / 2
        live_spread = ask_px - bid_px

    # ===============================
    # METRIC CALCULATIONS
    # ===============================
    # Net Position
    net_position = position_qty

    # Exposure = position_qty × mid price
    if bid_px is not None and ask_px is not None:
        mid_price = (bid_px + ask_px) / 2
        exposure = abs(position_qty * mid_price)
    else:
        mid_price = None
        exposure = 0


    # Utilization %
    utilization_pct = (
        (abs(position_qty) / cash_latest) * 100
        if cash_latest else 0
    )

    # Net P&L (DB-BASED — AS YOU DEFINED)
    net_pnl = cash_latest - cash_oldest

    # Realized P&L %
    realized_pnl_pct = (
        (net_pnl / cash_oldest) * 100
        if cash_oldest else 0
    )

except Exception as e:
    st.error(f"Failed to load inventory / L1 data: {e}")
    st.stop()


# =================================================
# METRIC RENDERER
# =================================================
def metric_with_help(label, tooltip, value):
    st.markdown(
        f"""
        <div style="font-size:14px; color:#6c757d;">
            <span title="{tooltip}" style="cursor: help;">
                {label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.metric(label=label, value=value, label_visibility="collapsed")


# =================================================
# DASHBOARD METRICS
# =================================================
c1, c2, c3 = st.columns(3)

with c1:
    metric_with_help(
        "Net Position",
        "Net number of shares currently held",
        f"{net_position:,.0f}",
    )

with c2:
    metric_with_help(
        "Exposure",
        "Total inventory risk in monetary terms.",
        f"{exposure:,.2f}",
    )

with c3:
    metric_with_help(
        "Utilization %",
        "Percentage of allowed inventory risk currently in use.",
        f"{utilization_pct:.2f}%",
    )


c4, c5, c6 = st.columns(3)

with c4:
    metric_with_help(
        "Realized P&L %",
        "Profit or loss from completed trades",
        f"{realized_pnl_pct:.2f}%",
    )

with c5:
    metric_with_help(
        "Unrealized P&L",
        "Mark-to-market profit or loss on current inventory.",
        f"{unrealized_pnl:,.2f}",
    )

with c6:
    metric_with_help(
        "Net P&L",
        "Total trading result.",
        f"{net_pnl:,.2f}",
    )


c7, c8 = st.columns(2)

with c7:
    metric_with_help(
        "Mid Price",
        "Centre price between the best bid and best ask",
        f"{live_mid_price:.4f}" if live_mid_price is not None else "—",
    )

with c8:
    metric_with_help(
        "Spread",
        "Distance between the best ask and the best bid",
        f"{live_spread:.4f}" if live_spread is not None else "—",
    )

st.divider()


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def init_book():
    return pd.DataFrame({"price": [None] * 10, "size": [None] * 10})

def clean_side(df: pd.DataFrame):
    rows = []
    for _, row in df.iterrows():
        if pd.isna(row["price"]) or pd.isna(row["size"]):
            continue

        price = float(row["price"])
        size = float(row["size"])

        if size < 0:
            raise ValueError("Size must be ≥ 0")

        rows.append([price, size])
    return rows

# -------------------------------------------------
# Session state init (order book)
# -------------------------------------------------
if "bids_data" not in st.session_state:
    st.session_state.bids_data = init_book()

if "asks_data" not in st.session_state:
    st.session_state.asks_data = init_book()

# -------------------------------------------------
# Order Book Snapshot
# -------------------------------------------------
st.subheader(" Order Book Snapshot")

col_bids, col_asks = st.columns(2)

with col_bids:
    st.markdown(
        '<div class="bids-box"><div class="side-header bids-header"> BIDS</div>',
        unsafe_allow_html=True,
    )
    bids_df = st.data_editor(
        st.session_state.bids_data,
        num_rows="fixed",
        width="stretch",
        key="bids_editor",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_asks:
    st.markdown(
        '<div class="asks-box"><div class="side-header asks-header"> ASKS</div>',
        unsafe_allow_html=True,
    )
    asks_df = st.data_editor(
        st.session_state.asks_data,
        num_rows="fixed",
        width="stretch",
        key="asks_editor",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Sticky action bar
# -------------------------------------------------
st.markdown('<div class="sticky-actions">', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with st.columns(1)[0]:
    if st.button(" Validate & Submit Snapshot", width="stretch", disabled=not is_server_up):
        try:
            # -------- Clean inputs --------
            bids = clean_side(bids_df)
            asks = clean_side(asks_df)

            if not bids or not asks:
                st.error("Bids and Asks cannot be empty")
                st.stop()

            # -------- Rule 1: Sorting --------
            if bids != sorted(bids, key=lambda x: -x[0]):
                st.error(" Bid prices must be in descending order")
                st.stop()

            if asks != sorted(asks, key=lambda x: x[0]):
                st.error(" Ask prices must be in ascending order")
                st.stop()

            # -------- Rule 2: Best Ask > Best Bid --------
            best_bid = bids[0][0]   # highest bid
            best_ask = asks[0][0]   # lowest ask

            if best_ask <= best_bid:
                st.error(" Best Ask must be greater than Best Bid")
                st.stop()

            # -------- Submit snapshot --------
            payload = {"bids": bids, "asks": asks}

            with st.spinner("Validating & sending snapshot…"):
                response = requests.post(
                    TRIGGER_ENDPOINT,
                    json=payload,
                    timeout=10
                )

            if response.status_code in (200, 201):
                st.success(" Snapshot validated and accepted")
                st.rerun()
            else:
                st.error(response.text)

        except Exception as e:
            st.error(str(e))

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# KPI SECTIONS
# -------------------------------------------------
st.divider()

# =========================
# Market Quality KPIs
# =========================
st.subheader(" Market Quality KPIs")
k1, k2, k3 = st.columns(3)
market_quality = fetch_market_quality_metrics()

spread_tightening = market_quality.get("spread_tightening_contribution")
volatility_stab = market_quality.get("volatility_stabilisation")
presence_ratio = market_quality.get("presence_ratio")
market_dominance = market_quality.get("market_dominance")
fill_ratio = market_quality.get("fill_ratio")

with k1:
    metric_with_help(
        "Spread Tightening Contribution",
        "Estimated reduction in average bid–ask spread attributable to this liquidity provider.",
        f"{spread_tightening:.2f}%" if spread_tightening is not None else "—",
    )

with k2:
    metric_with_help(
        "Apparent Volatility Stabilisation",
        "Relative reduction in short-term price volatility during active quoting periods.",
        f"{volatility_stab:.2f}%" if volatility_stab is not None else "—",
    )

with k3:
    metric_with_help(
        "Market Impact Reduction",
        "Estimated decrease in price impact from trades due to added passive liquidity.",
        "-12.3%",
    )

# =========================
# Liquidity Provisioning KPIs
# =========================
st.subheader(" Liquidity Provisioning KPIs")
l1, l2, l3, l4 = st.columns(4)

with l1:
    metric_with_help(
        "Presence Ratio",
        "Percentage of time both bid and ask quotes are active within the defined competitive range.",
        f"{presence_ratio:.2f}%" if presence_ratio is not None else "—",
    )

with l2:
    metric_with_help(
        "Market Dominance",
        "Share of total displayed liquidity at the best price levels contributed by this strategy.",
        f"{market_dominance:.2f}%" if market_dominance is not None else "—",
    )

with l3:
    metric_with_help(
        "Execution / Fill Ratio",
        "Proportion of submitted orders that receive executions.",
        f"{fill_ratio:.2f}%" if fill_ratio is not None else "—",
    )

with l4:
    metric_with_help(
        "Liquidity Availability Under Stress",
        "Ability to maintain active two-sided quotes during periods of elevated volatility or volume.",
        "High",
    )

# =========================
# Internal Risk KPIs
# =========================
st.subheader(" Internal Risk KPIs")
r1, r2, r3, r4 = st.columns(4)

with r1:
    metric_with_help(
        "Inventory Deviation",
        "Standardized deviation of inventory from its neutral target level.",
        "1.8σ",
    )

with r2:
    metric_with_help(
        "Inventory Variance",
        "Statistical variance of inventory levels over time.",
        "0.024",
    )

with r3:
    metric_with_help(
        "Exposure Half Life",
        "Estimated time required for inventory exposure to reduce by 50% through normal trading flow.",
        "3.6 min",
    )

with r4:
    metric_with_help(
        "Position Risk Breaches",
        "Number of times predefined inventory or exposure limits were exceeded.",
        "0",
    )









def normalize_open_orders(df):
    """
    Normalize backend dataframe → frontend schema
    Scales automatically with backend size
    """
    if df is None or df.empty:
        print("dataframe",df)
        return pd.DataFrame(columns=["timestamp", "price", "size"])
    

    return (
        df.rename(columns={
            "ts": "timestamp",
            "qty": "size",
        })[["timestamp", "price", "size"]]
    )





if "open_buy_orders" not in st.session_state:
    st.session_state.open_buy_orders = pd.DataFrame(
        columns=["timestamp", "price", "size"]
    )

if "open_sell_orders" not in st.session_state:
    st.session_state.open_sell_orders = pd.DataFrame(
        columns=["timestamp", "price", "size"]
    )


result = test_get_open_orders()

if isinstance(result, dict):
    st.session_state.open_buy_orders = normalize_open_orders(result.get("buys"))
    st.session_state.open_sell_orders = normalize_open_orders(result.get("sells"))
else:
    st.error("Invalid open orders response")


# -------------------------------------------------
# Open Orders
# -------------------------------------------------
st.divider()
st.subheader(" Open Orders")

col_buy, col_sell = st.columns(2)

with col_buy:
    st.markdown(
        '<div class="bids-box"><div class="side-header bids-header"> OPEN BUY ORDERS</div>',
        unsafe_allow_html=True,
    )

    st.data_editor(
        st.session_state.open_buy_orders,
        width="stretch",
        key="open_buy_orders_editor",
        disabled=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

with col_sell:
    st.markdown(
        '<div class="asks-box"><div class="side-header asks-header"> OPEN SELL ORDERS</div>',
        unsafe_allow_html=True,
    )

    st.data_editor(
        st.session_state.open_sell_orders,
        width="stretch",
        key="open_sell_orders_editor",
        disabled=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)




# ==================================================
# IMPORTS
# ==================================================
import streamlit as st
import pandas as pd


# ==================================================
# UTILITY: FILLS → DATAFRAME
# ==================================================
def fills_to_df(fills, max_rows=10):
    """
    Converts fill records to a fixed-size dataframe
    Columns:
    timestamp | side | price | size
    """

    empty_df = pd.DataFrame({
        "timestamp": [None] * max_rows,
        "side": [None] * max_rows,
        "price": [None] * max_rows,
        "size": [None] * max_rows,
    })

    if not fills:
        return empty_df

    df = pd.DataFrame(fills)

    # Normalize API fields
    df["price"] = df.get("fill_px")
    df["size"] = df.get("fill_qty")

    # df = df[["side", "price", "size"]]
    if "timestamp" not in df.columns:
        df["timestamp"] = None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Latest fills first
    
    df = df[["timestamp", "side", "price", "size"]]
    df = df.sort_values(by="timestamp", ascending=False)
    df = df.drop(columns=["timestamp"])
    # Pad rows if needed
    if len(df) < max_rows:
        df = pd.concat([df, empty_df.iloc[len(df):]], ignore_index=True)

    return df


# ==================================================
# FETCH RECENT FILLS
# ==================================================
latest_buy_fills = []
latest_sell_fills = []

try:
    # THESE FUNCTIONS MUST EXIST IN YOUR PROJECT
    latest_buy_fills = get_latest_n_orders_by_side(10, "B")
    latest_sell_fills = get_latest_n_orders_by_side(10, "S")

except Exception as e:
    st.error(f"Failed to load recent fills: {e}")


# ==================================================
# STORE IN SESSION STATE
# ==================================================
st.session_state.recent_buy_fills = fills_to_df(latest_buy_fills)
st.session_state.recent_sell_fills = fills_to_df(latest_sell_fills)


# ==================================================
# DEBUG LOGS (SAFE)
# ==================================================
# print("Fetching recent BUY fills:", latest_buy_fills)
# print("Fetching recent SELL fills:", latest_sell_fills)


# ==================================================
# UI — ROW 8 : RECENT FILLS
# ==================================================
st.markdown("## Recent Fills")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        '<div class="bids-box"><div class="side-header bids-header"> BUY FILLS</div>',
        unsafe_allow_html=True,
    )
    st.data_editor(
        st.session_state.recent_buy_fills,
        num_rows="fixed",
        width="stretch",
        key="recent_buy_fills_editor",
    )

with col2:
    st.markdown(
        '<div class="asks-box"><div class="side-header asks-header"> SELL FILLS</div>',
        unsafe_allow_html=True,
    )
    st.data_editor(
        st.session_state.recent_sell_fills,
        num_rows="fixed",
        width="stretch",
        key="recent_sell_fills_editor",
    )
