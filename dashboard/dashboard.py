import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Dashboard E-Commerce Olist", page_icon="🛒", layout="wide")
sns.set_theme(style="white")

HIGHLIGHT, MUTED = "#1f4e79", "#c9d3dd"
SEGMENT_ORDER = [
    "Repeat Customer",
    "Recent High Value",
    "Recent Low Value",
    "Lapsed High Value",
    "Lapsed Low Value",
]


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_data.csv")
    return pd.read_csv(path, parse_dates=["order_purchase_timestamp"])


def category_summary(df):
    out = (
        df.groupby("product_category_name_english")
        .agg(revenue=("price", "sum"), orders=("order_id", "nunique"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    out["share_pct"] = out["revenue"] / out["revenue"].sum() * 100
    return out


def state_summary(df):
    order_level = df.drop_duplicates("order_id")
    out = (
        order_level.groupby("customer_state")
        .agg(
            customers=("customer_unique_id", "nunique"),
            orders=("order_id", "nunique"),
            avg_delivery_days=("delivery_time_days", "mean"),
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean"),
        )
        .reset_index()
    )
    revenue = df.groupby("customer_state")["price"].sum().rename("revenue").reset_index()
    return (
        out.merge(revenue, on="customer_state")
        .sort_values("customers", ascending=False)
        .reset_index(drop=True)
    )


def compute_rfm(df):
    reference_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("price", "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = (reference_date - rfm["last_purchase"]).dt.days
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    conditions = [
        rfm["frequency"] >= 2,
        (rfm["r_score"] >= 3) & (rfm["m_score"] >= 3),
        rfm["r_score"] >= 3,
        rfm["m_score"] >= 3,
    ]
    choices = ["Repeat Customer", "Recent High Value", "Recent Low Value", "Lapsed High Value"]
    rfm["segment"] = np.select(conditions, choices, default="Lapsed Low Value")
    summary = (
        rfm.groupby("segment")
        .agg(customers=("customer_unique_id", "count"), revenue=("monetary", "sum"))
        .reindex(SEGMENT_ORDER)
        .fillna(0)
        .reset_index()
    )
    summary["customers_pct"] = summary["customers"] / summary["customers"].sum() * 100
    summary["revenue_pct"] = summary["revenue"] / summary["revenue"].sum() * 100
    return rfm, summary


# ---------------------------------------------------------------- plots
def barh_highlight(ax, labels, values, highlight_n, fmt, xlabel):
    colors = [HIGHLIGHT if i < highlight_n else MUTED for i in range(len(values))]
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v, i, " " + fmt(v), va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.15 if len(values) else 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.tick_params(left=False)
    sns.despine(ax=ax, left=True)


def plot_category(cat_df, top_n):
    top = cat_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 0.45 * top_n + 1.5))
    barh_highlight(
        ax, top["product_category_name_english"], top["revenue"] / 1e6,
        min(5, top_n), lambda v: f"{v:,.2f}", "Revenue (R$ juta)",
    )
    ax.set_title(f"Top {top_n} kategori produk berdasarkan revenue", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_state_bars(state_df):
    top_customers = state_df.head(10)
    slowest = state_df[state_df["orders"] >= 100].sort_values("avg_delivery_days", ascending=False).head(10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    barh_highlight(
        axes[0], top_customers["customer_state"], top_customers["customers"], 3,
        lambda v: f"{v:,.0f}", "Jumlah pelanggan unik",
    )
    axes[0].set_title("10 state dengan pelanggan terbanyak", loc="left", fontweight="bold")
    barh_highlight(
        axes[1], slowest["customer_state"], slowest["avg_delivery_days"], 3,
        lambda v: f"{v:.1f}", "Rata-rata waktu pengiriman (hari)",
    )
    axes[1].set_title("10 state dengan pengiriman terlama", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_geo(state_df):
    geo = state_df.dropna(subset=["lat", "lng"])
    fig, ax = plt.subplots(figsize=(8, 8))
    sizes = geo["customers"] / geo["customers"].max() * 2500 + 60
    sc = ax.scatter(
        geo["lng"], geo["lat"], s=sizes, c=geo["avg_delivery_days"],
        cmap="YlOrRd", edgecolor="black", linewidth=0.5, alpha=0.85,
    )
    for _, row in geo.iterrows():
        ax.annotate(row["customer_state"], (row["lng"], row["lat"]), ha="center", va="center", fontsize=8)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Rata-rata waktu pengiriman (hari)")
    ax.set_title("Sebaran pelanggan dan waktu pengiriman per state", loc="left", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_rfm(summary):
    top_revenue = summary["revenue_pct"].max()
    colors = [HIGHLIGHT if v == top_revenue else MUTED for v in summary["revenue_pct"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].barh(summary["segment"], summary["customers_pct"], color=MUTED)
    axes[0].set_title("Proporsi pelanggan per segmen", loc="left", fontweight="bold")
    axes[0].set_xlabel("Persentase pelanggan (%)")
    axes[1].barh(summary["segment"], summary["revenue_pct"], color=colors)
    axes[1].set_title("Proporsi revenue per segmen", loc="left", fontweight="bold")
    axes[1].set_xlabel("Persentase revenue (%)")
    for ax, col in zip(axes, ["customers_pct", "revenue_pct"]):
        ax.invert_yaxis()
        ax.set_xlim(0, max(summary[col].max() * 1.15, 1))
        for i, v in enumerate(summary[col]):
            ax.text(v, i, f" {v:.1f}%", va="center", fontsize=9)
        ax.set_ylabel("")
        ax.tick_params(left=False)
    sns.despine(left=True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- layout
st.title("🛒 Dashboard Analisis E-Commerce Olist")
st.caption("Pesanan berstatus delivered. Revenue = total harga produk (tanpa ongkos kirim).")

df = load_data()
min_date = df["order_purchase_timestamp"].min().date()
max_date = df["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.header("Filter")
    date_range = st.date_input(
        "Rentang tanggal pembelian",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Pilih tanggal awal dan tanggal akhir pada filter di sebelah kiri.")
    st.stop()

start_date, end_date = date_range
mask = (df["order_purchase_timestamp"].dt.date >= start_date) & (
    df["order_purchase_timestamp"].dt.date <= end_date
)
filtered = df[mask]

if filtered.empty:
    st.warning("Tidak ada data pada rentang tanggal yang dipilih.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total revenue", f"R$ {filtered['price'].sum() / 1e6:,.2f} juta")
col2.metric("Total pesanan", f"{filtered['order_id'].nunique():,}")
col3.metric("Pelanggan unik", f"{filtered['customer_unique_id'].nunique():,}")
col4.metric("Rata-rata pengiriman", f"{filtered.drop_duplicates('order_id')['delivery_time_days'].mean():.1f} hari")

tab_cat, tab_geo, tab_rfm = st.tabs(["Kategori Produk", "Wilayah (Geospatial)", "Segmentasi Pelanggan (RFM)"])

with tab_cat:
    st.subheader("Kategori produk dengan revenue tertinggi")
    top_n = st.slider("Jumlah kategori yang ditampilkan", min_value=5, max_value=15, value=10)
    cat_df = category_summary(filtered)
    fig = plot_category(cat_df, top_n)
    st.pyplot(fig)
    plt.close(fig)
    top5_share = cat_df.head(5)["share_pct"].sum()
    st.markdown(
        f"**Insight:** kategori teratas adalah **{cat_df.loc[0, 'product_category_name_english']}** "
        f"(R$ {cat_df.loc[0, 'revenue'] / 1e6:,.2f} juta). "
        f"Lima kategori teratas menyumbang **{top5_share:.1f}%** dari total revenue pada periode terpilih."
    )

with tab_geo:
    st.subheader("Pelanggan dan waktu pengiriman per negara bagian")
    state_df = state_summary(filtered)
    fig = plot_state_bars(state_df)
    st.pyplot(fig)
    plt.close(fig)
    fig = plot_geo(state_df)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Ukuran lingkaran = jumlah pelanggan unik. Warna = rata-rata waktu pengiriman (makin gelap makin lama).")
    top_state = state_df.iloc[0]
    slow = state_df[state_df["orders"] >= 100].sort_values("avg_delivery_days", ascending=False)
    insight = (
        f"**Insight:** state dengan pelanggan unik terbanyak adalah **{top_state['customer_state']}** "
        f"({top_state['customers']:,.0f} pelanggan)."
    )
    if not slow.empty:
        insight += (
            f" Pengiriman terlama (minimal 100 pesanan) ada di **{slow.iloc[0]['customer_state']}** "
            f"dengan rata-rata {slow.iloc[0]['avg_delivery_days']:.1f} hari."
        )
    st.markdown(insight)

with tab_rfm:
    st.subheader("Segmentasi pelanggan dengan RFM")
    st.caption(
        "Skor Recency dan Monetary dibagi kuartil (binning). Pelanggan dengan frequency >= 2 masuk "
        "Repeat Customer; sisanya dikelompokkan manual berdasarkan skor R dan M."
    )
    if filtered["customer_unique_id"].nunique() < 4:
        st.warning("Data terlalu sedikit untuk segmentasi RFM. Perluas rentang tanggal.")
    else:
        rfm_df, rfm_summary = compute_rfm(filtered)
        fig = plot_rfm(rfm_summary)
        st.pyplot(fig)
        plt.close(fig)
        repeat_pct = (rfm_df["frequency"] >= 2).mean() * 100
        best = rfm_summary.sort_values("revenue_pct", ascending=False).iloc[0]
        st.markdown(
            f"**Insight:** hanya **{repeat_pct:.1f}%** pelanggan yang berbelanja lebih dari sekali. "
            f"Segmen dengan kontribusi revenue terbesar adalah **{best['segment']}** ({best['revenue_pct']:.1f}%)."
        )
        st.dataframe(
            rfm_summary.rename(columns={
                "segment": "Segmen", "customers": "Pelanggan", "revenue": "Revenue (R$)",
                "customers_pct": "% Pelanggan", "revenue_pct": "% Revenue",
            }).round(2),
            use_container_width=True,
            hide_index=True,
        )

st.caption("Dibuat dengan Streamlit untuk Proyek Analisis Data Dicoding.")
