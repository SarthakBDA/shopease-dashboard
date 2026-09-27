"""
ShopEase Dynamic Analytical Dashboard  -  FBDA Group Project
Group ID 001_033_040  |  seed 1033040  |  2,500 random records
Run:  streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import shopease_core as sc

# ----------------------------------------------------------------------------- look & feel
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SEQ = sns.light_palette("#2a78d6", as_cmap=True)
DIV = sns.diverging_palette(250, 20, s=75, l=50, center="light", as_cmap=True)

sns.set_theme(style="whitegrid", palette=PALETTE, rc={
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "lines.linewidth": 2,
})

st.set_page_config(page_title="ShopEase Analytics | Group 001_033_040", page_icon="🛒", layout="wide")
st.markdown("""
<style>
div[data-testid="stMetric"] {background:#f4f3ef;border-radius:10px;padding:12px 14px;border:1px solid #e6e5e0}
div[data-testid="stMetricValue"] {font-size:1.55rem}
.block-container {padding-top:1.6rem}
.small {color:#52514e;font-size:0.85rem}
</style>""", unsafe_allow_html=True)


def fig(w=7, h=4):
    f, ax = plt.subplots(figsize=(w, h))
    return f, ax


def show(f):
    f.tight_layout()
    st.pyplot(f, width="stretch")
    plt.close(f)


def color_map(levels):
    return {lv: PALETTE[i % len(PALETTE)] for i, lv in enumerate(sorted(map(str, levels)))}


def fmt_money(x):
    return f"EGP {x:,.0f}"


def fmt_short(x):
    return f"EGP {x / 1e6:.2f}M" if abs(x) >= 1e6 else (f"EGP {x / 1e3:.1f}K" if abs(x) >= 1e4 else f"EGP {x:,.0f}")


def result_card(res: dict):
    """Pretty-print one hypothesis-test result."""
    p = res.get("p-value", np.nan)
    decision = res.get("Decision (α=0.05)", "")
    icon = "🟥" if decision == "Reject H0" else "🟩"
    ptxt = "n/a (compare with critical value)" if pd.isna(p) else (f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}")
    st.markdown(f"**{res['Test']}**  \n{icon} **{decision}** · statistic = `{res['Statistic']:.4f}` · p-value = `{ptxt}`")
    extra = {k: v for k, v in res.items() if k not in ("Test", "Statistic", "p-value", "Decision (α=0.05)")}
    if extra:
        st.dataframe(pd.DataFrame([extra]).round(4), hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading and cleaning ShopEase data…")
def load(seed: int, n: int):
    raw, cl, log, sample = sc.get_data(seed=seed, n=n)
    return raw, cl, log, sample


st.sidebar.title("🛒 ShopEase Analytics")
st.sidebar.caption(f"Group ID **{sc.GROUP_ID}** · seed **{sc.SEED}**")
base = st.sidebar.radio("Analysis base", ["Group sample (2,500 records)", "Full cleaned data (10,000)"], index=0)

raw, clean_df, clean_log, sample = load(sc.SEED, sc.SAMPLE_SIZE)
data = sample if base.startswith("Group") else clean_df

# ----------------------------------------------------------------------------- filters
st.sidebar.header("Filters")
dmin, dmax = data["Order Date"].min().date(), data["Order Date"].max().date()
dr = st.sidebar.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax)
keep_undated = st.sidebar.checkbox("Keep orders with invalid/missing date", value=True)


def ms(label, col):
    opts = sorted(data[col].dropna().astype(str).unique())
    return st.sidebar.multiselect(label, opts, default=opts)


f_city = ms("City", "City")
f_cat = ms("Category", "Category")
f_gender = ms("Gender", "Gender")
f_pay = ms("Payment method", "Payment Method")
f_status = ms("Order status", "Order Status")
a_lo, a_hi = st.sidebar.slider("Customer age", 18, 90, (18, 90))
disc_opts = sorted(data["Discount"].unique())
f_disc = st.sidebar.multiselect("Discount level", disc_opts, default=disc_opts, format_func=lambda v: f"{v:.0%}")

mask = (data["City"].isin(f_city) & data["Category"].isin(f_cat) & data["Gender"].isin(f_gender)
        & data["Payment Method"].isin(f_pay) & data["Order Status"].isin(f_status)
        & data["Customer Age"].between(a_lo, a_hi) & data["Discount"].isin(f_disc))
if isinstance(dr, (list, tuple)) and len(dr) == 2:
    in_range = data["Order Date"].between(pd.Timestamp(dr[0]), pd.Timestamp(dr[1]) + pd.Timedelta(days=1))
    mask &= in_range | (data["Order Date"].isna() & keep_undated)
df = data[mask].copy()

st.sidebar.markdown(f"**{len(df):,}** of {len(data):,} records selected")
st.sidebar.download_button("⬇️ Download filtered data (CSV)", df.to_csv(index=False).encode(),
                           "shopease_filtered.csv", "text/csv")
st.sidebar.markdown(f"<span class='small'>Data: <a href='{sc.KAGGLE_URL}'>Kaggle – ShopEase Raw Orders</a></span>",
                    unsafe_allow_html=True)

if len(df) < 20:
    st.warning("Fewer than 20 records match the filters – widen the selection to run the analysis.")
    st.stop()

# ----------------------------------------------------------------------------- header + KPIs
st.title("ShopEase Dynamic Analytical Dashboard")
st.caption("Descriptive · Visual · Inferential · Causal analysis of e-commerce orders "
           f"(Jan–Jun 2026, Egypt) · base: {base} · every chart and test responds to the sidebar filters")

rev = df["Total Amount"].sum()
k = st.columns(6)
k[0].metric("Net revenue", fmt_short(rev), help=fmt_money(rev))
k[1].metric("Orders", f"{len(df):,}")
k[2].metric("Avg order value", fmt_money(df["Total Amount"].mean()))
k[3].metric("Avg rating", f"{df['Rating'].mean():.2f} ★")
k[4].metric("Cancellation rate", f"{df['Is Cancelled'].mean():.1%}")
k[5].metric("Discounts", fmt_short(df["Discount Amount"].sum()),
            help=f"{df['Discount Amount'].sum() / df['Gross Amount'].sum():.1%} of gross sales given away as discount")

tabs = st.tabs(["📊 Overview", "🗂️ Data", "📐 Descriptive", "🎨 Visual explorer",
                "🧪 Inferential", "📈 Regression", "💡 Insights"])

# ============================================================================= 1. OVERVIEW
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        g = df.groupby("Category")["Total Amount"].sum().sort_values()
        f, ax = fig(6, 3.6)
        ax.barh(g.index, g.values, color=PALETTE[0], height=0.6)
        for y, v in enumerate(g.values):
            ax.text(v, y, f"  {v / rev:.0%}", va="center", color=INK2, fontsize=9)
        ax.set_title("Revenue by category")
        ax.set_xlabel("Net revenue (EGP)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
        show(f)
    with c2:
        m = df.dropna(subset=["Order Month"]).groupby("Order Month").agg(Revenue=("Total Amount", "sum"),
                                                                          Orders=("OrderID", "count"))
        f, ax = fig(6, 3.6)
        ax.plot(m.index, m["Revenue"], marker="o", color=PALETTE[0])
        ax.set_title("Monthly net revenue")
        ax.set_ylabel("EGP")
        ax.set_ylim(0, m["Revenue"].max() * 1.15)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
        show(f)
    c3, c4, c5 = st.columns([1.1, 1, 1])
    with c3:
        g = df.groupby("City").agg(Orders=("OrderID", "count"), AOV=("Total Amount", "mean")).sort_values("Orders")
        f, ax = fig(5, 3.4)
        ax.barh(g.index, g["Orders"], color=PALETTE[0], height=0.6)
        ax.set_title("Orders by city")
        ax.set_xlabel("Orders")
        show(f)
    with c4:
        s = df["Order Status"].value_counts()
        cm = {"Delivered": PALETTE[2], "Pending": PALETTE[3], "Cancelled": PALETTE[7]}
        f, ax = fig(4, 3.4)
        ax.pie(s.values, labels=s.index, autopct="%1.1f%%", startangle=90, colors=[cm.get(i, GRID) for i in s.index],
               wedgeprops={"width": 0.45, "edgecolor": SURFACE, "linewidth": 2}, textprops={"color": INK})
        ax.set_title("Order status mix")
        show(f)
    with c5:
        s = df["Rating"].value_counts().sort_index()
        f, ax = fig(4, 3.4)
        ax.bar(s.index.astype(int).astype(str), s.values, color=PALETTE[0], width=0.6)
        ax.set_title("Rating distribution (delivered)")
        ax.set_xlabel("Stars")
        show(f)
    st.markdown("##### Top products by revenue")
    tp = (df.groupby(["Product", "Category"]).agg(Revenue=("Total Amount", "sum"), Orders=("OrderID", "count"),
                                                   Units=("Quantity", "sum"), AvgRating=("Rating", "mean"))
          .sort_values("Revenue", ascending=False).head(10).reset_index())
    tp["Share of revenue"] = tp["Revenue"] / rev
    st.dataframe(tp.style.format({"Revenue": "{:,.0f}", "AvgRating": "{:.2f}", "Share of revenue": "{:.1%}"}),
                 hide_index=True, width="stretch")

# ============================================================================= 2. DATA
with tabs[1]:
    st.subheader("Description of data")
    info = pd.DataFrame({
        "Item": ["Data name", "Source", "File size", "Data type", "Dataset span", "Raw dimension",
                 "Cleaned dimension", "Analysis sample", "Seed"],
        "Value": ["ShopEase Raw Orders (shopease_raw_orders.csv)", sc.KAGGLE_URL, "≈ 1.06 MB (1,108,848 bytes)",
                  "Cross-sectional (order-level records; order dates Jan–Jun 2026)", "Static snapshot",
                  f"{raw.shape[1]} variables × {raw.shape[0]:,} records",
                  f"{clean_df.shape[1]} variables (incl. derived) × {clean_df.shape[0]:,} records",
                  f"{len(sample):,} random records", f"random.seed / random_state = {sc.SEED}"]})
    st.dataframe(info, hide_index=True, width="stretch")
    st.subheader("Variable dictionary")
    vd = pd.DataFrame([
        ["OrderID", "Unique order number", "Non-categorical – Index (nominal)", "Integer"],
        ["CustomerID", "Customer code", "Categorical – Nominal (identifier)", "Text"],
        ["Order Date", "Timestamp the order was placed", "Date-time", "Text → datetime"],
        ["Customer Age", "Age of the customer (years)", "Non-categorical – Ratio", "Integer"],
        ["Gender", "Customer gender", "Categorical – Nominal", "Text"],
        ["City", "Delivery city (Egypt)", "Categorical – Nominal", "Text"],
        ["Category", "Product category", "Categorical – Nominal", "Text"],
        ["Product", "Product name", "Categorical – Nominal", "Text"],
        ["Quantity", "Units ordered", "Non-categorical – Ratio (discrete)", "Integer"],
        ["Unit Price", "List price per unit (EGP)", "Non-categorical – Ratio", "Integer"],
        ["Discount", "Discount fraction (0–20%)", "Categorical – Ordinal (5 levels) / Ratio", "Decimal"],
        ["Payment Method", "Cash, Visa, Mastercard, Wallet, Bank Transfer", "Categorical – Nominal", "Text"],
        ["Order Status", "Delivered / Pending / Cancelled", "Categorical – Nominal", "Text"],
        ["DeliveryDate", "Timestamp of delivery", "Date-time", "Text → datetime"],
        ["Rating", "Customer rating 1–5 (delivered orders)", "Categorical – Ordinal", "Integer"],
        ["Total Amount", "Net order value = Qty × Price × (1 – Discount)", "Non-categorical – Ratio", "Decimal"],
        ["Delivery Days*", "Delivery Date – Order Date (days)", "Non-categorical – Ratio", "Integer"],
        ["Age Group*, Order Month*, Order Weekday*, Order Hour*", "Derived time / demographic bands",
         "Categorical – Ordinal", "Text / Integer"],
    ], columns=["Variable", "Definition", "Variable type", "Format"])
    st.dataframe(vd, hide_index=True, width="stretch")
    st.caption("* derived during cleaning")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Missing data – raw file")
        mi = raw.isna().sum()
        extra = pd.Series({"Order Date": int((raw["Order Date"].astype(str).str.strip() == "not available").sum())})
        mtab = pd.DataFrame({"Missing (blank)": mi, "Text 'not available'": extra}).fillna(0).astype(int)
        mtab = mtab[mtab.sum(axis=1) > 0]
        mtab["% of rows"] = 100 * mtab.sum(axis=1) / len(raw)
        st.dataframe(mtab.style.format({"% of rows": "{:.2f}%"}), width="stretch")
    with c2:
        st.subheader("Cleaning log")
        st.dataframe(clean_log, hide_index=True, width="stretch")
    st.subheader("Filtered records")
    st.dataframe(df.drop(columns=["Log Total Amount"]).head(500), width="stretch", height=320)

# ============================================================================= 3. DESCRIPTIVE
with tabs[2]:
    st.subheader("Non-categorical variables")
    sel = st.multiselect("Variables", sc.NUM_COLS, default=sc.NUM_COLS, key="desc_num")
    if sel:
        ns = sc.numeric_summary(df, sel)
        st.dataframe(ns.style.format("{:,.3f}"), width="stretch")
        st.caption("Count · Min · Max · Percentiles | Central tendency: Mean · Median · Mode | "
                   "Dispersion: Range · IQR · Std Dev · CV · Skewness · Excess kurtosis")
    c1, c2 = st.columns(2)
    for col, method in zip((c1, c2), ("pearson", "spearman")):
        with col:
            cm_ = sc.correlations(df, sel if len(sel) > 1 else sc.NUM_COLS, method)
            f, ax = fig(6, 5)
            sns.heatmap(cm_, annot=True, fmt=".2f", cmap=DIV, vmin=-1, vmax=1, center=0, ax=ax,
                        linewidths=1, linecolor=SURFACE, cbar_kws={"shrink": .7}, annot_kws={"size": 8})
            ax.set_title(f"{method.title()} correlation")
            show(f)
    st.subheader("Categorical variables")
    st.dataframe(sc.categorical_overview(df), width="stretch")
    cc = st.selectbox("Frequency table for", sc.CAT_COLS, index=2)
    ft = sc.categorical_summary(df, cc)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.dataframe(ft.style.format({"Relative Frequency": "{:.2%}", "Cumulative %": "{:.1f}"}),
                     width="stretch")
    with c2:
        f, ax = fig(6, max(2.6, 0.32 * len(ft)))
        ft_s = ft.sort_values("Frequency")
        ax.barh(ft_s.index.astype(str), ft_s["Frequency"], color=PALETTE[0], height=0.6)
        ax.set_title(f"Frequency of {cc}")
        show(f)

# ============================================================================= 4. VISUAL EXPLORER
with tabs[3]:
    kind = st.radio("Chart type", ["Scatter", "Line", "Box-Whisker", "Violin", "Heat map", "Pair plot",
                                   "Bar", "Histogram", "Pie"], horizontal=True)
    hue_opts = ["(none)", "Gender", "Category", "Order Status", "City", "Payment Method", "Age Group"]
    c1, c2, c3 = st.columns(3)
    if kind == "Scatter":
        x = c1.selectbox("X", sc.NUM_COLS, index=0)
        y = c2.selectbox("Y", sc.NUM_COLS, index=4)
        hue = c3.selectbox("Colour by", hue_opts, index=0)
        f, ax = fig(9, 4.6)
        d = df.dropna(subset=[x, y])
        if hue == "(none)":
            sns.regplot(data=d, x=x, y=y, ax=ax, scatter_kws={"s": 14, "alpha": .45, "color": PALETTE[0]},
                        line_kws={"color": PALETTE[1]}, x_jitter=.15 if d[x].nunique() < 15 else 0)
        else:
            sns.scatterplot(data=d, x=x, y=y, hue=hue, ax=ax, s=16, alpha=.6, palette=color_map(d[hue].unique()))
        ax.set_title(f"{y} vs {x}")
        show(f)
    elif kind == "Line":
        grain = c1.selectbox("Time grain", ["Day", "Week", "Month"], index=1)
        metric = c2.selectbox("Metric", ["Revenue", "Orders", "Avg order value", "Avg rating"])
        hue = c3.selectbox("Split by", ["(none)", "Category", "Gender", "City", "Order Status"])
        rule = {"Day": "D", "Week": "W-MON", "Month": "MS"}[grain]
        d = df.dropna(subset=["Order Date"]).set_index("Order Date")
        agg = {"Revenue": ("Total Amount", "sum"), "Orders": ("OrderID", "count"),
               "Avg order value": ("Total Amount", "mean"), "Avg rating": ("Rating", "mean")}[metric]
        f, ax = fig(10, 4.4)
        if hue == "(none)":
            s = d.resample(rule)[agg[0]].agg(agg[1])
            ax.plot(s.index, s.values, color=PALETTE[0], marker="o" if grain != "Day" else None, ms=4)
        else:
            cmap = color_map(d[hue].unique())
            for lv, g in d.groupby(hue):
                s = g.resample(rule)[agg[0]].agg(agg[1])
                ax.plot(s.index, s.values, label=lv, color=cmap[str(lv)])
            ax.legend(frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(.5, -.12))
        ax.set_title(f"{metric} by {grain.lower()}")
        show(f)
    elif kind in ("Box-Whisker", "Violin"):
        y = c1.selectbox("Numeric variable", sc.NUM_COLS, index=4)
        x = c2.selectbox("Group by", [h for h in hue_opts if h != "(none)"], index=1)
        logy = c3.checkbox("Log scale", value=(y in ("Total Amount", "Unit Price")))
        d = df.dropna(subset=[y])
        order = sorted(d[x].dropna().astype(str).unique())
        f, ax = fig(10, 4.6)
        if kind == "Box-Whisker":
            sns.boxplot(data=d, x=x, y=y, ax=ax, order=order, palette=color_map(order), width=.55, fliersize=2)
        else:
            sns.violinplot(data=d, x=x, y=y, ax=ax, order=order, palette=color_map(order), inner="quartile", cut=0)
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"{kind}: {y} by {x}")
        show(f)
    elif kind == "Heat map":
        mode = c1.selectbox("Heat map of", ["Correlation matrix", "Revenue: Category × City",
                                            "Orders: Weekday × Hour", "Avg rating: Category × Delivery Days"])
        f, ax = fig(10, 5)
        if mode == "Correlation matrix":
            sns.heatmap(df[sc.NUM_COLS].corr(), annot=True, fmt=".2f", cmap=DIV, center=0, vmin=-1, vmax=1, ax=ax,
                        linewidths=1, linecolor=SURFACE)
        elif mode == "Revenue: Category × City":
            pt = df.pivot_table(index="Category", columns="City", values="Total Amount", aggfunc="sum") / 1e3
            sns.heatmap(pt, annot=True, fmt=",.0f", cmap=SEQ, ax=ax, linewidths=1, linecolor=SURFACE,
                        cbar_kws={"label": "EGP '000"})
        elif mode == "Orders: Weekday × Hour":
            pt = df.pivot_table(index="Order Weekday", columns="Order Hour", values="OrderID", aggfunc="count",
                                observed=False)
            sns.heatmap(pt, cmap=SEQ, ax=ax, linewidths=.5, linecolor=SURFACE)
        else:
            pt = df.pivot_table(index="Category", columns="Delivery Days", values="Rating", aggfunc="mean")
            sns.heatmap(pt, annot=True, fmt=".2f", cmap=SEQ, ax=ax, linewidths=1, linecolor=SURFACE)
        ax.set_title(mode)
        show(f)
    elif kind == "Pair plot":
        vars_ = c1.multiselect("Variables", sc.NUM_COLS,
                               default=["Customer Age", "Quantity", "Total Amount", "Rating"])
        hue = c2.selectbox("Colour by", ["(none)", "Gender", "Order Status", "Category"], index=1)
        n = c3.slider("Points (random subset for speed)", 200, min(2500, len(df)), min(800, len(df)))
        if len(vars_) >= 2:
            d = df.dropna(subset=vars_).sample(min(n, len(df.dropna(subset=vars_))), random_state=sc.SEED)
            g = sns.pairplot(d, vars=vars_, hue=None if hue == "(none)" else hue, corner=True, height=2.1,
                             palette=None if hue == "(none)" else color_map(d[hue].unique()),
                             plot_kws={"s": 12, "alpha": .5})
            g.figure.set_facecolor(SURFACE)
            st.pyplot(g.figure, width="stretch")
            plt.close(g.figure)
    elif kind == "Bar":
        x = c1.selectbox("Category axis", ["Category", "City", "Product", "Payment Method", "Order Status",
                                           "Age Group", "Order Weekday", "Gender"])
        metric = c2.selectbox("Metric", ["Revenue", "Orders", "Avg order value", "Avg rating", "Cancellation rate"])
        hue = c3.selectbox("Split by", ["(none)", "Gender", "Order Status", "Category"])
        agg = {"Revenue": ("Total Amount", "sum"), "Orders": ("OrderID", "count"),
               "Avg order value": ("Total Amount", "mean"), "Avg rating": ("Rating", "mean"),
               "Cancellation rate": ("Is Cancelled", "mean")}[metric]
        f, ax = fig(10, 4.6)
        if hue == "(none)":
            s = df.groupby(x, observed=True)[agg[0]].agg(agg[1]).sort_values(ascending=False)
            ax.bar(s.index.astype(str), s.values, color=PALETTE[0], width=.6)
        else:
            s = df.groupby([x, hue], observed=True)[agg[0]].agg(agg[1]).reset_index()
            sns.barplot(data=s, x=x, y=agg[0], hue=hue, ax=ax, palette=color_map(s[hue].unique()),
                        edgecolor=SURFACE, linewidth=2)
            ax.legend(frameon=False)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by {x}")
        plt.setp(ax.get_xticklabels(), rotation=30 if x == "Product" else 0, ha="right" if x == "Product" else "center")
        show(f)
    elif kind == "Histogram":
        x = c1.selectbox("Variable", sc.NUM_COLS + ["Log Total Amount"], index=4)
        bins = c2.slider("Bins", 5, 80, 30)
        kde = c3.checkbox("Density curve", value=True)
        f, ax = fig(10, 4.4)
        sns.histplot(df[x].dropna(), bins=bins, kde=kde, ax=ax, color=PALETTE[0], edgecolor=SURFACE)
        ax.axvline(df[x].mean(), color=PALETTE[1], ls="--", lw=1.5, label=f"mean {df[x].mean():,.2f}")
        ax.axvline(df[x].median(), color=PALETTE[6], ls=":", lw=1.5, label=f"median {df[x].median():,.2f}")
        ax.legend(frameon=False)
        ax.set_title(f"Distribution of {x}")
        show(f)
    elif kind == "Pie":
        x = c1.selectbox("Share of", ["Category", "Order Status", "Payment Method", "City", "Gender", "Age Group"])
        metric = c2.selectbox("Measured by", ["Orders", "Revenue"])
        s = (df[x].value_counts() if metric == "Orders" else df.groupby(x, observed=True)["Total Amount"].sum())
        s = s.sort_values(ascending=False)
        f, ax = fig(6, 5)
        ax.pie(s.values, labels=s.index.astype(str), autopct="%1.1f%%", startangle=90,
               colors=[color_map(s.index)[str(i)] for i in s.index],
               wedgeprops={"edgecolor": SURFACE, "linewidth": 2})
        ax.set_title(f"Share of {metric.lower()} by {x}")
        show(f)

# ============================================================================= 5. INFERENTIAL
with tabs[4]:
    alpha = st.select_slider("Significance level α", [0.01, 0.05, 0.10], value=0.05)
    st.caption("Decisions below are printed at α = 0.05; compare the p-value with your chosen α.")
    t_ci, t_mean, t_var, t_prop, t_corr, t_norm, t_np, t_chi = st.tabs(
        ["Confidence interval", "Test of mean", "Test of variance", "Test of proportion",
         "Test of correlation", "Normality", "Non-parametric", "Chi-square"])
    two_level = {"Gender": ("Male", "Female"), "Discount Given": (1, 0)}

    with t_ci:
        c1, c2 = st.columns(2)
        v = c1.selectbox("Mean of", sc.NUM_COLS, index=4, key="ci_v")
        conf = c2.select_slider("Confidence", [0.90, 0.95, 0.99], value=0.95)
        r = sc.ci_mean(df[v], conf)
        st.markdown(f"**{conf:.0%} CI for mean {v}:** {r['Mean']:,.3f}  →  "
                    f"[{r['Lower']:,.3f}, {r['Upper']:,.3f}]  (n = {r['n']}, SE = {r['SE']:,.3f})")
        rows = []
        for lv, g in df.groupby("Category"):
            rr = sc.ci_mean(g[v], conf)
            rows.append({"Category": lv, **rr})
        cdf = pd.DataFrame(rows)
        f, ax = fig(8, 3.2)
        ax.errorbar(cdf["Mean"], cdf["Category"], xerr=[cdf["Mean"] - cdf["Lower"], cdf["Upper"] - cdf["Mean"]],
                    fmt="o", color=PALETTE[0], capsize=4, ms=7)
        ax.axvline(r["Mean"], color=INK2, ls="--", lw=1)
        ax.set_title(f"{conf:.0%} CI of mean {v} by category")
        show(f)
        p = st.selectbox("Proportion of", ["Is Cancelled", "Is Delivered", "Discount Given", "High Rating"])
        x = df[p].dropna()
        rp = sc.ci_prop(int(x.sum()), len(x), conf)
        st.markdown(f"**{conf:.0%} Wilson CI for proportion {p}:** {rp['Proportion']:.3%} → "
                    f"[{rp['Lower']:.3%}, {rp['Upper']:.3%}] (n = {rp['n']})")

    with t_mean:
        c1, c2, c3 = st.columns(3)
        v = c1.selectbox("Variable", sc.NUM_COLS, index=4, key="tm_v")
        mu = c2.number_input("One-sample H0 mean", value=float(round(df[v].mean(), 0)))
        result_card(sc.t_one_sample(df[v], mu))
        g = c3.selectbox("Two-sample t: groups", list(two_level), key="tm_g")
        result_card(sc.t_two_sample(df, v, g, *two_level[g]))
        ga = st.selectbox("ANOVA factor", ["Category", "City", "Payment Method", "Age Group", "Order Status"], key="an")
        result_card(sc.anova(df, v, ga))
        st.dataframe(df.groupby(ga, observed=True)[v].agg(["count", "mean", "std", "median"]).round(3).T,
                     width="stretch")

    with t_var:
        c1, c2 = st.columns(2)
        v = c1.selectbox("Variable", sc.NUM_COLS, index=4, key="tv_v")
        g = c2.selectbox("Factor", ["Category", "City", "Payment Method", "Age Group", "Gender"], key="tv_g")
        result_card(sc.f_test_var(df, v, "Gender", "Male", "Female"))
        result_card(sc.levene(df, v, g))
        result_card(sc.bartlett(df, v, g))
        st.caption("Bartlett assumes normality; Levene (median-centred) is robust when data are skewed.")

    with t_prop:
        flag = st.selectbox("Proportion (event)", ["Is Cancelled", "Is Delivered", "Discount Given"])
        c1, c2 = st.columns(2)
        p0 = c1.number_input("One-sample H0 proportion", 0.0, 1.0, 0.10, 0.01)
        result_card(sc.z_one_prop(int(df[flag].sum()), len(df), p0))
        result_card(sc.z_two_prop(df, flag, "Gender", "Male", "Female"))
        g = c2.selectbox("Chi-square comparison of proportion across", ["Category", "City", "Payment Method"])
        res, ct = sc.chi2_independence(df, g, flag)
        result_card(res)

    with t_corr:
        c1, c2 = st.columns(2)
        x = c1.selectbox("X", sc.NUM_COLS, index=6, key="tc_x")
        y = c2.selectbox("Y", sc.NUM_COLS, index=5, key="tc_y")
        if x != y:
            result_card(sc.corr_test(df, x, y))
            st.caption("t = r·√(n−2)/√(1−r²) with n−2 d.f.; Spearman ρ reported as the rank-based check.")

    with t_norm:
        v = st.selectbox("Variable", sc.NUM_COLS + ["Log Total Amount"], index=4, key="tn_v")
        st.dataframe(sc.normality(df[v], v).round(4), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            f, ax = fig(5, 3.6)
            sns.histplot(df[v].dropna(), kde=True, ax=ax, color=PALETTE[0], edgecolor=SURFACE)
            ax.set_title(f"Histogram of {v}")
            show(f)
        with c2:
            from scipy import stats as _st
            f, ax = fig(5, 3.6)
            (osm, osr), (sl, ic, _) = _st.probplot(df[v].dropna(), dist="norm")
            ax.scatter(osm, osr, s=8, color=PALETTE[0], alpha=.6)
            ax.plot(osm, sl * osm + ic, color=PALETTE[1])
            ax.set_title("Normal Q-Q plot")
            ax.set_xlabel("Theoretical quantiles")
            show(f)

    with t_np:
        v = st.selectbox("Variable", sc.NUM_COLS, index=4, key="np_v")
        g = st.selectbox("Two-group factor (Mann-Whitney)", list(two_level), key="np_g")
        result_card(sc.mann_whitney(df, v, g, *two_level[g]))
        mu = st.number_input("Wilcoxon signed-rank H0 median", value=float(df[v].median()))
        try:
            result_card(sc.wilcoxon_one(df[v], mu, v))
        except ValueError as e:
            st.info(f"Wilcoxon not computable: {e}")
        k = st.selectbox("Kruskal-Wallis factor", ["City", "Category", "Payment Method", "Age Group"], key="kw")
        result_card(sc.kruskal(df, v, k))
        st.markdown("**Friedman test** – blocks = order month, treatments = category (monthly total of the variable)")
        try:
            res, tab = sc.friedman(df, v, "Order Month", "Category")
            result_card(res)
            st.dataframe(tab.round(1), width="stretch")
        except Exception as e:
            st.info(f"Friedman needs ≥ 2 complete months × categories: {e}")

    with t_chi:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Variable A", sc.CAT_COLS, index=4, key="ch_a")
        b = c2.selectbox("Variable B", sc.CAT_COLS, index=5, key="ch_b")
        if a != b:
            res, ct = sc.chi2_independence(df, a, b)
            result_card(res)
            st.dataframe(ct, width="stretch")
        gof = st.selectbox("Goodness of fit (H0: equal shares)", ["Category", "Gender", "Order Weekday", "City"])
        result_card(sc.chi2_gof(df, gof))

# ============================================================================= 6. REGRESSION
with tabs[5]:
    mf = sc.model_frame(df)
    num_x = ["qty", "price", "log_price", "discount", "age", "days", "rating"]
    cat_x = ["category", "gender", "city", "payment", "age_group"]
    st.caption("Variable names: amount = Total Amount, qty = Quantity, price = Unit Price, age = Customer Age, "
               "days = Delivery Days, payment = Payment Method")
    mtype = st.radio("Model", ["Linear (OLS)", "Polynomial", "Regression with categorical variables",
                               "Logistic", "Time-series (daily revenue)", "Report models (M1–M7)"], horizontal=True)
    try:
        if mtype == "Linear (OLS)":
            y = st.selectbox("Dependent", ["amount", "log_amount", "rating", "days"], index=1)
            X = st.multiselect("Independent", [c for c in num_x if c != y], default=["qty", "log_price", "discount", "age"])
            model = sc.ols(mf.dropna(subset=[y] + X), f"{y} ~ {' + '.join(X)}")
        elif mtype == "Polynomial":
            c1, c2, c3 = st.columns(3)
            y = c1.selectbox("Dependent", ["rating", "amount", "log_amount"])
            x = c2.selectbox("Predictor", ["days", "age", "discount", "qty"])
            deg = c3.slider("Degree", 1, 4, 2)
            d = mf.dropna(subset=[y, x])
            terms = " + ".join([x] + [f"I({x}**{k})" for k in range(2, deg + 1)])
            model = sc.ols(d, f"{y} ~ {terms}")
            grid = pd.DataFrame({x: np.linspace(d[x].min(), d[x].max(), 100)})
            f, ax = fig(9, 3.8)
            means = d.groupby(x)[y].mean() if d[x].nunique() < 30 else None
            ax.scatter(d[x] + np.random.uniform(-.15, .15, len(d)), d[y], s=6, alpha=.15, color=PALETTE[0])
            if means is not None:
                ax.plot(means.index, means.values, "o", color=INK, ms=6, label="group mean")
            ax.plot(grid[x], model.predict(grid), color=PALETTE[1], lw=2.5, label=f"degree-{deg} fit")
            ax.legend(frameon=False)
            ax.set_title(f"Polynomial regression: {y} on {x}")
            show(f)
        elif mtype == "Regression with categorical variables":
            y = st.selectbox("Dependent", ["amount", "log_amount", "rating"])
            Xn = st.multiselect("Numeric predictors", num_x, default=["age", "discount"])
            Xc = st.multiselect("Categorical predictors (dummy-coded)", cat_x, default=["category", "gender", "city"])
            rhs = Xn + [f"C({c})" for c in Xc]
            model = sc.ols(mf.dropna(subset=[y] + Xn), f"{y} ~ {' + '.join(rhs)}")
        elif mtype == "Logistic":
            y = st.selectbox("Binary outcome", ["cancelled", "high_rating"])
            Xn = st.multiselect("Numeric predictors", ["discount", "age", "log_amount", "qty", "days"],
                                default=["discount", "age", "log_amount"] if y == "cancelled" else ["days", "discount", "age"])
            Xc = st.multiselect("Categorical predictors", cat_x, default=["payment"] if y == "cancelled" else [])
            rhs = Xn + [f"C({c})" for c in Xc]
            model = sc.logit(mf.dropna(subset=[y] + Xn), f"{y} ~ {' + '.join(rhs)}")
            odds = pd.DataFrame({"Odds ratio": np.exp(model.params), "p-value": model.pvalues}).round(4)
            st.markdown("**Odds ratios**")
            st.dataframe(odds, width="stretch")
        elif mtype == "Time-series (daily revenue)":
            ts = sc.daily_series(df).dropna()
            model = sc.ols(ts, "Revenue ~ t + Revenue_lag1 + C(Weekday)")
            f, ax = fig(10, 3.8)
            ax.plot(ts.index, ts["Revenue"], color=PALETTE[0], lw=1.2, label="actual")
            ax.plot(ts.index, model.fittedvalues, color=PALETTE[1], lw=2, label="fitted")
            ax.plot(ts.index, ts["Revenue"].rolling(7).mean(), color=PALETTE[6], lw=1.5, ls="--", label="7-day avg")
            ax.legend(frameon=False)
            ax.set_title("Daily net revenue: actual vs fitted")
            show(f)
        else:
            models = sc.standard_models(df)
            rows = []
            for name, m in models.items():
                rows.append({"Model": name, "n": int(m.nobs),
                             "R² / pseudo-R²": getattr(m, "rsquared", None) if hasattr(m, "rsquared") else m.prsquared,
                             "Model p-value": m.f_pvalue if hasattr(m, "f_pvalue") else m.llr_pvalue, "AIC": m.aic})
            st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
            pick = st.selectbox("Show coefficients for", list(models))
            model = models[pick]
        if hasattr(model, "rsquared"):
            c = st.columns(4)
            c[0].metric("R²", f"{model.rsquared:.4f}")
            c[1].metric("Adj. R²", f"{model.rsquared_adj:.4f}")
            c[2].metric("F p-value", f"{model.f_pvalue:.2e}")
            c[3].metric("n", f"{int(model.nobs):,}")
        else:
            c = st.columns(3)
            c[0].metric("McFadden pseudo-R²", f"{model.prsquared:.4f}")
            c[1].metric("LLR p-value", f"{model.llr_pvalue:.4f}")
            c[2].metric("n", f"{int(model.nobs):,}")
        st.dataframe(sc.model_table(model).style.format("{:.4f}").apply(
            lambda r: ["background-color:#e3eefb" if r["p-value"] < 0.05 else "" for _ in r], axis=1),
            width="stretch")
        st.caption("Shaded rows are significant at 5%.")
        with st.expander("Full statsmodels summary"):
            st.text(model.summary().as_text())
    except Exception as e:
        st.error(f"Model could not be estimated for this selection: {e}")

# ============================================================================= 7. INSIGHTS
with tabs[6]:
    cat_rev = df.groupby("Category")["Total Amount"].sum().sort_values(ascending=False)
    prod_rev = df.groupby("Product")["Total Amount"].sum().sort_values(ascending=False)
    city = df.groupby("City").agg(orders=("OrderID", "count"), aov=("Total Amount", "mean"))
    dq = df.groupby("Discount")["Quantity"].mean()
    st.markdown(f"""
**Live insights for the current selection ({len(df):,} orders)**

1. **Revenue concentration** – {cat_rev.index[0]} earns **{cat_rev.iloc[0] / rev:.1%}** of revenue; the single product
   *{prod_rev.index[0]}* alone brings **{prod_rev.iloc[0] / rev:.1%}**.
2. **Discounts do not buy volume** – average units per order are {dq.min():.2f}–{dq.max():.2f} at every discount level,
   yet discounts cost **{fmt_money(df['Discount Amount'].sum())}** ({df['Discount Amount'].sum() / df['Gross Amount'].sum():.1%} of gross sales).
3. **Fulfilment leakage** – {df['Is Cancelled'].mean():.1%} of orders are cancelled and
   {(df['Order Status'] == 'Pending').mean():.1%} still pending; together
   **{fmt_money(df.loc[df['Order Status'] != 'Delivered', 'Total Amount'].sum())}** of order value is not realised.
4. **Market footprint** – {city['orders'].idxmax()} is the largest market ({city['orders'].max() / len(df):.1%} of orders);
   {city['aov'].idxmax()} has the highest average order value ({fmt_money(city['aov'].max())}).
5. **Satisfaction** – mean rating {df['Rating'].mean():.2f}/5; {(df['Rating'] >= 4).sum() / df['Rating'].notna().sum():.1%}
   of rated orders score 4★ or 5★, and {df.loc[df['Order Status'] == 'Delivered', 'Rating'].isna().mean():.1%}
   of delivered orders carry no rating.
""")
    st.info("Open the **Inferential** and **Regression** tabs to test whether these differences are statistically "
            "significant for the current filter selection.")

st.markdown("---")
st.caption("FBDA Group Project · Group 001_033_040 · Abhinav Chhabra (075001) · Vidit Garg (075033) · "
           "Sarthak Keshaw Nigam (075040) · FORE School of Management")
