"""
ShopEase Raw Orders - shared data-cleaning and statistics engine.
Used by both the Colab notebook (report) and the Streamlit dashboard (app.py),
so every number in the report can be reproduced live in the dashboard.

Group ID : 001_033_040  ->  seed = 1033040
"""
import random
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

GROUP_ID = "001_033_040"
SEED = int(GROUP_ID.replace("_", ""))          # 1033040
SAMPLE_SIZE = 2500

KAGGLE_URL = "https://www.kaggle.com/datasets/mernahabib/shopease-raw-orders-dataset"
KAGGLE_SLUG = "mernahabib/shopease-raw-orders-dataset"
RAW_FILE = "shopease_raw_orders.csv"
GITHUB_MIRROR = ("https://raw.githubusercontent.com/H-Mohanty77/Shopease-Analytics-Dashboard/"
                 "main/shopease_raw_orders.csv")

NUM_COLS = ["Customer Age", "Quantity", "Unit Price", "Discount", "Total Amount",
            "Rating", "Delivery Days", "Order Hour"]
CAT_COLS = ["Gender", "City", "Category", "Product", "Payment Method", "Order Status",
            "Age Group", "Order Weekday", "Order Month"]

AGE_BINS = [17, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
VALID_DISCOUNTS = np.array([0.0, 0.05, 0.10, 0.15, 0.20])


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- loading
def load_raw(path: str = RAW_FILE) -> pd.DataFrame:
    """Local file first (Streamlit / repo), then Kaggle via kagglehub, then GitHub mirror."""
    import os
    if os.path.exists(path):
        return pd.read_csv(path)
    try:
        import kagglehub, glob
        folder = kagglehub.dataset_download(KAGGLE_SLUG)
        f = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)[0]
        return pd.read_csv(f)
    except Exception:
        return pd.read_csv(GITHUB_MIRROR)


# --------------------------------------------------------------------------- cleaning
def _parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    out = pd.to_datetime(s, format="%d-%m-%Y %H:%M", errors="coerce")
    for fmt in ["%d-%m-%Y", "%b %d %Y", "%Y-%m-%d", "%d/%m/%Y"]:
        miss = out.isna()
        if miss.any():
            out[miss] = pd.to_datetime(s[miss], format=fmt, errors="coerce")
    return out  # impossible dates (31/04, 30-Feb) and "not available" stay NaT


def _title(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.title()


def clean(raw: pd.DataFrame):
    """Returns (clean_df, log) where log lists every cleaning action with row counts."""
    df = raw.copy()
    df[["Quantity", "Unit Price", "Total Amount", "Customer Age", "Rating"]] = \
        df[["Quantity", "Unit Price", "Total Amount", "Customer Age", "Rating"]].astype(float)
    has_time = df["Order Date"].astype(str).str.contains(":")
    log = []

    # 1. text standardisation
    for c in ["City", "Category", "Product", "Payment Method", "Order Status"]:
        df[c] = _title(df[c])
    g = df["Gender"].astype("string").str.strip().str.lower()
    df["Gender"] = g.map({"male": "Male", "m": "Male", "female": "Female", "f": "Female"})
    df["Order Status"] = df["Order Status"].replace({"Canceled": "Cancelled"})
    df["CustomerID"] = df["CustomerID"].astype("string").str.strip().str.upper()
    changed = pd.Series(False, index=df.index)
    for c in ["Gender", "City", "Category", "Product", "Payment Method", "Order Status"]:
        changed |= (raw[c].astype("string") != df[c]).fillna(False)
    log.append(("Standardised case/whitespace in Gender, City, Category, Product, Payment Method, "
                "Order Status (e.g. ' CAIRO ' -> 'Cairo', 'f' -> 'Female', 'Canceled' -> 'Cancelled')",
                int(changed.sum())))

    # 2. duplicates
    n0 = len(df)
    df = df.drop_duplicates()
    n1 = len(df)
    df = df.drop_duplicates(subset="OrderID", keep="first")
    log.append(("Removed exact duplicate rows", n0 - n1))
    log.append(("Removed repeated OrderIDs (same order re-entered with formatting differences)", n1 - len(df)))

    # 3. dates
    od = _parse_dates(df["Order Date"])
    log.append(("Order Date in non-standard formats converted (dd-mm-yyyy, 'Mon dd yyyy', ISO)",
                int((~df["Order Date"].astype(str).str.match(r"^\d\d-\d\d-\d{4} \d\d:\d\d$") & od.notna()).sum())))
    log.append(("Order Date invalid ('not available', 31-Apr, 30-Feb) -> set missing", int(od.isna().sum())))
    df["Order Date"] = od
    df["DeliveryDate"] = _parse_dates(df["DeliveryDate"])

    # 4. discount -> fraction
    d = df["Discount"].astype("string").str.strip()
    pct = d.str.endswith("%").fillna(False)
    dnum = pd.to_numeric(d.str.rstrip("%"), errors="coerce").astype(float)
    dnum[pct] = dnum[pct] / 100
    log.append(("Discount written as percentage text ('10%') converted to fraction (0.10)", int(pct.sum())))
    df["Discount"] = dnum

    # 5. quantity: negative sign-entry errors -> absolute value; zero -> back-solve from Total Amount
    neg = df["Quantity"] < 0
    df.loc[neg, "Quantity"] = df.loc[neg, "Quantity"].abs()
    log.append(("Negative Quantity (sign-entry error) converted to absolute value", int(neg.sum())))

    # 6. unit price: product list price is fixed -> repair impossible prices with product's modal price
    list_price = (df[(df["Unit Price"] > 0)].groupby("Product")["Unit Price"]
                  .agg(lambda x: x.mode().iloc[0]))
    exp = df["Product"].map(list_price)
    bad_price = (df["Unit Price"] <= 0) | ((exp.notna()) & (df["Unit Price"] != exp))
    df.loc[bad_price & exp.notna(), "Unit Price"] = exp[bad_price & exp.notna()]
    log.append(("Impossible Unit Price (negative, zero, or 75,000-150,000 outliers) replaced by the "
                "product's list price", int((bad_price & exp.notna()).sum())))

    # 7. missing product -> infer from (Category, Unit Price), which is unique in this catalogue
    key = df.dropna(subset=["Product"]).groupby(["Category", "Unit Price"])["Product"].agg(
        lambda x: x.mode().iloc[0])
    miss_p = df["Product"].isna()
    df.loc[miss_p, "Product"] = [key.get((c, p), np.nan) for c, p in
                                 zip(df.loc[miss_p, "Category"], df.loc[miss_p, "Unit Price"])]
    log.append(("Missing Product inferred from Category + Unit Price", int(miss_p.sum() - df["Product"].isna().sum())))
    still = df["Product"].isna() | (df["Unit Price"] <= 0)
    df = df[~still]
    log.append(("Rows dropped where Product/Unit Price could not be recovered", int(still.sum())))

    # 8. quantity zero -> back-solve; missing discount -> back-solve from Total Amount, else mode (0)
    zero_q = df["Quantity"] == 0
    back_q = (df["Total Amount"] / (df["Unit Price"] * (1 - df["Discount"].fillna(0)))).round()
    df.loc[zero_q, "Quantity"] = back_q[zero_q]
    df = df[df["Quantity"].between(1, 10)]
    log.append(("Zero Quantity back-solved from Total Amount", int(zero_q.sum())))

    miss_d = df["Discount"].isna()
    implied = 1 - df["Total Amount"] / (df["Quantity"] * df["Unit Price"])
    snapped = implied.apply(lambda v: VALID_DISCOUNTS[np.abs(VALID_DISCOUNTS - v).argmin()])
    ok = (implied - snapped).abs() < 0.005
    df.loc[miss_d & ok, "Discount"] = snapped[miss_d & ok]
    df["Discount"] = df["Discount"].fillna(0.0)
    log.append(("Missing Discount back-solved from Total Amount (else mode = 0%)", int(miss_d.sum())))

    # 9. total amount recomputed as the single source of truth
    recomputed = (df["Quantity"] * df["Unit Price"] * (1 - df["Discount"])).round(2)
    log.append(("Total Amount inconsistent with Qty x Price x (1-Discount) -> recomputed",
                int(((recomputed - df["Total Amount"]).abs() > 1).sum())))
    df["Total Amount"] = recomputed

    # 10. age & rating validity
    bad_age = ~df["Customer Age"].between(18, 90) & df["Customer Age"].notna()
    log.append(("Impossible Customer Age (-3, 5, 150) set to missing", int(bad_age.sum())))
    df.loc[bad_age, "Customer Age"] = np.nan
    n_age_na = int(df["Customer Age"].isna().sum())
    df["Customer Age"] = df["Customer Age"].fillna(df["Customer Age"].median())
    log.append(("Missing Customer Age imputed with median", n_age_na))

    bad_r = ~df["Rating"].between(1, 5) & df["Rating"].notna()
    log.append(("Out-of-scale Rating (0, -1, 6) set to missing", int(bad_r.sum())))
    df.loc[bad_r, "Rating"] = np.nan
    notdel = df["Order Status"] != "Delivered"
    df.loc[notdel, "Rating"] = np.nan  # only delivered orders can be rated
    log.append(("Rating left missing (not imputed): structurally absent for Pending/Cancelled orders",
                int(df["Rating"].isna().sum())))

    df["Payment Method"] = df["Payment Method"].fillna("Unknown")
    df["CustomerID"] = df["CustomerID"].fillna("UNKNOWN")
    log.append(("Missing Payment Method / CustomerID labelled 'Unknown'",
                int(raw["Payment Method"].isna().sum() + raw["CustomerID"].isna().sum())))

    # 11. derived features
    df["Delivery Days"] = (df["DeliveryDate"] - df["Order Date"]).dt.total_seconds() / 86400
    df.loc[(df["Delivery Days"] < 0) | notdel, "Delivery Days"] = np.nan
    df["Delivery Days"] = df["Delivery Days"].round(0)
    df["Gross Amount"] = df["Quantity"] * df["Unit Price"]
    df["Discount Amount"] = df["Gross Amount"] - df["Total Amount"]
    df["Order Month"] = df["Order Date"].dt.strftime("%Y-%m")
    df["Order Weekday"] = pd.Categorical(df["Order Date"].dt.day_name(), categories=WEEKDAYS, ordered=True)
    df["Order Hour"] = df["Order Date"].dt.hour
    df.loc[~has_time.loc[df.index], "Order Hour"] = np.nan
    df["Age Group"] = pd.cut(df["Customer Age"], AGE_BINS, labels=AGE_LABELS)
    df["Is Cancelled"] = (df["Order Status"] == "Cancelled").astype(int)
    df["Is Delivered"] = (df["Order Status"] == "Delivered").astype(int)
    df["High Rating"] = np.where(df["Rating"].isna(), np.nan, (df["Rating"] >= 4).astype(float))
    df["Discount Given"] = (df["Discount"] > 0).astype(int)
    df["Log Total Amount"] = np.log(df["Total Amount"])

    df = df.reset_index(drop=True)
    return df, pd.DataFrame(log, columns=["Cleaning action", "Rows affected"])


def draw_sample(clean_df: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = SEED) -> pd.DataFrame:
    set_seed(seed)
    return clean_df.sample(n=n, random_state=seed).reset_index(drop=True)


def get_data(path: str = RAW_FILE, seed: int = SEED, n: int = SAMPLE_SIZE):
    raw = load_raw(path)
    cl, log = clean(raw)
    return raw, cl, log, draw_sample(cl, n, seed)


# --------------------------------------------------------------------------- descriptive
def numeric_summary(df: pd.DataFrame, cols=NUM_COLS) -> pd.DataFrame:
    rows = {}
    for c in cols:
        x = df[c].dropna()
        mode = x.mode()
        rows[c] = {
            "Count": len(x), "Missing": int(df[c].isna().sum()),
            "Min": x.min(), "P25": x.quantile(.25), "Median": x.median(), "P75": x.quantile(.75),
            "P90": x.quantile(.90), "Max": x.max(), "Mean": x.mean(),
            "Mode": mode.iloc[0] if len(mode) else np.nan,
            "Range": x.max() - x.min(), "IQR": x.quantile(.75) - x.quantile(.25),
            "Std Dev": x.std(), "CV %": 100 * x.std() / x.mean() if x.mean() else np.nan,
            "Skewness": stats.skew(x), "Kurtosis (excess)": stats.kurtosis(x),
        }
    return pd.DataFrame(rows).T


def categorical_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vc = df[col].value_counts(dropna=False)
    out = pd.DataFrame({"Frequency": vc, "Relative Frequency": vc / vc.sum()})
    out["Cumulative %"] = 100 * out["Relative Frequency"].cumsum()
    out.index.name = col
    return out


def categorical_overview(df: pd.DataFrame, cols=CAT_COLS) -> pd.DataFrame:
    rows = []
    for c in cols:
        vc = df[c].value_counts()
        rows.append({"Variable": c, "Count": int(vc.sum()), "Levels": len(vc),
                     "Highest": f"{vc.index[0]} ({vc.iloc[0]}, {vc.iloc[0] / vc.sum():.1%})",
                     "Lowest": f"{vc.index[-1]} ({vc.iloc[-1]}, {vc.iloc[-1] / vc.sum():.1%})"})
    return pd.DataFrame(rows).set_index("Variable")


def correlations(df: pd.DataFrame, cols=NUM_COLS, method="pearson") -> pd.DataFrame:
    return df[cols].corr(method=method)


# --------------------------------------------------------------------------- inferential
def _r(name, stat, p, alpha=0.05, **extra):
    d = {"Test": name, "Statistic": stat, "p-value": p,
         "Decision (α=0.05)": "Reject H0" if p < alpha else "Fail to reject H0"}
    d.update(extra)
    return d


def ci_mean(x, conf=0.95):
    x = pd.Series(x).dropna()
    m, se = x.mean(), stats.sem(x)
    lo, hi = stats.t.interval(conf, len(x) - 1, loc=m, scale=se)
    return {"n": len(x), "Mean": m, "SE": se, "Lower": lo, "Upper": hi}


def ci_prop(successes, n, conf=0.95):
    lo, hi = proportion_confint(successes, n, alpha=1 - conf, method="wilson")
    return {"n": n, "Proportion": successes / n, "Lower": lo, "Upper": hi}


def t_one_sample(x, mu):
    x = pd.Series(x).dropna()
    t, p = stats.ttest_1samp(x, mu)
    return _r(f"One-sample t (H0: mean = {mu})", t, p, mean=x.mean(), n=len(x))


def t_two_sample(df, y, g, a, b, equal_var=False):
    xa, xb = df.loc[df[g] == a, y].dropna(), df.loc[df[g] == b, y].dropna()
    t, p = stats.ttest_ind(xa, xb, equal_var=equal_var)
    return _r(f"Welch t: {y} by {g} ({a} vs {b})", t, p, **{f"mean {a}": xa.mean(), f"mean {b}": xb.mean()})


def anova(df, y, g):
    groups = [v[y].dropna() for _, v in df.groupby(g, observed=True) if v[y].notna().sum() > 1]
    f, p = stats.f_oneway(*groups)
    return _r(f"One-way ANOVA: {y} by {g}", f, p, groups=len(groups))


def f_test_var(df, y, g, a, b):
    xa, xb = df.loc[df[g] == a, y].dropna(), df.loc[df[g] == b, y].dropna()
    f = xa.var(ddof=1) / xb.var(ddof=1)
    d1, d2 = len(xa) - 1, len(xb) - 1
    p = 2 * min(stats.f.cdf(f, d1, d2), 1 - stats.f.cdf(f, d1, d2))
    return _r(f"F-test of variances: {y} ({a} vs {b})", f, p, **{f"var {a}": xa.var(), f"var {b}": xb.var()})


def levene(df, y, g):
    groups = [v[y].dropna() for _, v in df.groupby(g, observed=True) if v[y].notna().sum() > 1]
    s, p = stats.levene(*groups, center="median")
    return _r(f"Levene (median): {y} by {g}", s, p)


def bartlett(df, y, g):
    groups = [v[y].dropna() for _, v in df.groupby(g, observed=True) if v[y].notna().sum() > 1]
    s, p = stats.bartlett(*groups)
    return _r(f"Bartlett: {y} by {g}", s, p)


def z_one_prop(successes, n, p0):
    z, p = proportions_ztest(successes, n, value=p0, prop_var=p0)
    return _r(f"One-proportion z (H0: p = {p0})", z, p, observed=successes / n)


def z_two_prop(df, flag, g, a, b):
    s = [int(df.loc[df[g] == a, flag].sum()), int(df.loc[df[g] == b, flag].sum())]
    n = [int((df[g] == a).sum()), int((df[g] == b).sum())]
    z, p = proportions_ztest(s, n)
    return _r(f"Two-proportion z: {flag} rate ({a} vs {b})", z, p,
              **{f"p {a}": s[0] / n[0], f"p {b}": s[1] / n[1]})


def chi2_independence(df, a, b):
    ct = pd.crosstab(df[a], df[b])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.values.sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    return _r(f"Chi-square independence: {a} × {b}", chi2, p, dof=dof, **{"Cramér's V": v}), ct


def chi2_gof(df, col, expected_props=None):
    obs = df[col].value_counts().sort_index()
    if expected_props is None:
        exp = np.full(len(obs), obs.sum() / len(obs))
        label = "uniform"
    else:
        exp = np.array([expected_props[k] for k in obs.index]) * obs.sum()
        label = "specified"
    chi2, p = stats.chisquare(obs, exp)
    return _r(f"Chi-square goodness of fit: {col} ({label})", chi2, p, dof=len(obs) - 1)


def corr_test(df, x, y):
    d = df[[x, y]].dropna()
    r, p = stats.pearsonr(d[x], d[y])
    n = len(d)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    rho, ps = stats.spearmanr(d[x], d[y])
    return _r(f"t-test of correlation: {x} ~ {y}", t, p, r=r, n=n, **{"Spearman ρ": rho, "Spearman p": ps})


def normality(x, name=""):
    x = pd.Series(x).dropna()
    xs = x.sample(min(len(x), 5000), random_state=SEED)
    out = []
    w, p = stats.shapiro(xs)
    out.append(_r(f"Shapiro-Wilk {name}", w, p))
    k, p = stats.kstest((x - x.mean()) / x.std(ddof=1), "norm")
    out.append(_r(f"Kolmogorov-Smirnov {name}", k, p))
    ad = stats.anderson(x, dist="norm")
    cv5 = ad.critical_values[list(ad.significance_level).index(5.0)]
    out.append({"Test": f"Anderson-Darling {name}", "Statistic": ad.statistic, "p-value": np.nan,
                "Decision (α=0.05)": "Reject H0" if ad.statistic > cv5 else "Fail to reject H0",
                "critical value 5%": cv5})
    jb, p = stats.jarque_bera(x)
    out.append(_r(f"Jarque-Bera {name}", jb, p))
    return pd.DataFrame(out)


def mann_whitney(df, y, g, a, b):
    xa, xb = df.loc[df[g] == a, y].dropna(), df.loc[df[g] == b, y].dropna()
    u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    return _r(f"Mann-Whitney U: {y} ({a} vs {b})", u, p, **{f"median {a}": xa.median(), f"median {b}": xb.median()})


def wilcoxon_one(x, mu, name=""):
    x = pd.Series(x).dropna()
    w, p = stats.wilcoxon(x - mu)
    return _r(f"Wilcoxon signed-rank {name} (H0: median = {mu})", w, p, median=x.median())


def kruskal(df, y, g):
    groups = [v[y].dropna() for _, v in df.groupby(g, observed=True) if v[y].notna().sum() > 1]
    h, p = stats.kruskal(*groups)
    return _r(f"Kruskal-Wallis: {y} by {g}", h, p)


def friedman(df, value, block, treatment, agg="sum"):
    tab = df.pivot_table(index=block, columns=treatment, values=value, aggfunc=agg, observed=True).dropna()
    s, p = stats.friedmanchisquare(*[tab[c] for c in tab.columns])
    return _r(f"Friedman: {value} across {treatment} (blocks = {block})", s, p, blocks=len(tab)), tab


# --------------------------------------------------------------------------- causal / regression
RENAME = {"Total Amount": "amount", "Log Total Amount": "log_amount", "Quantity": "qty",
          "Unit Price": "price", "Discount": "discount", "Customer Age": "age", "Rating": "rating",
          "Delivery Days": "days", "Category": "category", "Gender": "gender", "City": "city",
          "Payment Method": "payment", "Is Cancelled": "cancelled", "High Rating": "high_rating",
          "Order Status": "status", "Age Group": "age_group"}


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Formula-friendly column names for statsmodels."""
    d = df.rename(columns=RENAME).copy()
    d["log_price"] = np.log(d["price"])
    d["days_sq"] = d["days"] ** 2
    d["age_sq"] = d["age"] ** 2
    return d


def ols(df, formula):
    return smf.ols(formula, data=df).fit()


def logit(df, formula):
    return smf.logit(formula, data=df).fit(disp=False)


def model_table(m):
    t = pd.DataFrame({"coef": m.params, "std err": m.bse, "t/z": m.tvalues, "p-value": m.pvalues})
    ci = m.conf_int()
    t["[0.025"] = ci[0]
    t["0.975]"] = ci[1]
    return t


def daily_series(df):
    s = (df.dropna(subset=["Order Date"]).set_index("Order Date")
         .resample("D")["Total Amount"].agg(["sum", "count"]).rename(columns={"sum": "Revenue", "count": "Orders"}))
    s["t"] = np.arange(len(s))
    s["Weekday"] = s.index.day_name()
    s["Revenue_lag1"] = s["Revenue"].shift(1)
    return s


def standard_models(df):
    """The fixed set of models reported in the notebook (formula names: see RENAME)."""
    d = model_frame(df)
    dr = d.dropna(subset=["rating", "days"])
    ts = daily_series(df).dropna()
    m = {}
    m["M1 OLS (log-log): log_amount ~ qty + log_price + discount + age"] = ols(
        d, "log_amount ~ qty + log_price + discount + age")
    m["M2 OLS with categorical: amount ~ age + discount + C(category) + C(gender) + C(city)"] = ols(
        d, "amount ~ age + discount + C(category) + C(gender) + C(city)")
    m["M3 Polynomial: rating ~ days + days²"] = ols(dr, "rating ~ days + days_sq")
    m["M4 Categorical: rating ~ days + discount + C(category) + C(payment)"] = ols(
        dr, "rating ~ days + discount + C(category) + C(payment)")
    m["M5 Time-series: daily Revenue ~ trend t + Revenue(t-1) + C(Weekday)"] = ols(
        ts, "Revenue ~ t + Revenue_lag1 + C(Weekday)")
    m["M6 Logistic: P(cancelled) ~ discount + age + log_amount + C(payment) + C(category)"] = logit(
        d, "cancelled ~ discount + age + log_amount + C(payment) + C(category)")
    m["M7 Logistic: P(rating ≥ 4) ~ days + discount + age"] = logit(
        dr, "high_rating ~ days + discount + age")
    return m


# --------------------------------------------------------------------------- plotting style (notebook)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def set_style():
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette=PALETTE, rc={
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#e6e5e0",
        "grid.color": "#e6e5e0", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
