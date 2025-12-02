import re
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Helpers to parse CSV (wide or tidy) ----------

def _coerce_time(col):
    """Coerce a 'time' column to float hours."""
    c = pd.to_numeric(col, errors="coerce")
    if c.notna().all():
        return c.astype(float)

    # If Incucyte-like strings, try to pull out numbers
    s = col.astype(str).str.extract(r"([0-9]*\.?[0-9]+)")[0]
    return pd.to_numeric(s, errors="coerce").astype(float)


def _base_group_name(colname: str) -> str:
    """
    Strip off common replicate suffixes, e.g.
      'A_R1' -> 'A'
      'Drug1_Rep2' -> 'Drug1'
    """
    m = re.match(r"^(.*)_(R\d+|Rep\d+|rep\d+)$", str(colname))
    return m.group(1) if m else str(colname)


def read_incucyte_csv(path_or_buffer) -> pd.DataFrame:
    """
    Read either:
      A) WIDE format: 'time' + one column per group (e.g., Time,A,B,C)
         Optional replicates via suffixes: A_R1, A_R2, B_R1 ...
      B) TIDY format: time, group, replicate, value

    Returns a tidy DataFrame with columns: time, group, replicate, value
    """
    df = pd.read_csv(path_or_buffer)
    lower_map = {c.lower(): c for c in df.columns}
    cols_lower = set(lower_map.keys())

    # --------- WIDE format ---------
    if "time" in cols_lower and not {"group", "replicate", "value"}.issubset(cols_lower):
        time_col = lower_map["time"]
        value_cols_in_order = [c for c in df.columns if c != time_col]
        df = df.rename(columns={time_col: "time"})

        long = df.melt(
            id_vars="time",
            value_vars=value_cols_in_order,
            var_name="col",
            value_name="value",
        )

        # Identify replicates only if suffix looks like a replicate tag
        m = long["col"].astype(str).str.extract(r"^(.*)_(R\d+|Rep\d+|rep\d+)$")
        has_rep = m.notna().all(axis=1)

        long["group"] = np.where(has_rep, m[0], long["col"].astype(str))
        long["replicate"] = np.where(has_rep, m[1], "R1")

        # Preserve group order from header
        group_order = []
        seen = set()
        for c in value_cols_in_order:
            base = _base_group_name(c)
            if base not in seen:
                group_order.append(base)
                seen.add(base)

        long["group"] = long["group"].astype(str)
        long["group"] = pd.Categorical(long["group"], categories=group_order, ordered=True)

        long["time"] = _coerce_time(long["time"])
        long["replicate"] = long["replicate"].astype(str)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")

        long = long.dropna(subset=["time", "value"])
        return long[["time", "group", "replicate", "value"]].reset_index(drop=True)

    # --------- TIDY format ---------
    required = {"time", "group", "value"}
    if required.issubset(cols_lower):
        rename_map = {lower_map[c]: c for c in cols_lower if c in required}
        df = df.rename(columns={v: k for k, v in lower_map.items() if k in required})

        if "replicate" not in cols_lower:
            df["replicate"] = "R1"
        else:
            df = df.rename(columns={lower_map["replicate"]: "replicate"})

        df["time"] = _coerce_time(df["time"])
        df["group"] = df["group"].astype(str)
        df["replicate"] = df["replicate"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = df.dropna(subset=["time", "value"])
        return df[["time", "group", "replicate", "value"]].reset_index(drop=True)

    raise ValueError(
        "Could not detect wide or tidy format. "
        "Need either:\n"
        "  Wide: 'time' + one column per group\n"
        "  Tidy: time, group, (replicate), value"
    )


def aggregate_mean_sd(df: pd.DataFrame, interval_hours: float = None):
    """
    Aggregate replicates to group-level mean ± SD.
    If interval_hours is not None, bin time into that spacing (e.g., 4 h).
    """
    d = df.copy()
    if interval_hours is not None and interval_hours > 0:
        d["time_bin"] = (
            np.floor(d["time"] / interval_hours) * interval_hours
        ).astype(float)
        time_col = "time_bin"
    else:
        time_col = "time"

    group_stats = (
        d.groupby(["group", time_col], as_index=False)["value"]
        .agg(mean="mean", sd="std", n="count")
    )
    group_stats = group_stats.rename(columns={time_col: "time"})
    return group_stats


# ---------- Streamlit UI ----------

st.title("Incucyte Timecourse Plotter")

st.markdown(
    """
Upload a CSV with either:

- **Wide format**: `Time, A, B, C` (replicates as `A_R1`, `A_R2`, etc)  
- **Tidy format**: `time, group, replicate, value`

The app will:

1. Parse groups & replicates  
2. Compute **mean ± SD** per group  
3. Let you customise **axis labels**, **group names**, and **colours**  
4. Plot both **mean curves** and **replicate spaghetti**  
"""
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        tidy = read_incucyte_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("Preview of parsed tidy data")
    st.dataframe(tidy.head())

    # Unique groups
    groups = list(pd.unique(tidy["group"].astype(str)))

    st.sidebar.header("Plot settings")

    # Axis labels
    x_label = st.sidebar.text_input("X axis label", value="Time (h)")
    y_label = st.sidebar.text_input("Y axis label", value="Confluence / Intensity")

    # Optional binning
    interval = st.sidebar.number_input(
        "Time binning (hours, 0 = no binning)",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )
    interval_hours = interval if interval > 0 else None

    # Error type (currently SD only, but easy to extend)
    error_choice = st.sidebar.selectbox("Error bars", ["SD", "None"], index=0)

    # Editable group table for custom names & colours
    st.sidebar.markdown("### Groups")
    default_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    group_df = pd.DataFrame(
        {
            "group": groups,
            "display_name": groups,
            "color": default_colors[: len(groups)],
        }
    )

    edited_group_df = st.sidebar.data_editor(
        group_df,
        num_rows="dynamic",
        use_container_width=True,
        key="group_editor",
    )

    # Compute stats
    stats = aggregate_mean_sd(tidy, interval_hours=interval_hours)

    # Merge display names & colours
    stats = stats.merge(
        edited_group_df, on="group", how="left", validate="many_to_one"
    )
    tidy_merged = tidy.merge(
        edited_group_df, on="group", how="left", validate="many_to_one"
    )

    # ---------- Plot: mean ± SD ----------

    fig, ax = plt.subplots(figsize=(8, 5))

    for g, sub in stats.groupby("group"):
        sub = sub.sort_values("time")
        name = sub["display_name"].iloc[0]
        color = sub["color"].iloc[0] or None

        ax.plot(sub["time"], sub["mean"], label=name, color=color, linewidth=2)

        if error_choice == "SD" and sub["sd"].notna().any():
            ax.fill_between(
                sub["time"],
                sub["mean"] - sub["sd"],
                sub["mean"] + sub["sd"],
                alpha=0.2,
                color=color,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    st.subheader("Mean ± SD per group")
    st.pyplot(fig)

    # ---------- Plot: replicate spaghetti ----------

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for (g, r), sub in tidy_merged.groupby(["group", "replicate"]):
        sub = sub.sort_values("time")
        name = sub["display_name"].iloc[0]
        color = sub["color"].iloc[0] or None
        ax2.plot(sub["time"], sub["value"], color=color, alpha=0.4, label=name)

    # Only one label per group in legend
    handles, labels = ax2.get_legend_handles_labels()
    seen = {}
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            new_handles.append(h)
            new_labels.append(l)
            seen[l] = True

    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.legend(new_handles, new_labels, title="Group", bbox_to_anchor=(1.05, 1),
               loc="upper left")
    ax2.grid(True, alpha=0.3)
    st.subheader("Replicate spaghetti plot")
    st.pyplot(fig2)

    # ---------- Downloadable stats ----------

    st.subheader("Summary (mean ± SD)")
    st.dataframe(stats[["group", "display_name", "time", "mean", "sd", "n"]])

    csv_buffer = io.StringIO()
    stats.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download summary CSV",
        data=csv_buffer.getvalue(),
        file_name="incucyte_summary_mean_sd.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV to begin.")
