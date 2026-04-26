import json
import re
import copy

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import geopandas as gpd
import fiona
import folium
from streamlit_folium import st_folium


# ============================================================
# CONFIG
# ============================================================

CSV_FILE = "data/WB_election_2021.csv"
KML_FILE = "data/wb_acs_map.kml"
BINS = 30

CSV_AC_NAME_COL = "AC NAME"
CSV_PARTY_COL = "PARTY"
CSV_VOTE_COL = "TOTAL"
CSV_ELECTORS_COL = "TOTAL ELECTORS"

KML_AC_NO_COL = "ac_no"
KML_AC_NAME_COL = "ac_name"
KML_DISTRICT_COL = "dist_name"

MAP_CENTER = [23.5, 87.5]
MAP_ZOOM = 7

PARTY_COLORS = {
    "BJP": "#ff9800",      # orange
    "AITC": "#2ca25f",     # green
    "TMC": "#2ca25f",
    "CPIM": "#d73027",     # red
    "INC": "#756bb1",
    "Other": "#bdbdbd",
    "Others": "#bdbdbd",
}

MAP_OTHER_COLOR = "#d73027"
BACKGROUND_GREY = "#d9d9d9"


# ============================================================
# Utilities
# ============================================================

def clean_ac_name(x):
    x = str(x).upper().strip()

    # Remove reservation suffix from KML names
    x = re.sub(r"\(SC\)", "", x)
    x = re.sub(r"\(ST\)", "", x)

    x = x.replace(".", "")
    x = x.replace("-", " ")
    x = x.replace("_", " ")

    x = re.sub(r"\s+", " ", x)
    return x.strip()

def load_csv_data(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    return df


def read_kml_to_geojson(kml_file):
    fiona.drvsupport.supported_drivers["KML"] = "rw"
    fiona.drvsupport.supported_drivers["LIBKML"] = "rw"

    gdf = gpd.read_file(kml_file, driver="KML")

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    gdf["AC_NAME_CLEAN"] = gdf[KML_AC_NAME_COL].apply(clean_ac_name)

    geojson = json.loads(gdf.to_json())

    for feature in geojson["features"]:
        props = feature["properties"]
        clean_name = clean_ac_name(props[KML_AC_NAME_COL])
        props["AC_NAME_CLEAN"] = clean_name
        feature["id"] = clean_name

    return geojson

# ============================================================
# Election dataframe
# ============================================================

def make_ac_party_vote_dataframe(df):
    df = df.copy()

    df[CSV_VOTE_COL] = pd.to_numeric(df[CSV_VOTE_COL], errors="coerce").fillna(0)
    df[CSV_ELECTORS_COL] = pd.to_numeric(df[CSV_ELECTORS_COL], errors="coerce").fillna(0)

    df["PARTY_NORM"] = df[CSV_PARTY_COL].replace({
        "CPI(M)": "CPIM",
        "AITC": "AITC",
        "BJP": "BJP",
        "INC": "INC",
    })

    main_parties = ["BJP", "AITC", "CPIM", "INC"]

    party_votes = (
        df[df["PARTY_NORM"].isin(main_parties)]
        .pivot_table(
            index=CSV_AC_NAME_COL,
            columns="PARTY_NORM",
            values=CSV_VOTE_COL,
            aggfunc="sum",
            fill_value=0,
        )
    )

    for party in main_parties:
        if party not in party_votes.columns:
            party_votes[party] = 0

    party_votes = party_votes[main_parties]

    total_votes_by_ac = df.groupby(CSV_AC_NAME_COL)[CSV_VOTE_COL].sum()
    party_votes["Other count total"] = total_votes_by_ac - party_votes.sum(axis=1)

    electors = df.groupby(CSV_AC_NAME_COL)[CSV_ELECTORS_COL].first()
    party_votes["Total Electors"] = electors

    ranked = (
        df.sort_values([CSV_AC_NAME_COL, CSV_VOTE_COL], ascending=[True, False])
        .groupby(CSV_AC_NAME_COL)
    )

    winner_rows = (
        ranked.nth(0)
        .reset_index()
        [[CSV_AC_NAME_COL, "PARTY_NORM", CSV_VOTE_COL]]
        .rename(columns={
            "PARTY_NORM": "Winner",
            CSV_VOTE_COL: "Winner Vote",
        })
    )

    second_rows = (
        ranked.nth(1)
        .reset_index()
        [[CSV_AC_NAME_COL, "PARTY_NORM", CSV_VOTE_COL]]
        .rename(columns={
            "PARTY_NORM": "Second Party",
            CSV_VOTE_COL: "Second Best Vote",
        })
    )

    third_rows = (
        ranked.nth(2)
        .reset_index()
        [[CSV_AC_NAME_COL, "PARTY_NORM", CSV_VOTE_COL]]
        .rename(columns={
            "PARTY_NORM": "Third Party",
            CSV_VOTE_COL: "Third Highest Vote",
        })
    )

    ac_df = party_votes.reset_index()

    ac_df = ac_df.rename(columns={
        "BJP": "BJP vote count",
        "AITC": "AITC vote count",
        "CPIM": "CPIM vote count",
        "INC": "INC vote count",
    })

    ac_df = ac_df.merge(winner_rows, on=CSV_AC_NAME_COL, how="left")
    ac_df = ac_df.merge(second_rows, on=CSV_AC_NAME_COL, how="left")
    ac_df = ac_df.merge(third_rows, on=CSV_AC_NAME_COL, how="left")

    ac_df = ac_df.fillna(0)
    ac_df["AC_NAME_CLEAN"] = ac_df[CSV_AC_NAME_COL].apply(clean_ac_name)

    return ac_df


def add_winner_and_differences(ac_df):
    ac_df = ac_df.copy()

    ac_df["Winner Second Difference"] = (
        ac_df["Winner Vote"] - ac_df["Second Best Vote"]
    )

    ac_df["BJP AITC Difference"] = (
        ac_df["BJP vote count"] - ac_df["AITC vote count"]
    ).abs()

    ac_df["BJP minus AITC signed"] = (
        ac_df["BJP vote count"] - ac_df["AITC vote count"]
    )

    ac_df["Leading Party BJP/AITC"] = np.where(
        ac_df["BJP minus AITC signed"] > 0,
        "BJP",
        "AITC",
    )

    return ac_df

def attach_votes_to_geojson(geojson, ac_df):
    ac_map = ac_df.set_index("AC_NAME_CLEAN").to_dict(orient="index")
    out = copy.deepcopy(geojson)

    for feature in out["features"]:
        props = feature["properties"]
        clean_name = props["AC_NAME_CLEAN"]

        row = ac_map.get(clean_name, None)

        if row is None:
            props["Matched"] = False
            props["BJP vote count"] = 0
            props["AITC vote count"] = 0
            props["CPIM vote count"] = 0
            props["INC vote count"] = 0
            props["Other count total"] = 0
            props["Total Electors"] = 0
            props["Winner"] = "Unknown"
            props["BJP AITC Difference"] = 0
            props["Leading Party BJP/AITC"] = "Unknown"
            props["Winner Second Difference"] = 0
        else:
            props["Matched"] = True
            for key, val in row.items():
                if isinstance(val, (np.integer, np.floating)):
                    val = float(val)
                props[key] = val

    return out


def filter_geojson_by_difference(geojson, min_diff, max_diff):
    out = copy.deepcopy(geojson)
    out["features"] = [
        f for f in out["features"]
        if min_diff <= float(f["properties"].get("BJP AITC Difference", 0)) <= max_diff
    ]
    return out


# ============================================================
# Plotly charts
# ============================================================

def plot_age_distribution(df, bins=25):
    age = pd.to_numeric(df["AGE"], errors="coerce").dropna()
    q1, median, q3 = np.percentile(age, [25, 50, 75])

    stats_text = (
        f"Entries: {len(age)}<br>"
        f"Mean: {age.mean():.2f}<br>"
        f"Median: {median:.2f}<br>"
        f"Std: {age.std():.2f}<br>"
        f"Q1: {q1:.2f}<br>"
        f"Q3: {q3:.2f}<br>"
        f"Min: {age.min():.0f}<br>"
        f"Max: {age.max():.0f}"
    )

    fig = go.Figure()
    fig.add_histogram(x=age, nbinsx=bins)

    for val, label in [(q1, "Q1"), (median, "Median"), (q3, "Q3")]:
        fig.add_vline(x=val, line_dash="dash", annotation_text=label)

    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        align="right",
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
    )

    fig.update_layout(
        title="Candidate Age Distribution",
        xaxis_title="Age",
        yaxis_title="Number of Candidates",
        height=420,
    )

    return fig


def pie_from_column(df, column, title):
    counts = df[column].fillna("Unknown").value_counts()

    fig = go.Figure()
    fig.add_pie(
        labels=counts.index,
        values=counts.values,
        hole=0.3,
        textinfo="label+percent",
    )
    fig.update_layout(title=title, height=420)

    return fig


def make_total_seat_pie(ac_df):
    seat_counts = ac_df["Winner"].replace({"AITC": "TMC"}).value_counts()

    colors = [
        PARTY_COLORS.get(label, PARTY_COLORS["Other"])
        for label in seat_counts.index
    ]

    fig = go.Figure()
    fig.add_pie(
        labels=seat_counts.index,
        values=seat_counts.values,
        hole=0.3,
        textinfo="label+value+percent",
        marker=dict(colors=colors, line=dict(color="black", width=1)),
    )

    fig.update_layout(title="Total Seats Won", height=430)
    return fig


def make_vote_share_pie(ac_df):
    vote_totals = {
        "BJP": ac_df["BJP vote count"].sum(),
        "TMC": ac_df["AITC vote count"].sum(),
        "CPIM": ac_df["CPIM vote count"].sum(),
        "INC": ac_df["INC vote count"].sum(),
        "Others": ac_df["Other count total"].sum(),
    }

    colors = [
        PARTY_COLORS.get(label, PARTY_COLORS["Other"])
        for label in vote_totals.keys()
    ]

    fig = go.Figure()
    fig.add_pie(
        labels=list(vote_totals.keys()),
        values=list(vote_totals.values()),
        hole=0.3,
        textinfo="label+percent",
        marker=dict(colors=colors, line=dict(color="black", width=1)),
    )

    fig.update_layout(title="Vote Share", height=430)
    return fig


def make_bjp_aitc_histogram(ac_df, bins=30, min_diff=None, max_diff=None):
    data = ac_df["BJP AITC Difference"].dropna()
    bin_edges = np.histogram_bin_edges(data, bins=bins)

    bjp_data = ac_df.loc[
        ac_df["Leading Party BJP/AITC"] == "BJP",
        "BJP AITC Difference"
    ]

    tmc_data = ac_df.loc[
        ac_df["Leading Party BJP/AITC"] == "AITC",
        "BJP AITC Difference"
    ]

    bjp_counts, _ = np.histogram(bjp_data, bins=bin_edges)
    tmc_counts, _ = np.histogram(tmc_data, bins=bin_edges)

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = bin_edges[1:] - bin_edges[:-1]

    q1, median, q3 = np.percentile(data, [25, 50, 75])

    fig = go.Figure()

    fig.add_bar(
        x=centers,
        y=bjp_counts,
        width=widths,
        name="BJP ahead",
        marker_color=PARTY_COLORS["BJP"],
        marker_line_color="black",
        marker_line_width=0.4,
        customdata=np.column_stack([bin_edges[:-1], bin_edges[1:]]),
        hovertemplate=(
            "Difference: %{customdata[0]:.0f}–%{customdata[1]:.0f}<br>"
            "BJP-ahead seats: %{y}<extra></extra>"
        ),
    )

    fig.add_bar(
        x=centers,
        y=tmc_counts,
        width=widths,
        name="TMC ahead",
        marker_color=PARTY_COLORS["AITC"],
        marker_line_color="black",
        marker_line_width=0.4,
        customdata=np.column_stack([bin_edges[:-1], bin_edges[1:]]),
        hovertemplate=(
            "Difference: %{customdata[0]:.0f}–%{customdata[1]:.0f}<br>"
            "TMC-ahead seats: %{y}<extra></extra>"
        ),
    )

    for val, label in [(q1, "Q1"), (median, "Median"), (q3, "Q3")]:
        fig.add_vline(x=val, line_dash="dash", annotation_text=label)

    if min_diff is not None and max_diff is not None:
        fig.add_vrect(
            x0=min_diff,
            x1=max_diff,
            opacity=0.22,
            line_width=0,
            fillcolor="#ffcc00",
        )

    stats_text = (
        f"Entries: {len(data)}<br>"
        f"Mean: {data.mean():.1f}<br>"
        f"Median: {median:.1f}<br>"
        f"Q1: {q1:.1f}<br>"
        f"Q3: {q3:.1f}<br>"
        f"Min: {data.min():.0f}<br>"
        f"Max: {data.max():.0f}"
    )

    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        align="right",
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
    )

    fig.update_layout(
        title="BJP vs TMC Vote Difference",
        xaxis_title="|BJP vote - TMC vote|",
        yaxis_title="Number of Assembly Constituencies",
        barmode="stack",
        height=650,
        bargap=0.05,
        legend_title="Leader in BJP–TMC contest",
    )

    return fig


def filter_by_difference_quartile(ac_df, quartile="All"):
    df = ac_df.copy()
    data = df["BJP AITC Difference"].dropna()

    q1, q2, q3 = np.percentile(data, [25, 50, 75])

    if quartile == "Q1: closest seats":
        return df[df["BJP AITC Difference"] <= q1].copy()

    if quartile == "Q2":
        return df[
            (df["BJP AITC Difference"] > q1)
            & (df["BJP AITC Difference"] <= q2)
        ].copy()

    if quartile == "Q3":
        return df[
            (df["BJP AITC Difference"] > q2)
            & (df["BJP AITC Difference"] <= q3)
        ].copy()

    if quartile == "Q4: largest margins":
        return df[df["BJP AITC Difference"] > q3].copy()

    return df


def make_third_vote_vs_difference_plot(
    ac_df,
    quartile="All",
    n_seats=30,
    sort_by="BJP AITC Difference",
    log_y=False,
):
    plot_df = filter_by_difference_quartile(ac_df, quartile).copy()

    if sort_by == "Third Highest Vote":
        plot_df = plot_df.sort_values("Third Highest Vote", ascending=False)
    else:
        plot_df = plot_df.sort_values("BJP AITC Difference", ascending=True)

    plot_df = plot_df.head(n_seats).copy()

    bar_colors = plot_df["Leading Party BJP/AITC"].map({
        "BJP": PARTY_COLORS["BJP"],
        "AITC": PARTY_COLORS["AITC"],
    }).fillna("#bdbdbd")

    third_colors = plot_df["Third Party"].map({
        "CPIM": PARTY_COLORS["CPIM"],
        "INC": PARTY_COLORS["INC"],
    }).fillna("#444444")

    fig = go.Figure()

    fig.add_bar(
        x=plot_df[CSV_AC_NAME_COL],
        y=plot_df["BJP AITC Difference"],
        name="BJP–TMC difference",
        marker_color=bar_colors,
        customdata=np.column_stack([
            plot_df["Winner"],
            plot_df["BJP vote count"],
            plot_df["AITC vote count"],
            plot_df["Third Party"],
            plot_df["Third Highest Vote"],
        ]),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Winner: %{customdata[0]}<br>"
            "BJP: %{customdata[1]:,.0f}<br>"
            "TMC: %{customdata[2]:,.0f}<br>"
            "BJP–TMC diff: %{y:,.0f}<br>"
            "Third party: %{customdata[3]}<br>"
            "Third-party vote: %{customdata[4]:,.0f}<extra></extra>"
        ),
    )

    fig.add_scatter(
        x=plot_df[CSV_AC_NAME_COL],
        y=plot_df["Third Highest Vote"],
        mode="markers+lines",
        name="Third-party vote count",
        marker=dict(
            size=10,
            color=third_colors,
            line=dict(color="black", width=0.8),
        ),
        line=dict(color="#444444", dash="dot"),
        customdata=np.column_stack([
            plot_df["Third Party"],
            plot_df["Third Highest Vote"],
        ]),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Third party: %{customdata[0]}<br>"
            "Third-party vote: %{customdata[1]:,.0f}<extra></extra>"
        ),
    )

    fig.update_layout(
        title=f"BJP–TMC Difference and Third-party Vote Count ({quartile}, {len(plot_df)} seats)",
        xaxis_title="Assembly Constituency",
        yaxis_title="Votes",
        height=650,
        xaxis_tickangle=-60,
        legend_title="Quantity",
    )

    if log_y:
        fig.update_yaxes(type="log", title="Votes [log scale]")

    return fig

#heatmap
def make_diff_vs_third_vote_heatmap(
    ac_df,
    x_bin_width=3000,
    y_bin_width=3000,
    quartile="All",
):
    plot_df = filter_by_difference_quartile(ac_df, quartile).copy()

    x = plot_df["BJP AITC Difference"].to_numpy()
    y = plot_df["Third Highest Vote"].to_numpy()

    x_max = np.ceil(x.max() / x_bin_width) * x_bin_width
    y_max = np.ceil(y.max() / y_bin_width) * y_bin_width
    max_diag = max(x_max, y_max)

    x_edges = np.arange(0, x_max + x_bin_width, x_bin_width)
    y_edges = np.arange(0, y_max + y_bin_width, y_bin_width)

    total_hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    bjp_df = plot_df[plot_df["Winner"] == "BJP"]
    tmc_df = plot_df[plot_df["Winner"] == "AITC"]

    bjp_hist, _, _ = np.histogram2d(
        bjp_df["BJP AITC Difference"],
        bjp_df["Third Highest Vote"],
        bins=[x_edges, y_edges],
    )

    tmc_hist, _, _ = np.histogram2d(
        tmc_df["BJP AITC Difference"],
        tmc_df["Third Highest Vote"],
        bins=[x_edges, y_edges],
    )

    other_hist = total_hist - bjp_hist - tmc_hist

    q1, q2, q3 = np.percentile(
        ac_df["BJP AITC Difference"].dropna(),
        [25, 50, 75],
    )

    above_diag = plot_df[
        plot_df["Third Highest Vote"] > plot_df["BJP AITC Difference"]
    ]
    below_diag = plot_df[
        plot_df["Third Highest Vote"] <= plot_df["BJP AITC Difference"]
    ]

    def seat_summary(df_part):
        total = len(df_part)
        bjp = int((df_part["Winner"] == "BJP").sum())
        tmc = int((df_part["Winner"] == "AITC").sum())
        other = total - bjp - tmc
        return total, bjp, tmc, other

    total_all, bjp_all, tmc_all, other_all = seat_summary(plot_df)
    total_above, bjp_above, tmc_above, other_above = seat_summary(above_diag)
    total_below, bjp_below, tmc_below, other_below = seat_summary(below_diag)

    stats_text = (
        f"<b>{quartile}</b><br>"
        f"Total seats: {total_all}<br>"
        f"<span style='color:#ff9800;'>BJP:</span> {bjp_all}<br>"
        f"<span style='color:#2ca25f;'>TMC:</span> {tmc_all}<br>"
        f"<span style='color:#d73027;'>Others:</span> {other_all}<br><br>"
        f"<b>Above diagonal</b><br>"
        f"Third vote > BJP–TMC diff<br>"
        f"Seats: {total_above}<br>"
        f"<span style='color:#ff9800;'>BJP:</span> {bjp_above}<br>"
        f"<span style='color:#2ca25f;'>TMC:</span> {tmc_above}<br>"
        f"<span style='color:#d73027;'>Others:</span> {other_above}<br><br>"
        f"<b>Below diagonal</b><br>"
        f"Third vote ≤ BJP–TMC diff<br>"
        f"Seats: {total_below}<br>"
        f"<span style='color:#ff9800;'>BJP:</span> {bjp_below}<br>"
        f"<span style='color:#2ca25f;'>TMC:</span> {tmc_below}<br>"
        f"<span style='color:#d73027;'>Others:</span> {other_below}"
    )

    fig = go.Figure()

    # Draw each heatmap cell manually as split rectangles
    max_count = np.nanmax(total_hist) if np.nanmax(total_hist) > 0 else 1

    for ix in range(len(x_edges) - 1):
        for iy in range(len(y_edges) - 1):
            total = total_hist[ix, iy]
            if total == 0:
                continue

            bjp = bjp_hist[ix, iy]
            tmc = tmc_hist[ix, iy]
            other = other_hist[ix, iy]

            bjp_tmc_total = bjp + tmc
            if bjp_tmc_total > 0:
                tmc_frac = tmc / bjp_tmc_total
                bjp_frac = bjp / bjp_tmc_total
            else:
                tmc_frac = 0
                bjp_frac = 0

            x0 = x_edges[ix]
            x1 = x_edges[ix + 1]
            y0 = y_edges[iy]
            y1 = y_edges[iy + 1]

            split_x = x0 + (x1 - x0) * tmc_frac

            opacity = 0.25 + 0.75 * (total / max_count)

            # TMC fraction: left part
            if tmc_frac > 0:
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=split_x,
                    y0=y0,
                    y1=y1,
                    line=dict(color="rgba(0,0,0,0.35)", width=0.5),
                    fillcolor=f"rgba(44,162,95,{opacity})",
                )

            # BJP fraction: right part
            if bjp_frac > 0:
                fig.add_shape(
                    type="rect",
                    x0=split_x,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    line=dict(color="rgba(0,0,0,0.35)", width=0.5),
                    fillcolor=f"rgba(255,152,0,{opacity})",
                )

            # Optional tiny red stripe for other winners
            if other > 0:
                stripe_h = (y1 - y0) * min(0.18, other / total)
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y0 + stripe_h,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    fillcolor=f"rgba(215,48,39,{opacity})",
                )

            # Invisible marker only for hover
            fig.add_trace(
                go.Scatter(
                    x=[0.5 * (x0 + x1)],
                    y=[0.5 * (y0 + y1)],
                    mode="markers",
                    marker=dict(size=18, color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hovertemplate=(
                        f"BJP–TMC diff: {x0:,.0f}–{x1:,.0f}<br>"
                        f"Third-party vote: {y0:,.0f}–{y1:,.0f}<br>"
                        f"Total seats: {total:.0f}<br>"
                        f"BJP-won: {bjp:.0f}<br>"
                        f"TMC-won: {tmc:.0f}<br>"
                        f"Other-won: {other:.0f}<extra></extra>"
                    ),
                )
            )

    # Legend proxies
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color=PARTY_COLORS["AITC"]),
        name="TMC fraction in cell",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color=PARTY_COLORS["BJP"]),
        name="BJP fraction in cell",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color=MAP_OTHER_COLOR),
        name="Other winner stripe",
    ))

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_diag,
        y1=max_diag,
        line=dict(color="black", width=3, dash="dash"),
    )

    for val, label in [(q1, "Q1"), (q2, "Median"), (q3, "Q3")]:
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color="red",
            annotation_text=label,
            annotation_position="top",
        )

    fig.add_annotation(
        x=max_diag * 0.68,
        y=max_diag * 0.78,
        text="Third-party vote = BJP–TMC difference",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="black",
        borderwidth=1,
    )

    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        align="left",
        bordercolor="black",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.92)",
    )

    fig.update_layout(
        title=(
            "Split-cell heatmap: BJP–TMC Difference vs Third-party Vote<br>"
            "<sup>Each occupied cell is split by BJP/TMC seat composition; opacity shows total seat density</sup>"
        ),
        xaxis_title=f"|BJP vote - TMC vote|, bin width = {x_bin_width:,}",
        yaxis_title=f"Third-party vote count, bin width = {y_bin_width:,}",
        height=730,
        plot_bgcolor="white",
        legend_title="Cell composition",
    )

    fig.update_xaxes(range=[0, x_max])
    fig.update_yaxes(range=[0, y_max])

    return fig

#scatter
def make_diff_vs_third_vote_scatter(ac_df, quartile="All", log_y=False):
    plot_df = filter_by_difference_quartile(ac_df, quartile).copy()

    q1, q2, q3 = np.percentile(
        ac_df["BJP AITC Difference"].dropna(),
        [25, 50, 75],
    )

    x_max = plot_df["BJP AITC Difference"].max()
    y_max = plot_df["Third Highest Vote"].max()
    max_diag = max(x_max, y_max)

    colors = plot_df["Leading Party BJP/AITC"].map({
        "BJP": PARTY_COLORS["BJP"],
        "AITC": PARTY_COLORS["AITC"],
    }).fillna("#bdbdbd")

    above_diag = plot_df[
        plot_df["Third Highest Vote"] > plot_df["BJP AITC Difference"]
    ]

    below_diag = plot_df[
        plot_df["Third Highest Vote"] <= plot_df["BJP AITC Difference"]
    ]

    def count_summary(df_part):
        bjp = int((df_part["Leading Party BJP/AITC"] == "BJP").sum())
        tmc = int((df_part["Leading Party BJP/AITC"] == "AITC").sum())
        return bjp, tmc

    bjp_above, tmc_above = count_summary(above_diag)
    bjp_below, tmc_below = count_summary(below_diag)

    stats_text = (
        f"<b>{quartile}</b><br>"
        f"Total seats: {len(plot_df)}<br><br>"
        f"<b>Above diagonal</b><br>"
        f"Third vote > BJP–TMC diff<br>"
        f"<span style='color:#ff9800;'>BJP ahead:</span> {bjp_above}<br>"
        f"<span style='color:#2ca25f;'>TMC ahead:</span> {tmc_above}<br><br>"
        f"<b>Below diagonal</b><br>"
        f"Third vote ≤ BJP–TMC diff<br>"
        f"<span style='color:#ff9800;'>BJP ahead:</span> {bjp_below}<br>"
        f"<span style='color:#2ca25f;'>TMC ahead:</span> {tmc_below}"
    )

    fig = go.Figure()

    fig.add_scatter(
        x=plot_df["BJP AITC Difference"],
        y=plot_df["Third Highest Vote"],
        mode="markers",
        marker=dict(
            size=9,
            color=colors,
            opacity=0.75,
            line=dict(color="black", width=0.6),
        ),
        customdata=np.column_stack([
            plot_df[CSV_AC_NAME_COL],
            plot_df["Winner"],
            plot_df["BJP vote count"],
            plot_df["AITC vote count"],
            plot_df["Third Party"],
            plot_df["Third Highest Vote"],
            plot_df["BJP AITC Difference"],
            plot_df["Leading Party BJP/AITC"],
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Winner: %{customdata[1]}<br>"
            "BJP: %{customdata[2]:,.0f}<br>"
            "TMC: %{customdata[3]:,.0f}<br>"
            "BJP/TMC leader: %{customdata[7]}<br>"
            "Third party: %{customdata[4]}<br>"
            "Third-party vote: %{customdata[5]:,.0f}<br>"
            "BJP–TMC diff: %{customdata[6]:,.0f}<extra></extra>"
        ),
        showlegend=False,
    )

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(size=10, color=PARTY_COLORS["BJP"], line=dict(color="black", width=0.6)),
        name="BJP ahead",
    ))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(size=10, color=PARTY_COLORS["AITC"], line=dict(color="black", width=0.6)),
        name="TMC ahead",
    ))

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_diag,
        y1=max_diag,
        line=dict(color="black", width=3, dash="dash"),
    )

    fig.add_annotation(
        x=max_diag * 0.66,
        y=max_diag * 0.78,
        text="Third-party vote = BJP–TMC difference",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.80)",
        bordercolor="black",
        borderwidth=1,
    )

    quartile_regions = [
        (0, q1, "Q1<br>closest seats"),
        (q1, q2, "Q2"),
        (q2, q3, "Q3"),
        (q3, x_max, "Q4<br>largest margins"),
    ]

    for x0, x1, label in quartile_regions:
        region = ac_df[
            (ac_df["BJP AITC Difference"] >= x0)
            & (ac_df["BJP AITC Difference"] <= x1)
        ]

        above = region[
            region["Third Highest Vote"] > region["BJP AITC Difference"]
        ]

        below = region[
            region["Third Highest Vote"] <= region["BJP AITC Difference"]
        ]

        bjp_a, tmc_a = count_summary(above)
        bjp_b, tmc_b = count_summary(below)

        x_mid = 0.5 * (x0 + x1)

        fig.add_annotation(
            x=x_mid,
            y=y_max * 0.93,
            text=(
                f"<b>{label}</b><br>"
                f"Above: "
                f"<span style='color:#ff9800;'>BJP {bjp_a}</span>, "
                f"<span style='color:#2ca25f;'>TMC {tmc_a}</span><br>"
                f"Below: "
                f"<span style='color:#ff9800;'>BJP {bjp_b}</span>, "
                f"<span style='color:#2ca25f;'>TMC {tmc_b}</span>"
            ),
            showarrow=False,
            align="center",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(0,0,0,0.35)",
            borderwidth=1,
        )

    for val, label in [(q1, "Q1"), (q2, "Median"), (q3, "Q3")]:
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color="red",
            annotation_text=label,
            annotation_position="top",
        )

    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        align="left",
        bordercolor="black",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.92)",
    )

    fig.update_layout(
        title=(
            "Scatter: BJP–TMC Difference vs Third-party Vote<br>"
            "<sup>Diagonal separates seats where third-party vote exceeds the BJP–TMC margin</sup>"
        ),
        xaxis_title="|BJP vote - TMC vote|",
        yaxis_title="Third-party vote count",
        height=760,
        plot_bgcolor="white",
        legend_title="Leader",
    )

    if log_y:
        fig.update_yaxes(type="log", title="Third-party vote count [log scale]")

    fig.update_xaxes(range=[0, x_max * 1.05])

    return fig

# ============================================================
# Folium map
# ============================================================

def map_color_by_winner(feature):
    winner = feature["properties"].get("Winner", "Unknown")

    if winner == "BJP":
        return PARTY_COLORS["BJP"]
    elif winner == "AITC":
        return PARTY_COLORS["AITC"]
    elif winner == "CPIM":
        return PARTY_COLORS["CPIM"]
    elif winner == "Unknown":
        return "#bdbdbd"
    else:
        return MAP_OTHER_COLOR

def map_color_by_bjp_aitc_leader(feature):
    leader = feature["properties"].get("Leading Party BJP/AITC", "Unknown")

    if leader == "BJP":
        return PARTY_COLORS["BJP"]      # orange
    elif leader == "AITC":
        return PARTY_COLORS["AITC"]     # green
    else:
        return MAP_OTHER_COLOR          # red (or grey if you prefer)

def count_bjp_tmc_ahead(geojson):
    bjp = 0
    tmc = 0
    other = 0

    for feature in geojson["features"]:
        leader = feature["properties"].get("Leading Party BJP/AITC", "Unknown")

        if leader == "BJP":
            bjp += 1
        elif leader == "AITC":
            tmc += 1
        else:
            other += 1

    return bjp, tmc, other    

def add_tooltip_geojson(m, geojson, name, color_function, fill_opacity=0.72):
    folium.GeoJson(
        geojson,
        name=name,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                KML_AC_NO_COL,
                KML_AC_NAME_COL,
                KML_DISTRICT_COL,
                "Winner",
                "BJP vote count",
                "AITC vote count",
                "BJP AITC Difference",
                "Leading Party BJP/AITC",
                "Total Electors",
            ],
            aliases=[
                "AC No:",
                "AC Name:",
                "District:",
                "Winner:",
                "BJP Votes:",
                "TMC Votes:",
                "BJP-TMC Difference:",
                "BJP/TMC Leader:",
                "Total Electors:",
            ],
            sticky=True,
            localize=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=[
                KML_AC_NO_COL,
                KML_AC_NAME_COL,
                KML_DISTRICT_COL,
                "Winner",
                "BJP vote count",
                "AITC vote count",
                "BJP AITC Difference",
                "Leading Party BJP/AITC",
                "Total Electors",
            ],
            aliases=[
                "AC No:",
                "AC Name:",
                "District:",
                "Winner:",
                "BJP Votes:",
                "TMC Votes:",
                "BJP-TMC Difference:",
                "BJP/TMC Leader:",
                "Total Electors:",
            ],
            localize=True,
        ),
        style_function=lambda feature: {
            "fillColor": color_function(feature),
            "color": "black",
            "weight": 0.8,
            "fillOpacity": fill_opacity,
        },
        highlight_function=lambda feature: {
            "fillColor": color_function(feature),
            "color": "black",
            "weight": 3,
            "fillOpacity": 0.95,
        },
    ).add_to(m)

def make_folium_map(full_geojson, selected_geojson=None, min_diff=None, max_diff=None):
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        tiles="cartodbpositron",
    )

    folium.GeoJson(
        full_geojson,
        name="All constituencies background",
        style_function=lambda feature: {
            "fillColor": BACKGROUND_GREY,
            "color": "black",
            "weight": 0.7,
            "fillOpacity": 0.18,
        },
    ).add_to(m)

    if selected_geojson is None:
        add_tooltip_geojson(
            m,
            full_geojson,
            name="Seat winner",
            color_function=map_color_by_winner,
            fill_opacity=0.72,
        )

        bjp_ahead, tmc_ahead, other_ahead = count_bjp_tmc_ahead(full_geojson)

        legend_title = "Default seat winner map"
        subtitle = "Orange: BJP winner | Green: TMC winner | Red: Others"
        count_text = (
            f"BJP ahead of TMC: <b>{bjp_ahead}</b><br>"
            f"TMC ahead of BJP: <b>{tmc_ahead}</b>"
        )

    else:
        add_tooltip_geojson(
            m,
            selected_geojson,
            name="Selected constituencies",
            color_function=map_color_by_bjp_aitc_leader,
            fill_opacity=0.78,
        )

        bjp_ahead, tmc_ahead, other_ahead = count_bjp_tmc_ahead(selected_geojson)

        legend_title = f"BJP–TMC vote difference: {min_diff:.0f} – {max_diff:.0f}"
        subtitle = "Orange: BJP ahead | Green: TMC ahead"
        count_text = (
            f"Selected seats: <b>{bjp_ahead + tmc_ahead + other_ahead}</b><br>"
            f"BJP ahead: <b>{bjp_ahead}</b><br>"
            f"TMC ahead: <b>{tmc_ahead}</b>"
        )

    legend_html = f"""
    <div style="
        position: fixed;
        top: 15px;
        left: 55px;
        z-index: 9999;
        background-color: rgba(255,255,255,0.90);
        padding: 10px;
        border: 1.5px solid black;
        border-radius: 8px;
        font-size: 14px;">
        <b>{legend_title}</b><br>
        {subtitle}<br><br>
        <span style="color:#ff9800;"><b>■</b></span> BJP<br>
        <span style="color:#2ca25f;"><b>■</b></span> TMC/AITC<br>
        <span style="color:#d73027;"><b>■</b></span> Others<br><br>
        {count_text}
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    return m

# ============================================================
# Streamlit UI helpers
# ============================================================

def boxed_header(title):
    st.markdown(
        f"""
        <div style="
            border: 1.5px solid rgba(0,0,0,0.35);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0 1rem 0;
            background-color: rgba(245,245,245,0.55);">
            <h3 style="margin:0;">{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Main app
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="West Bengal 2021 Election Dashboard")

    st.title("West Bengal Assembly Election 2021 Dashboard")

    df = load_csv_data(CSV_FILE)                ## maine election data (credit: kaggle)
    geojson = read_kml_to_geojson(KML_FILE)     ## geometry of WB assembly map (credit: OpenStreetMap)

    ac_df = make_ac_party_vote_dataframe(df)
    ac_df = add_winner_and_differences(ac_df)
    
    full_geojson = attach_votes_to_geojson(geojson, ac_df)


    ##visualize difference in MAP
    st.sidebar.header("Controls")
    bins = st.sidebar.slider("Histogram bins", 10, 80, BINS)

    min_available = int(ac_df["BJP AITC Difference"].min())
    max_available = int(ac_df["BJP AITC Difference"].max())
    
    with st.sidebar.form("map_control_form"):
        st.markdown("### Vote-difference range")

        min_diff_input = st.number_input(
            "Minimum vote difference",
            min_value=min_available,
            max_value=max_available,
            value=st.session_state.get("min_diff", min_available),
            step=500,
        )

        max_diff_input = st.number_input(
            "Maximum vote difference",
            min_value=min_available,
            max_value=max_available,
            value=st.session_state.get("max_diff", max_available),
            step=500,
        )
        
        visualize = st.form_submit_button("Visualize selected range")
        reset_map = st.form_submit_button("Default seat map")

    if "show_selected" not in st.session_state:
        st.session_state["show_selected"] = False

    if visualize:
        st.session_state["show_selected"] = True
        st.session_state["min_diff"] = min_diff_input
        st.session_state["max_diff"] = max_diff_input

    if reset_map:
        st.session_state["show_selected"] = False

    
    ## candidate summary    
    boxed_header("Candidate-level summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.plotly_chart(plot_age_distribution(df), width="stretch")

    with c2:
        st.plotly_chart(
            pie_from_column(df, "SEX", "Gender Distribution"),
            width="stretch",
        )

    with c3:
        st.plotly_chart(
            pie_from_column(df, "CATEGORY", "Category Distribution"),
            width="stretch",
        )


    ## summary 2
    boxed_header("Seat and vote share summary")

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.plotly_chart(make_total_seat_pie(ac_df), width="stretch")

    with c2:
        with st.container(border=True):
            st.plotly_chart(make_vote_share_pie(ac_df), width="stretch")

    boxed_header("BJP vs TMC vote-difference histogram and map")

    if st.session_state["show_selected"]:
        min_sel = st.session_state["min_diff"]
        max_sel = st.session_state["max_diff"]

        if min_sel > max_sel:
            st.error("Minimum vote difference cannot be greater than maximum vote difference.")
            selected_geojson = None
            selected_ac_df = ac_df.copy()
        else:
            selected_ac_df = ac_df[
                (ac_df["BJP AITC Difference"] >= min_sel)
                & (ac_df["BJP AITC Difference"] <= max_sel)
            ].copy()

            selected_geojson = filter_geojson_by_difference(
                full_geojson,
                min_sel,
                max_sel,
            )
    else:
        min_sel = None
        max_sel = None
        selected_geojson = None
        selected_ac_df = ac_df.copy()

    c1, c2 = st.columns([1, 1])

    with c1:
        hist_fig = make_bjp_aitc_histogram(
            ac_df,
            bins=bins,
            min_diff=min_sel,
            max_diff=max_sel,
        )
        st.plotly_chart(hist_fig, width="stretch")

    with c2:
        if selected_geojson is None:
            m = make_folium_map(full_geojson)
        else:
            m = make_folium_map(
                full_geojson,
                selected_geojson=selected_geojson,
                min_diff=min_sel,
                max_diff=max_sel,
            )

        st_folium(m, width=850, height=650, returned_objects=[])        

    boxed_header("Selected constituencies")

    st.write(f"Number of displayed constituencies: **{len(selected_ac_df)}**")

    st.dataframe(
        selected_ac_df[
            [
                "AC NAME",
                "Winner",
                "BJP vote count",
                "AITC vote count",
                "BJP AITC Difference",
                "Leading Party BJP/AITC",
                "Total Electors",
                "Winner Second Difference",
            ]
        ].sort_values("BJP AITC Difference", ascending=False),
        width="stretch",
    )

    csv_out = selected_ac_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download selected constituencies as CSV",
        data=csv_out,
        file_name="selected_constituencies.csv",
        mime="text/csv",
    )

    ## Third force
    boxed_header("Third-party strength in BJP–TMC contests")

    q_choice = st.selectbox(
        "Choose BJP–TMC vote-difference quartile",
        [
            "All",
            "Q1: closest seats",
            "Q2",
            "Q3",
            "Q4: largest margins",
        ],
    )

    n_seats = st.slider(
        "Number of seats to show",
        min_value=5,
        max_value=100,
        value=70,
        step=5,
    )

    sort_by = st.radio(
        "Sort seats by",
        ["BJP AITC Difference", "Third Highest Vote"],
        horizontal=True,
    )

    log_y = st.toggle(
        "Use log scale for y-axis",
        value=True,
    )

    third_fig = make_third_vote_vs_difference_plot(
        ac_df,
        quartile=q_choice,
        n_seats=n_seats,
        sort_by=sort_by,
        log_y=log_y,
    )

    st.plotly_chart(third_fig, width="stretch")

    ## Seat Heatmap vs Third Force
    
    boxed_header("2D density: BJP–TMC difference vs third-party vote")

    st.markdown(
        """
        This plot checks whether the third-party vote count is larger than the BJP–TMC margin.

        - **Above diagonal**: third-party votes are larger than the BJP–TMC difference.
        - **Below diagonal**: BJP–TMC difference is larger than third-party votes.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        x_bin_width = st.number_input(
            "X-axis bin width: BJP–TMC vote difference",
            min_value=500,
            max_value=50000,
            value=3000,
            step=500,
        )

    with c2:
        y_bin_width = st.number_input(
            "Y-axis bin width: third-party vote count",
            min_value=500,
            max_value=50000,
            value=3000,
            step=500,
        )
        
    with c3:
        heatmap_quartile = st.selectbox(
            "Choose quartile region",
            [
                "All",
                "Q1: closest seats",
                "Q2",
                "Q3",
                "Q4: largest margins",
            ],
            index=0,
        )

    heatmap_fig = make_diff_vs_third_vote_heatmap(
        ac_df,
        x_bin_width=x_bin_width,
        y_bin_width=y_bin_width,
        quartile=heatmap_quartile,
    )

    st.plotly_chart(heatmap_fig, width="stretch")


    ##scatter
    boxed_header("Scatter view: volatility and third-party vote")

    scatter_log_y = st.toggle(
        "Use log scale for scatter y-axis",
        value=False,
    )

    scatter_fig = make_diff_vs_third_vote_scatter(
        ac_df,
        quartile=heatmap_quartile,
        log_y=scatter_log_y,
    )

    st.plotly_chart(scatter_fig, width="stretch")
    
if __name__ == "__main__":
    main()
