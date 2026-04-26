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

    # Normalize party names
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

    # Actual winner from candidate-level rows
    winner_rows = (
        df.sort_values(CSV_VOTE_COL, ascending=False)
        .groupby(CSV_AC_NAME_COL)
        .head(1)
        [[CSV_AC_NAME_COL, "PARTY_NORM", CSV_VOTE_COL]]
        .rename(
            columns={
                "PARTY_NORM": "Winner",
                CSV_VOTE_COL: "Winner Vote",
            }
        )
    )

    # Actual second-best candidate vote
    second_rows = (
        df.sort_values(CSV_VOTE_COL, ascending=False)
        .groupby(CSV_AC_NAME_COL)
        .nth(1)
        .reset_index()
        [[CSV_AC_NAME_COL, CSV_VOTE_COL]]
        .rename(columns={CSV_VOTE_COL: "Second Best Vote"})
    )

    ac_df = party_votes.reset_index()

    ac_df = ac_df.rename(
        columns={
            "BJP": "BJP vote count",
            "AITC": "AITC vote count",
            "CPIM": "CPIM vote count",
            "INC": "INC vote count",
        }
    )

    ac_df = ac_df.merge(winner_rows, on=CSV_AC_NAME_COL, how="left")
    ac_df = ac_df.merge(second_rows, on=CSV_AC_NAME_COL, how="left")

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
    counts, _ = np.histogram(data, bins=bin_edges)

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = bin_edges[1:] - bin_edges[:-1]

    q1, median, q3 = np.percentile(data, [25, 50, 75])

    fig = go.Figure()

    fig.add_bar(
        x=centers,
        y=counts,
        width=widths,
        customdata=np.column_stack([bin_edges[:-1], bin_edges[1:]]),
        hovertemplate=(
            "Vote difference: %{customdata[0]:.0f}–%{customdata[1]:.0f}<br>"
            "No. of ACs: %{y}<extra></extra>"
        ),
        marker_color="#9ecae1",
        marker_line_color="black",
        marker_line_width=0.5,
    )

    for val, label in [(q1, "Q1"), (median, "Median"), (q3, "Q3")]:
        fig.add_vline(x=val, line_dash="dash", annotation_text=label)

    if min_diff is not None and max_diff is not None:
        fig.add_vrect(
            x0=min_diff,
            x1=max_diff,
            opacity=0.25,
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
        height=650,
        bargap=0.05,
    )

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

    # Light grey background layer: all constituencies
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
        # Default: seat winner map
        add_tooltip_geojson(
            m,
            full_geojson,
            name="Seat winner",
            color_function=map_color_by_winner,
            fill_opacity=0.72,
        )

        legend_title = "Seat winner map"
        subtitle = "Orange: BJP | Green: TMC | Red: Others"

    else:
        # Selected range: BJP/TMC leader map
        add_tooltip_geojson(
            m,
            selected_geojson,
            name="Selected constituencies",
            color_function=map_color_by_bjp_aitc_leader,
            fill_opacity=0.78,
        )

        legend_title = f"BJP–TMC vote difference: {min_diff:.0f} – {max_diff:.0f}"
        subtitle = "Orange: BJP ahead | Green: TMC ahead"

    legend_html = f"""
    <div style="
        position: fixed;
        top: 15px;
        left: 55px;
        z-index: 9999;
        background-color: rgba(255,255,255,0.88);
        padding: 10px;
        border: 1.5px solid black;
        border-radius: 8px;
        font-size: 14px;">
        <b>{legend_title}</b><br>
        {subtitle}<br>
        <span style="color:#ff9800;"><b>■</b></span> BJP<br>
        <span style="color:#2ca25f;"><b>■</b></span> TMC/AITC<br>
        <span style="color:#d73027;"><b>■</b></span> Others
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


if __name__ == "__main__":
    main()
