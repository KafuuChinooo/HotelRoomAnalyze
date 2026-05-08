from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ai_explainer import (  # noqa: E402
    TableAnalysisContext,
    TableAnalysisResponse,
    analyzeTableWithAI,
)
from clean_data import clean_hotel_bookings  # noqa: E402
from config import DEFAULT_DATA_PATH, MODEL_FEATURE_COLUMNS, MONTH_ORDER, TARGET  # noqa: E402
from features import add_features  # noqa: E402
from load_data import load_hotel_bookings  # noqa: E402
from model import METRICS_PATH, MODEL_PATH, load_model, predict_cancellation, train_cancellation_model  # noqa: E402


st.set_page_config(page_title="Hotel Booking Cancellation", page_icon="HB", layout="wide")

COLOR_OK = "#2a9d8f"
COLOR_RISK = "#e76f51"
COLOR_INK = "#25313b"
COLOR_MUTED = "#5f6b76"
COLOR_MID = "#457b9d"
COLOR_WARN = "#e9c46a"
COLOR_SCALE = ["#edf6f4", "#bde0d8", "#6ab7a8", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
PLOTLY_TEMPLATE = "plotly_white"


st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8ee;
        border-radius: 8px;
        padding: 14px 16px;
      }
      .insight-panel {
        border: 1px solid #dfe7ed;
        border-radius: 8px;
        padding: 14px 16px;
        background: #fbfcfd;
      }
      .small-muted { color: #66727c; font-size: 0.88rem; }
      div[data-testid="stButton"] > button {
        border-radius: 8px;
        border: 1px solid #dfe7ed;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dashboard_data(path: str) -> pd.DataFrame:
    df = load_hotel_bookings(path)
    return add_features(clean_hotel_bookings(df))


@st.cache_resource(show_spinner=False)
def load_dashboard_model(data_hash: int):
    del data_hash
    if not MODEL_PATH.exists():
        train_cancellation_model(load_dashboard_data(str(DEFAULT_DATA_PATH)))
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        hotels = st.multiselect("Hotel", sorted(df["hotel"].unique()), default=sorted(df["hotel"].unique()))
        segments = st.multiselect(
            "Market segment",
            sorted(df["market_segment"].unique()),
            default=sorted(df["market_segment"].unique()),
        )
        deposits = st.multiselect(
            "Deposit type",
            sorted(df["deposit_type"].unique()),
            default=sorted(df["deposit_type"].unique()),
        )
        customer_types = st.multiselect(
            "Customer type",
            sorted(df["customer_type"].unique()),
            default=sorted(df["customer_type"].unique()),
        )
        years = st.multiselect(
            "Arrival year",
            sorted(df["arrival_date_year"].unique()),
            default=sorted(df["arrival_date_year"].unique()),
        )
        months = st.multiselect(
            "Arrival month",
            list(MONTH_ORDER.keys()),
            default=list(MONTH_ORDER.keys()),
        )
        lead_min, lead_max = int(df["lead_time"].min()), int(df["lead_time"].max())
        lead_range = st.slider("Lead time", lead_min, lead_max, (lead_min, min(lead_max, 365)))
        status = st.radio("Cancellation", ["All", "Not canceled", "Canceled"], horizontal=True)

    month_nums = [MONTH_ORDER[m] for m in months]
    st.session_state["active_filter_summary"] = {
        "hotels": ", ".join(hotels),
        "market_segments": ", ".join(segments[:6]) + ("..." if len(segments) > 6 else ""),
        "deposit_types": ", ".join(deposits),
        "customer_types": ", ".join(customer_types),
        "arrival_years": ", ".join(str(year) for year in years),
        "arrival_months": ", ".join(months[:6]) + ("..." if len(months) > 6 else ""),
        "lead_time_min": lead_range[0],
        "lead_time_max": lead_range[1],
        "cancellation_status": status,
    }
    filtered = df[
        df["hotel"].isin(hotels)
        & df["market_segment"].isin(segments)
        & df["deposit_type"].isin(deposits)
        & df["customer_type"].isin(customer_types)
        & df["arrival_date_year"].isin(years)
        & df["arrival_month_num"].isin(month_nums)
        & df["lead_time"].between(*lead_range)
    ].copy()
    if status == "Not canceled":
        filtered = filtered[filtered[TARGET].eq(0)]
    elif status == "Canceled":
        filtered = filtered[filtered[TARGET].eq(1)]
    return filtered


def metric_row(df: pd.DataFrame) -> None:
    revenue_at_risk = float(df.loc[df[TARGET].eq(1), "adr"].sum()) if len(df) else 0
    cols = st.columns(6)
    metrics = [
        ("Bookings", f"{len(df):,}"),
        ("Cancellation rate", f"{(df[TARGET].mean() if len(df) else 0):.1%}"),
        ("Canceled bookings", f"{int(df[TARGET].sum()):,}" if len(df) else "0"),
        ("Avg lead time", f"{df['lead_time'].mean():.1f} days" if len(df) else "0 days"),
        ("Avg ADR", f"{df['adr'].mean():.2f}" if len(df) else "0"),
        ("ADR at risk", f"{revenue_at_risk:,.0f}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def cancellation_rate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, "bookings", "cancellations", "cancellation_rate"])
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(bookings=(TARGET, "size"), cancellations=(TARGET, "sum"), cancellation_rate=(TARGET, "mean"))
        .reset_index()
        .sort_values("bookings", ascending=False)
    )
    return grouped


def add_selection_meta(
    frame: pd.DataFrame,
    chart: str,
    dimension: str,
    value_col: str,
    label_col: str | None = None,
) -> pd.DataFrame:
    data = frame.copy()
    label_col = label_col or value_col
    data["_chart"] = chart
    data["_dimension"] = dimension
    data["_value"] = data[value_col].astype(str)
    data["_label"] = data[label_col].astype(str)
    return data


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=18, r=18, t=54, b=28),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=COLOR_INK),
        title=dict(font=dict(color=COLOR_INK, size=17)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", bordercolor="#cfd8df", font=dict(color=COLOR_INK, size=13)),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#edf2f5",
        zeroline=False,
        color=COLOR_INK,
        title_font=dict(color=COLOR_INK),
        tickfont=dict(color=COLOR_MUTED),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#edf2f5",
        zeroline=False,
        color=COLOR_INK,
        title_font=dict(color=COLOR_INK),
        tickfont=dict(color=COLOR_MUTED),
    )
    fig.update_traces(textfont=dict(color=COLOR_INK), selector=dict(type="bar"))
    fig.update_traces(textfont=dict(color=COLOR_INK), selector=dict(type="pie"))
    fig.update_traces(textfont=dict(color=COLOR_INK), selector=dict(type="treemap"))
    return fig


def _dataframe_selection_rows(state: Any) -> list[int]:
    if state is None:
        return []
    selection = getattr(state, "selection", None)
    if selection is None and isinstance(state, dict):
        selection = state.get("selection")
    if selection is None:
        return []
    rows = getattr(selection, "rows", None)
    if rows is None and isinstance(selection, dict):
        rows = selection.get("rows")
    return [int(row) for row in rows or []]


def _table_context(table_name: str, df: pd.DataFrame, selected_rows: list[int]) -> TableAnalysisContext:
    sample = df.iloc[selected_rows].head(5) if selected_rows else df.head(10)
    numeric_summary = df.select_dtypes(include="number").describe().round(3).to_dict()
    table_data = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "selected_rows": selected_rows[:10],
        "selected_sample": sample.to_dict(orient="records"),
        "numeric_summary": numeric_summary,
        "filters": st.session_state.get("active_filter_summary", {}),
    }
    return {
        "tableName": table_name,
        "tableData": json.dumps(table_data, ensure_ascii=False, default=str),
    }


def open_table_analysis_popup(table_name: str, df: pd.DataFrame, selected_rows: list[int]) -> None:
    st.session_state["table_analysis_context"] = _table_context(table_name, df, selected_rows)
    st.session_state["table_analysis_popup_open"] = True
    st.session_state["table_analysis_status"] = "initial"
    st.session_state["table_analysis_response"] = None
    st.session_state["table_analysis_error"] = None
    st.session_state["table_analysis_key"] = f"{table_name}:{','.join(str(row) for row in selected_rows[:10])}"


def close_table_analysis_popup() -> None:
    context = st.session_state.get("table_analysis_context") or {}
    key = st.session_state.get("table_analysis_key") or str(context.get("tableName", ""))
    st.session_state["dismissed_table_analysis_key"] = key
    st.session_state["table_analysis_popup_open"] = False
    st.session_state["table_analysis_status"] = "initial"
    st.session_state["table_analysis_response"] = None
    st.session_state["table_analysis_error"] = None


def render_table_analysis(response: TableAnalysisResponse) -> None:
    sections = [
        ("Tổng quan", response.get("overview", "")),
        ("Số liệu / mẫu nổi bật", response.get("keyPatterns", "")),
        ("Lý do có thể", response.get("possibleReasons", "")),
        ("Dự đoán / diễn giải", response.get("prediction", "")),
        ("Lưu ý", response.get("limitations", "")),
    ]
    for title, body in sections:
        if body:
            st.markdown(f"#### {title}")
            st.write(body)


@st.dialog("AI phân tích bảng", width="small", dismissible=True, on_dismiss="rerun")
def table_analysis_popup() -> None:
    context = st.session_state.get("table_analysis_context")
    if not context:
        st.warning("Chưa chọn bảng để phân tích.")
        if st.button("Đóng", use_container_width=True):
            close_table_analysis_popup()
            st.rerun()
        return

    table_name = context.get("tableName", "bảng này")
    st.write(f"Bạn muốn AI phân tích bảng **{table_name}** không?")
    status = st.session_state.get("table_analysis_status", "initial")

    if status == "loading":
        st.info("AI đang phân tích...")
    elif status == "success":
        response = st.session_state.get("table_analysis_response")
        if response:
            render_table_analysis(response)
    elif status == "error":
        st.error("Không thể phân tích bảng. Vui lòng thử lại.")
        if st.session_state.get("table_analysis_error"):
            st.caption(str(st.session_state["table_analysis_error"]))

    st.divider()
    left, right = st.columns(2)
    with left:
        if st.button("Không" if status == "initial" else "Đóng", use_container_width=True):
            close_table_analysis_popup()
            st.rerun()
    with right:
        if status in {"initial", "error"}:
            label = "Có, phân tích" if status == "initial" else "Thử lại"
            if st.button(label, type="primary", use_container_width=True):
                st.session_state["table_analysis_status"] = "loading"
                st.session_state["table_analysis_error"] = None
                with st.spinner("AI đang phân tích..."):
                    try:
                        st.session_state["table_analysis_response"] = analyzeTableWithAI(context, ROOT)
                        st.session_state["table_analysis_status"] = "success"
                    except Exception as exc:
                        st.session_state["table_analysis_response"] = None
                        st.session_state["table_analysis_error"] = str(exc)
                        st.session_state["table_analysis_status"] = "error"
                st.rerun()


def selectable_table(table_name: str, df: pd.DataFrame, key: str, height: int = 560) -> None:
    state = st.dataframe(
        df,
        use_container_width=True,
        height=height,
        key=key,
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_rows = _dataframe_selection_rows(state)
    if selected_rows:
        selection_key = f"{table_name}:{','.join(str(row) for row in selected_rows[:10])}"
        dismissed = st.session_state.get("dismissed_table_analysis_key")
        active = st.session_state.get("table_analysis_key")
        if selection_key != dismissed and selection_key != active:
            open_table_analysis_popup(table_name, df, selected_rows)


def selectable_chart(fig: go.Figure, key: str, chart: str, height: int = 360) -> None:
    st.plotly_chart(
        style_fig(fig, height=height),
        use_container_width=True,
        key=key,
        config={"displayModeBar": True, "displaylogo": False},
    )


def donut_cancellation(df: pd.DataFrame) -> go.Figure:
    counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0).reset_index()
    counts.columns = ["is_canceled", "bookings"]
    counts["label"] = counts["is_canceled"].map({0: "Not canceled", 1: "Canceled"})
    counts["cancellation_rate"] = counts["is_canceled"].astype(float)
    counts = add_selection_meta(counts, "Cancellation mix", "is_canceled", "is_canceled", "label")
    fig = px.pie(
        counts,
        names="label",
        values="bookings",
        hole=0.62,
        color="label",
        color_discrete_map={"Not canceled": COLOR_OK, "Canceled": COLOR_RISK},
        custom_data=["_chart", "_dimension", "_value", "_label", "bookings", "cancellation_rate"],
        title="Cancellation mix",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", insidetextfont=dict(color=COLOR_INK))
    return fig


def monthly_line(df: pd.DataFrame) -> go.Figure:
    monthly = (
        df.groupby(["arrival_date_year", "arrival_month_num"], dropna=False)
        .agg(bookings=(TARGET, "size"), cancellation_rate=(TARGET, "mean"))
        .reset_index()
        .sort_values(["arrival_date_year", "arrival_month_num"])
    )
    monthly["period"] = (
        monthly["arrival_date_year"].astype(str)
        + "-"
        + monthly["arrival_month_num"].astype(int).astype(str).str.zfill(2)
    )
    monthly = add_selection_meta(monthly, "Monthly cancellation trend", "period", "period")
    fig = px.line(
        monthly,
        x="period",
        y="cancellation_rate",
        markers=True,
        color_discrete_sequence=[COLOR_MID],
        custom_data=["_chart", "_dimension", "_value", "_label", "bookings", "cancellation_rate"],
        title="Monthly cancellation trend",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def bar_rate(df: pd.DataFrame, dimension: str, title: str, top_n: int = 12, horizontal: bool = False) -> go.Figure:
    data = cancellation_rate(df, dimension).head(top_n)
    data = add_selection_meta(data, title, dimension, dimension)
    if horizontal:
        data = data.sort_values("cancellation_rate")
        fig = px.bar(
            data,
            y=dimension,
            x="cancellation_rate",
            color="bookings",
            orientation="h",
            color_continuous_scale=COLOR_SCALE,
            custom_data=["_chart", "_dimension", "_value", "_label", "bookings", "cancellation_rate"],
            title=title,
        )
        fig.update_xaxes(tickformat=".0%")
    else:
        fig = px.bar(
            data,
            x=dimension,
            y="cancellation_rate",
            color="bookings",
            color_continuous_scale=COLOR_SCALE,
            custom_data=["_chart", "_dimension", "_value", "_label", "bookings", "cancellation_rate"],
            title=title,
        )
        fig.update_yaxes(tickformat=".0%")
    return fig


def stacked_status_bar(df: pd.DataFrame, dimension: str, title: str) -> go.Figure:
    table = (
        df.groupby([dimension, TARGET], dropna=False)
        .size()
        .rename("bookings")
        .reset_index()
        .sort_values("bookings", ascending=False)
    )
    rates = df.groupby(dimension, dropna=False)[TARGET].mean().rename("cancellation_rate").reset_index()
    table = table.merge(rates, on=dimension, how="left")
    table["status"] = table[TARGET].map({0: "Not canceled", 1: "Canceled"})
    totals = table.groupby(dimension)["bookings"].transform("sum")
    table["share"] = table["bookings"] / totals
    table = add_selection_meta(table, title, dimension, dimension)
    fig = px.bar(
        table,
        x=dimension,
        y="share",
        color="status",
        color_discrete_map={"Not canceled": COLOR_OK, "Canceled": COLOR_RISK},
        custom_data=["_chart", "_dimension", "_value", "_label", "bookings", "cancellation_rate"],
        title=title,
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def heatmap_hotel_month(df: pd.DataFrame) -> go.Figure:
    data = (
        df.groupby(["hotel", "arrival_month_num"], dropna=False)
        .agg(cancellation_rate=(TARGET, "mean"), bookings=(TARGET, "size"))
        .reset_index()
    )
    pivot = data.pivot(index="hotel", columns="arrival_month_num", values="cancellation_rate")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=COLOR_SCALE,
        zmin=0,
        zmax=max(0.01, float(df[TARGET].mean()) * 1.8 if len(df) else 1),
        title="Cancellation heatmap by hotel and month",
    )
    fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%", tickfont=dict(color=COLOR_INK), title_font=dict(color=COLOR_INK)))
    return fig


def country_treemap(df: pd.DataFrame) -> go.Figure:
    data = cancellation_rate(df, "country").head(25)
    data = add_selection_meta(data, "Top country risk map", "country", "country")
    fig = px.treemap(
        data,
        path=["country"],
        values="bookings",
        color="cancellation_rate",
        color_continuous_scale=COLOR_SCALE,
        custom_data=["_chart", "_dimension", "_value", "_label", "bookings", "cancellation_rate"],
        title="Top countries by volume and cancellation risk",
    )
    fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%", tickfont=dict(color=COLOR_INK), title_font=dict(color=COLOR_INK)))
    fig.update_traces(textfont=dict(color=COLOR_INK), marker=dict(line=dict(color="white", width=1)))
    return fig


def scatter_lead_adr(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(7000, len(df)), random_state=42) if len(df) else df
    sample = sample.copy()
    sample["status"] = sample[TARGET].map({0: "Not canceled", 1: "Canceled"})
    sample["_chart"] = "Lead time vs ADR scatter"
    sample["_dimension"] = "booking"
    sample["_value"] = sample.index.astype(str)
    sample["_label"] = sample["hotel"] + " | " + sample["market_segment"]
    fig = px.scatter(
        sample,
        x="lead_time",
        y="adr",
        color="status",
        opacity=0.45,
        color_discrete_map={"Not canceled": COLOR_OK, "Canceled": COLOR_RISK},
        hover_data=["hotel", "market_segment", "deposit_type", "country", "total_nights", "total_of_special_requests"],
        custom_data=["_chart", "_dimension", "_value", "_label", "lead_time", TARGET],
        title="Lead time vs ADR by booking status",
    )
    fig.update_yaxes(range=[max(-10, sample["adr"].quantile(0.01)), sample["adr"].quantile(0.99)])
    return fig


def box_lead_time(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["status"] = data[TARGET].map({0: "Not canceled", 1: "Canceled"})
    fig = px.box(
        data,
        x="status",
        y="lead_time",
        color="status",
        color_discrete_map={"Not canceled": COLOR_OK, "Canceled": COLOR_RISK},
        points=False,
        title="Lead time distribution by outcome",
    )
    return fig


def histogram_lead_time(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["status"] = data[TARGET].map({0: "Not canceled", 1: "Canceled"})
    fig = px.histogram(
        data,
        x="lead_time",
        color="status",
        nbins=55,
        barmode="overlay",
        opacity=0.72,
        color_discrete_map={"Not canceled": COLOR_OK, "Canceled": COLOR_RISK},
        title="Lead time distribution",
    )
    return fig


def default_row(df: pd.DataFrame) -> dict:
    row = {}
    for col in MODEL_FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = float(df[col].median())
        else:
            row[col] = str(df[col].mode(dropna=True).iloc[0])
    return row


def prediction_tab(df: pd.DataFrame, model) -> None:
    st.subheader("Cancellation probability")
    base = default_row(df)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            base["hotel"] = st.selectbox("Hotel", sorted(df["hotel"].unique()))
            base["lead_time"] = st.number_input("Lead time", 0, 800, int(base["lead_time"]))
            month = st.selectbox("Arrival month", list(MONTH_ORDER.keys()), index=int(base["arrival_month_num"]) - 1)
            base["arrival_month_num"] = MONTH_ORDER[month]
            base["arrival_date_year"] = st.selectbox("Arrival year", sorted(df["arrival_date_year"].unique()), index=0)
            base["arrival_date_week_number"] = st.number_input(
                "Arrival week number", 1, 53, int(base["arrival_date_week_number"])
            )
        with c2:
            base["stays_in_weekend_nights"] = st.number_input(
                "Weekend nights", 0, 20, int(base["stays_in_weekend_nights"])
            )
            base["stays_in_week_nights"] = st.number_input("Week nights", 0, 60, int(base["stays_in_week_nights"]))
            base["adults"] = st.number_input("Adults", 0, 10, int(base["adults"]))
            base["children"] = st.number_input("Children", 0, 10, int(base["children"]))
            base["babies"] = st.number_input("Babies", 0, 10, int(base["babies"]))
        with c3:
            countries = sorted(df["country"].unique())
            base["meal"] = st.selectbox("Meal", sorted(df["meal"].unique()))
            base["country"] = st.selectbox("Country", countries, index=countries.index(base["country"]))
            base["market_segment"] = st.selectbox("Market segment", sorted(df["market_segment"].unique()))
            base["distribution_channel"] = st.selectbox("Distribution channel", sorted(df["distribution_channel"].unique()))
            base["deposit_type"] = st.selectbox("Deposit type", sorted(df["deposit_type"].unique()))

        c4, c5, c6 = st.columns(3)
        with c4:
            base["customer_type"] = st.selectbox("Customer type", sorted(df["customer_type"].unique()))
            base["reserved_room_type"] = st.selectbox("Reserved room", sorted(df["reserved_room_type"].unique()))
            base["is_repeated_guest"] = int(st.checkbox("Repeated guest", value=bool(base["is_repeated_guest"])))
            base["previous_cancellations"] = st.number_input(
                "Previous cancellations", 0, 30, int(base["previous_cancellations"])
            )
        with c5:
            base["booking_changes"] = st.number_input("Booking changes", 0, 30, int(base["booking_changes"]))
            base["days_in_waiting_list"] = st.number_input(
                "Days in waiting list", 0, 400, int(base["days_in_waiting_list"])
            )
            base["required_car_parking_spaces"] = st.number_input(
                "Parking spaces", 0, 10, int(base["required_car_parking_spaces"])
            )
            base["total_of_special_requests"] = st.number_input(
                "Special requests", 0, 10, int(base["total_of_special_requests"])
            )
        with c6:
            base["adr"] = st.number_input("ADR", -10.0, 6000.0, float(base["adr"]), step=5.0)
            base["agent"] = st.number_input("Agent ID", 0, 600, int(base["agent"]))
            base["company"] = st.number_input("Company ID", 0, 600, int(base["company"]))
            base["previous_bookings_not_canceled"] = st.number_input(
                "Previous non-canceled bookings", 0, 100, int(base["previous_bookings_not_canceled"])
            )

        total_nights = base["stays_in_weekend_nights"] + base["stays_in_week_nights"]
        total_guests = base["adults"] + base["children"] + base["babies"]
        base["arrival_date_day_of_month"] = 15
        base["total_nights"] = total_nights
        base["total_guests"] = total_guests
        base["has_children"] = int(base["children"] + base["babies"] > 0)
        base["has_agent"] = int(base["agent"] > 0)
        base["has_company"] = int(base["company"] > 0)
        base["adr_per_person"] = base["adr"] / total_guests if total_guests else 0

        submitted = st.form_submit_button("Score booking", use_container_width=True)

    if submitted:
        probability = predict_cancellation(model, base)
        st.metric("Predicted cancellation probability", f"{probability:.1%}")
        if probability >= 0.65:
            st.warning("High-risk booking. Review deposit policy, lead time, and special request context.")
        elif probability >= 0.35:
            st.info("Moderate cancellation risk.")
        else:
            st.success("Lower cancellation risk.")


def dashboard_tab(df: pd.DataFrame) -> None:
    st.caption("Click any KPI card or chart mark to open the AI explanation modal.")
    a, b = st.columns([1, 1.45])
    with a:
        selectable_chart(donut_cancellation(df), "chart_cancel_mix", "Cancellation mix", height=340)
    with b:
        selectable_chart(monthly_line(df), "chart_monthly", "Monthly cancellation trend", height=340)

    c1, c2, c3 = st.columns(3)
    with c1:
        selectable_chart(bar_rate(df, "hotel", "Cancellation by hotel"), "chart_hotel", "Cancellation by hotel", height=360)
    with c2:
        selectable_chart(
            bar_rate(df, "market_segment", "Cancellation by market segment", horizontal=True),
            "chart_market",
            "Cancellation by market segment",
            height=360,
        )
    with c3:
        selectable_chart(
            stacked_status_bar(df, "deposit_type", "Canceled vs not canceled by deposit"),
            "chart_deposit_stack",
            "Canceled vs not canceled by deposit",
            height=360,
        )

    d1, d2 = st.columns([1.05, 1.35])
    with d1:
        selectable_chart(heatmap_hotel_month(df), "chart_heatmap", "Cancellation heatmap by hotel and month", height=390)
    with d2:
        selectable_chart(country_treemap(df), "chart_country_tree", "Top country risk map", height=390)

    e1, e2 = st.columns([1.45, 1])
    with e1:
        selectable_chart(scatter_lead_adr(df), "chart_scatter", "Lead time vs ADR scatter", height=420)
    with e2:
        selectable_chart(bar_rate(df, "lead_time_band", "Cancellation by lead-time band"), "chart_lead_band", "Cancellation by lead-time band", height=420)

    f1, f2 = st.columns(2)
    with f1:
        selectable_chart(box_lead_time(df), "chart_box_lead", "Lead time distribution by outcome", height=360)
    with f2:
        selectable_chart(histogram_lead_time(df), "chart_hist_lead", "Lead time distribution", height=360)


def model_tab(metrics: dict) -> None:
    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        c2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c3.metric("Precision", f"{metrics['precision']:.3f}")
        c4.metric("Recall", f"{metrics['recall']:.3f}")
        c5.metric("F1", f"{metrics['f1']:.3f}")
        top = pd.DataFrame(metrics.get("top_model_signals", []))
        if not top.empty:
            fig = px.bar(
                top.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=COLOR_SCALE,
                title="Model signal importance",
            )
            st.plotly_chart(style_fig(fig, height=560), use_container_width=True)
            st.caption("Chọn một dòng trong bảng dưới để AI phân tích bảng này.")
            selectable_table("Model signal importance", top, "table_model_signals", height=300)
    st.caption("Leakage columns excluded from scoring: reservation_status, reservation_status_date, assigned_room_type.")


def main() -> None:
    df = load_dashboard_data(str(DEFAULT_DATA_PATH))
    model = load_dashboard_model(hash(tuple(df.shape)))
    metrics = load_metrics()
    filtered = filter_data(df)
    st.session_state["filtered_dashboard_df"] = filtered

    st.title("Hotel Booking Cancellation Analytics")
    st.caption("Portfolio dashboard for demand, cancellation risk, segment diagnosis, and booking-level prediction.")
    metric_row(filtered)

    tab_dashboard, tab_prediction, tab_model, tab_data = st.tabs(["Dashboard", "Prediction", "Model", "Data"])

    with tab_dashboard:
        dashboard_tab(filtered)
    with tab_prediction:
        prediction_tab(df, model)
    with tab_model:
        model_tab(metrics)
    with tab_data:
        st.caption("Chọn một dòng trong bảng để mở popup phân tích bằng AI.")
        selectable_table("Filtered booking data", filtered, "table_filtered_booking_data", height=680)

    if st.session_state.get("table_analysis_popup_open"):
        table_analysis_popup()


if __name__ == "__main__":
    main()
