from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import requests

from config import TARGET


GEMMA_3_27B_MODEL = "gemma-3-27b-it"
DEFAULT_GEMINI_MODEL = "gemma-4-31b-it"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FALLBACK_GEMINI_MODELS = [GEMMA_3_27B_MODEL, "gemma-4-26b-a4b-it", "gemini-2.0-flash"]


class DashboardVariable(TypedDict, total=False):
    name: str
    value: str | int | float | bool
    description: str


class DashboardExplanationContext(TypedDict, total=False):
    dashboardId: str
    dashboardName: str
    metricId: str
    metricName: str
    metricValue: str | int | float
    timeRange: str
    chartType: str
    independentVariables: list[DashboardVariable]
    dependentVariable: DashboardVariable
    filters: dict[str, str | int | float | bool]
    rawDataSummary: str


class DashboardExplanationResponse(TypedDict, total=False):
    summary: str
    reasonAnalysis: str
    variableImpact: str
    dependentVariableEffect: str
    limitations: str


class DashboardExplanationError(RuntimeError):
    pass


class TableAnalysisContext(TypedDict, total=False):
    tableName: str
    tableData: str


class TableAnalysisResponse(TypedDict, total=False):
    overview: str
    keyPatterns: str
    possibleReasons: str
    prediction: str
    limitations: str


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def safe_number(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def segment_stats(df: pd.DataFrame, dimension: str, value: Any) -> dict[str, Any]:
    overall_rate = float(df[TARGET].mean()) if len(df) else 0.0
    result: dict[str, Any] = {
        "overall_bookings": int(len(df)),
        "overall_cancellation_rate": round(overall_rate, 4),
    }

    if dimension in df.columns:
        segment = df[df[dimension].astype(str).eq(str(value))]
        segment_rate = float(segment[TARGET].mean()) if len(segment) else 0.0
        result.update(
            {
                "segment_dimension": dimension,
                "segment_value": str(value),
                "segment_bookings": int(len(segment)),
                "segment_cancellation_rate": round(segment_rate, 4),
                "lift_vs_overall_pp": round((segment_rate - overall_rate) * 100, 2),
            }
        )

        for col in ["hotel", "market_segment", "deposit_type", "customer_type", "lead_time_band"]:
            if col in segment.columns and len(segment):
                rates = (
                    segment.groupby(col, dropna=False)
                    .agg(bookings=(TARGET, "size"), cancellation_rate=(TARGET, "mean"))
                    .sort_values("bookings", ascending=False)
                    .head(6)
                    .reset_index()
                )
                result[f"within_segment_by_{col}"] = [
                    {
                        col: str(row[col]),
                        "bookings": int(row["bookings"]),
                        "cancellation_rate": round(float(row["cancellation_rate"]), 4),
                    }
                    for _, row in rates.iterrows()
                ]

    for col in ["hotel", "market_segment", "deposit_type", "customer_type", "lead_time_band"]:
        if col in df.columns:
            rates = (
                df.groupby(col, dropna=False)
                .agg(bookings=(TARGET, "size"), cancellation_rate=(TARGET, "mean"))
                .sort_values("cancellation_rate", ascending=False)
                .head(6)
                .reset_index()
            )
            result[f"highest_risk_by_{col}"] = [
                {
                    col: str(row[col]),
                    "bookings": int(row["bookings"]),
                    "cancellation_rate": round(float(row["cancellation_rate"]), 4),
                }
                for _, row in rates.iterrows()
            ]

    numeric_cols = ["lead_time", "adr", "total_nights", "total_of_special_requests", "booking_changes"]
    result["numeric_context"] = {
        col: {
            "mean": round(float(df[col].mean()), 3),
            "median": round(float(df[col].median()), 3),
        }
        for col in numeric_cols
        if col in df.columns
    }
    return result


def fallback_explanation(selection: dict[str, Any], stats: dict[str, Any]) -> str:
    label = selection.get("label") or selection.get("value") or "selected segment"
    rate = stats.get("segment_cancellation_rate", selection.get("cancellation_rate"))
    lift = stats.get("lift_vs_overall_pp")
    overall = stats.get("overall_cancellation_rate", 0)
    bookings = stats.get("segment_bookings", selection.get("bookings"))

    lines = [
        f"- Phan khuc `{label}` co {bookings:,} booking va cancellation rate khoang {float(rate):.1%}."
        if bookings is not None and rate is not None
        else f"- Da chon `{label}` de phan tich.",
        f"- Muc trung binh cua tap dang loc la {float(overall):.1%}; chenh lech cua phan khuc nay la {lift:+.2f} diem phan tram."
        if lift is not None
        else "- Hay so sanh voi overall cancellation rate va cac phan khuc co cung hotel/market/deposit.",
        "- Cac bien nen doc cung nhau: lead time, deposit type, market segment, country, parking spaces va special requests.",
        "- Day la giai thich EDA/model-risk, khong phai quan he nhan qua chac chan.",
    ]
    return "\n".join(lines)


def mock_dashboard_explanation(context: DashboardExplanationContext) -> DashboardExplanationResponse:
    metric = context.get("metricName", "Selected metric")
    value = context.get("metricValue", "the current value")
    independent = context.get("independentVariables", [])
    variables = ", ".join(str(item.get("name", "")) for item in independent if item.get("name")) or "the selected filters and segment variables"
    dependent = context.get("dependentVariable", {}).get("name", "the cancellation outcome")
    return {
        "summary": f"{metric} shows {value} for the selected dashboard item.",
        "reasonAnalysis": (
            "This number may appear this way because the selected segment differs from the filtered dashboard average. "
            "Use the supporting values in the modal to compare it against the overall cancellation rate."
        ),
        "variableImpact": (
            f"Variables such as {variables} may be associated with the metric. This dashboard does not prove causality, "
            "so these variables should be treated as explanatory signals rather than confirmed causes."
        ),
        "dependentVariableEffect": (
            f"The metric may affect how you interpret {dependent}. Higher cancellation-related values can indicate more "
            "operational risk, while lower values can suggest more stable booking demand."
        ),
        "limitations": "This is a local fallback explanation. Connect a valid AI API key for fuller, context-specific analysis.",
    }


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_response(text: str) -> DashboardExplanationResponse:
    cleaned = _strip_json_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "summary": cleaned,
            "reasonAnalysis": "The AI returned a free-form explanation instead of structured JSON.",
            "variableImpact": "Review the summary for the model's variable-level reasoning.",
            "dependentVariableEffect": "Review the summary for discussion of the dependent variable.",
            "limitations": "The response format was not structured, so section mapping is approximate.",
        }

    return {
        "summary": str(data.get("summary", "")),
        "reasonAnalysis": str(data.get("reasonAnalysis", "")),
        "variableImpact": str(data.get("variableImpact", "")),
        "dependentVariableEffect": str(data.get("dependentVariableEffect", "")),
        "limitations": str(data.get("limitations", "")),
    }


def _build_dashboard_prompt(context: DashboardExplanationContext) -> str:
    return (
        "You are an AI assistant explaining a dashboard to a non-technical user.\n\n"
        "Dashboard context:\n"
        f"- Dashboard name: {context.get('dashboardName')}\n"
        f"- Metric name: {context.get('metricName')}\n"
        f"- Metric value: {context.get('metricValue')}\n"
        f"- Time range: {context.get('timeRange')}\n"
        f"- Chart type: {context.get('chartType')}\n"
        f"- Independent variables: {json.dumps(context.get('independentVariables', []), ensure_ascii=False, default=safe_number)}\n"
        f"- Dependent variable: {json.dumps(context.get('dependentVariable', {}), ensure_ascii=False, default=safe_number)}\n"
        f"- Filters: {json.dumps(context.get('filters', {}), ensure_ascii=False, default=safe_number)}\n"
        f"- Raw data summary: {context.get('rawDataSummary')}\n\n"
        "Explain:\n"
        "1. What this metric means.\n"
        "2. Why this number may appear this way.\n"
        "3. Which independent variables may influence it.\n"
        "4. How this metric may affect the dependent variable.\n"
        "5. What limitations the user should consider.\n\n"
        "Use clear, concise language. Do not invent data. If the provided context is insufficient, say what information is missing. "
        "Do not claim causal impact unless the data explicitly supports causality. Use cautious wording such as "
        "'may indicate', 'is associated with', or 'could influence'.\n\n"
        "Return only valid JSON using this exact structure:\n"
        "{\n"
        '  "summary": "...",\n'
        '  "reasonAnalysis": "...",\n'
        '  "variableImpact": "...",\n'
        '  "dependentVariableEffect": "...",\n'
        '  "limitations": "..."\n'
        "}"
    )


def _build_table_prompt(context: TableAnalysisContext) -> str:
    return (
        "You are analyzing a dashboard table for a normal user.\n\n"
        f"Table name: {context.get('tableName')}\n"
        f"Table data/context: {context.get('tableData')}\n\n"
        "Please give:\n"
        "1. A short overview of what this table shows.\n"
        "2. The most important pattern or number.\n"
        "3. Possible reasons for this result.\n"
        "4. A simple prediction or interpretation.\n\n"
        "Use simple language. Do not invent facts. If the table data is not enough, say that more data is needed.\n\n"
        "Return only valid JSON using this exact structure:\n"
        "{\n"
        '  "overview": "...",\n'
        '  "keyPatterns": "...",\n'
        '  "possibleReasons": "...",\n'
        '  "prediction": "...",\n'
        '  "limitations": "..."\n'
        "}"
    )


def _parse_table_response(text: str) -> TableAnalysisResponse:
    cleaned = _strip_json_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "overview": cleaned,
            "keyPatterns": "AI returned free-form text, so no separate key-pattern section was available.",
            "possibleReasons": "Review the overview for possible reasons.",
            "prediction": "Review the overview for interpretation.",
            "limitations": "The response was not structured JSON.",
        }
    return {
        "overview": str(data.get("overview", "")),
        "keyPatterns": str(data.get("keyPatterns", "")),
        "possibleReasons": str(data.get("possibleReasons", "")),
        "prediction": str(data.get("prediction", "")),
        "limitations": str(data.get("limitations", "")),
    }


def _mock_table_analysis(context: TableAnalysisContext) -> TableAnalysisResponse:
    return {
        "overview": f"Bảng `{context.get('tableName', 'selected table')}` hiển thị một phần dữ liệu đang được chọn trong dashboard.",
        "keyPatterns": "Có thể xem các cột số lượng, tỷ lệ hoặc giá trị trung bình để nhận ra nhóm nào nổi bật hơn.",
        "possibleReasons": "Kết quả có thể liên quan đến bộ lọc hiện tại, phân khúc khách, lead time, deposit type hoặc mùa đặt phòng.",
        "prediction": "Nếu một nhóm có cancellation rate cao hơn trung bình, nhóm đó có thể cần được theo dõi kỹ hơn trong vận hành.",
        "limitations": "Đây là gợi ý nội bộ vì chưa gọi được AI API hoặc dữ liệu bảng còn thiếu ngữ cảnh.",
    }


def analyzeTableWithAI(context: TableAnalysisContext, root: Path | None = None) -> TableAnalysisResponse:
    # TODO: If this project is later split into frontend/backend, move this call behind a backend endpoint.
    if root is None:
        root = Path.cwd()
    load_env_file(root / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if not api_key:
        return _mock_table_analysis(context)

    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You explain dashboard tables in simple language. Stay grounded in the supplied table data. "
                        "Do not invent facts or claim causality."
                    )
                }
            ]
        },
        "contents": [{"parts": [{"text": _build_table_prompt(context)}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "maxOutputTokens": 500,
            "responseMimeType": "application/json",
        },
    }

    errors = []
    models_to_try = [model]
    for fallback_model in [DEFAULT_GEMINI_MODEL, *FALLBACK_GEMINI_MODELS]:
        if fallback_model not in models_to_try:
            models_to_try.append(fallback_model)

    for selected_model in models_to_try:
        try:
            response = requests.post(
                GEMINI_ENDPOINT.format(model=selected_model),
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=25,
            )
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return _parse_table_response(text)
        except Exception as exc:
            errors.append(f"{selected_model}: {exc}")

    fallback = _mock_table_analysis(context)
    fallback["limitations"] = fallback.get("limitations", "") + " API error: " + " | ".join(errors)
    return fallback


def explain_dashboard(
    context: DashboardExplanationContext,
    root: Path,
) -> DashboardExplanationResponse:
    load_env_file(root / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if not api_key:
        # TODO: Replace this fallback with a real backend endpoint if the app is split into frontend/backend services.
        return mock_dashboard_explanation(context)

    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are a careful dashboard analyst. Explain only what is grounded in the provided dashboard context. "
                        "Use cautious, non-causal language unless causality is explicitly stated."
                    )
                }
            ]
        },
        "contents": [{"parts": [{"text": _build_dashboard_prompt(context)}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "maxOutputTokens": 650,
            "responseMimeType": "application/json",
        },
    }

    errors = []
    models_to_try = [model]
    for fallback_model in [DEFAULT_GEMINI_MODEL, *FALLBACK_GEMINI_MODELS]:
        if fallback_model not in models_to_try:
            models_to_try.append(fallback_model)

    for selected_model in models_to_try:
        try:
            response = requests.post(
                GEMINI_ENDPOINT.format(model=selected_model),
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=25,
            )
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return _parse_response(text)
        except Exception as exc:
            errors.append(f"{selected_model}: {exc}")

    raise DashboardExplanationError(" | ".join(errors))


def explain_with_ai(selection: dict[str, Any], stats: dict[str, Any], root: Path) -> str:
    load_env_file(root / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if not api_key:
        return fallback_explanation(selection, stats)

    prompt = {
        "selection": selection,
        "supporting_data": stats,
        "task": (
            "Giai thich bang tieng Viet vi sao phan khuc/diem du lieu duoc chon co cancellation rate "
            "nhu vay. Dua vao cac bang so lieu lien quan, noi ro yeu to nao lam tang/giam risk, "
            "anh huong cua chi so nay den van hanh/kinh doanh, va canh bao day la tuong quan/du bao "
            "chu khong phai ket luan nhan qua tuyet doi. Tra ve 4-6 bullet ngan, co so lieu."
        ),
    }

    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "Ban la data analyst cho khach san. Giai thich dashboard cancellation prediction "
                        "ro rang, thuc dung, khong phong dai ket qua."
                    )
                }
            ]
        },
        "contents": [{"parts": [{"text": json.dumps(prompt, ensure_ascii=False, default=safe_number)}]}],
        "generationConfig": {"temperature": 0.2, "topP": 0.8, "maxOutputTokens": 450},
    }

    errors = []
    models_to_try = [model]
    for fallback_model in [DEFAULT_GEMINI_MODEL, *FALLBACK_GEMINI_MODELS]:
        if fallback_model not in models_to_try:
            models_to_try.append(fallback_model)

    for selected_model in models_to_try:
        try:
            response = requests.post(
                GEMINI_ENDPOINT.format(model=selected_model),
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as exc:
            errors.append(f"{selected_model}: {exc}")

    return fallback_explanation(selection, stats) + "\n\nAI API fallback: " + " | ".join(errors)
