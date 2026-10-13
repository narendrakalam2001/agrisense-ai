"""Weather service: OpenWeatherMap integration + disease-risk advisory logic.

Powers the chatbot's ``weather_advisory`` tool and the Streamlit "Weather
Advisory" screen — turns a 5-day forecast into a per-day fungal disease
risk rating and a recommended spray schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import requests

from src.config import OPENWEATHERMAP_API_KEY, MissingAPIKeyError, configure_logging, require_api_key

logger = configure_logging(__name__)

OPENWEATHERMAP_BASE_URL = "https://api.openweathermap.org/data/2.5"
REQUEST_TIMEOUT_SECONDS = 10

# Disease-risk thresholds — see icar_guidelines.pdf §4 ("Weather and
# Disease Risk") for the reasoning behind these bands.
HIGH_RISK_HUMIDITY_PCT = 80
HIGH_RISK_TEMP_MIN_C = 15
HIGH_RISK_TEMP_MAX_C = 25
MODERATE_RISK_HUMIDITY_PCT = 65


class DiseaseRiskLevel(str, Enum):
    """Fungal disease risk level for a given day's forecast."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WeatherServiceError(Exception):
    """Raised when the weather API cannot be reached or returns bad data."""


@dataclass
class DailyForecast:
    """A single day's weather summary plus derived disease-risk fields."""

    date: str
    temp_min_c: float
    temp_max_c: float
    humidity_pct: float
    rain_expected: bool
    description: str
    disease_risk: DiseaseRiskLevel
    spray_recommended: bool

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "temp_min_c": self.temp_min_c,
            "temp_max_c": self.temp_max_c,
            "humidity_pct": self.humidity_pct,
            "rain_expected": self.rain_expected,
            "description": self.description,
            "disease_risk": self.disease_risk.value,
            "spray_recommended": self.spray_recommended,
        }


def _compute_disease_risk(avg_temp_c: float, humidity_pct: float, rain_expected: bool) -> DiseaseRiskLevel:
    """Classify fungal disease risk from a day's temperature/humidity/rain.

    Args:
        avg_temp_c: Average temperature for the day, Celsius.
        humidity_pct: Average relative humidity, percent.
        rain_expected: Whether rain is forecast for the day.

    Returns:
        A :class:`DiseaseRiskLevel`.
    """
    in_fungal_temp_band = HIGH_RISK_TEMP_MIN_C <= avg_temp_c <= HIGH_RISK_TEMP_MAX_C

    if humidity_pct >= HIGH_RISK_HUMIDITY_PCT and in_fungal_temp_band:
        return DiseaseRiskLevel.HIGH
    if rain_expected and humidity_pct >= MODERATE_RISK_HUMIDITY_PCT:
        return DiseaseRiskLevel.HIGH
    if humidity_pct >= MODERATE_RISK_HUMIDITY_PCT:
        return DiseaseRiskLevel.MEDIUM
    return DiseaseRiskLevel.LOW


def fetch_five_day_forecast(city_name: str, api_key: str | None = None) -> list[DailyForecast]:
    """Fetch and summarize a 5-day / 3-hour forecast for a city.

    Args:
        city_name: City name, e.g. "Nashik,IN". Country code recommended
            to disambiguate common Indian town names.
        api_key: Override for the OpenWeatherMap API key. Defaults to
            ``OPENWEATHERMAP_API_KEY`` from the environment.

    Returns:
        One :class:`DailyForecast` per day (up to 5 days), aggregated from
        the raw 3-hour data points.

    Raises:
        WeatherServiceError: If the API key is missing, the request fails,
            or the city cannot be found.
    """
    try:
        key = require_api_key("OPENWEATHERMAP_API_KEY", api_key or OPENWEATHERMAP_API_KEY)
    except MissingAPIKeyError as exc:
        raise WeatherServiceError(str(exc)) from exc

    try:
        response = requests.get(
            f"{OPENWEATHERMAP_BASE_URL}/forecast",
            params={"q": city_name, "appid": key, "units": "metric"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WeatherServiceError(f"OpenWeatherMap request failed: {exc}") from exc

    payload = response.json()
    if payload.get("cod") not in ("200", 200):
        raise WeatherServiceError(
            f"OpenWeatherMap error for '{city_name}': {payload.get('message', 'unknown error')}"
        )

    daily_buckets: dict[str, list[dict]] = {}
    for entry in payload.get("list", []):
        date_str = entry["dt_txt"].split(" ")[0]
        daily_buckets.setdefault(date_str, []).append(entry)

    forecasts: list[DailyForecast] = []
    for date_str, entries in sorted(daily_buckets.items())[:5]:
        temps = [e["main"]["temp"] for e in entries]
        humidities = [e["main"]["humidity"] for e in entries]
        rain_expected = any("rain" in e for e in entries)
        descriptions = [e["weather"][0]["description"] for e in entries if e.get("weather")]

        avg_temp = sum(temps) / len(temps)
        avg_humidity = sum(humidities) / len(humidities)
        risk = _compute_disease_risk(avg_temp, avg_humidity, rain_expected)

        forecasts.append(
            DailyForecast(
                date=date_str,
                temp_min_c=round(min(temps), 1),
                temp_max_c=round(max(temps), 1),
                humidity_pct=round(avg_humidity, 1),
                rain_expected=rain_expected,
                description=descriptions[0] if descriptions else "",
                disease_risk=risk,
                spray_recommended=(risk != DiseaseRiskLevel.HIGH and not rain_expected),
            )
        )

    logger.info("Fetched %d-day forecast for '%s'.", len(forecasts), city_name)
    return forecasts


def recommend_spray_days(forecasts: list[DailyForecast]) -> list[str]:
    """Pick the best upcoming days to spray from a forecast list.

    A good spray day has no rain expected and disease risk is not already
    HIGH (spraying preventively on a low/medium-risk, dry day is more
    effective than reacting on a high-risk day).

    Args:
        forecasts: Output of :func:`fetch_five_day_forecast`.

    Returns:
        List of ISO date strings recommended for spraying, best first.
    """
    good_days = [f.date for f in forecasts if f.spray_recommended]
    return good_days


def build_weather_alert_message(forecasts: list[DailyForecast], language: str = "hinglish") -> str:
    """Build a short farmer-facing alert summarizing the forecast risk.

    Args:
        forecasts: Output of :func:`fetch_five_day_forecast`.
        language: ``"hinglish"`` or ``"english"`` — controls message phrasing.

    Returns:
        A one-to-two sentence alert message, e.g.
        "Kal baarish hai — fungal disease risk HIGH. Preventive spray recommended."
    """
    if not forecasts:
        return "Weather data unavailable." if language == "english" else "Mausam data uplabdh nahi hai."

    tomorrow = forecasts[1] if len(forecasts) > 1 else forecasts[0]

    if language == "english":
        if tomorrow.rain_expected and tomorrow.disease_risk == DiseaseRiskLevel.HIGH:
            return "Rain expected tomorrow — fungal disease risk HIGH. Preventive spray recommended today."
        if tomorrow.rain_expected:
            return "Rain expected tomorrow — avoid spraying, it will wash off."
        if tomorrow.disease_risk == DiseaseRiskLevel.HIGH:
            return "High humidity expected tomorrow — fungal disease risk HIGH. Consider a preventive spray."
        return "Weather looks favorable tomorrow — good conditions for fieldwork."

    # Hinglish (matches the sample conversations in the project brief)
    if tomorrow.rain_expected and tomorrow.disease_risk == DiseaseRiskLevel.HIGH:
        return "Kal baarish hai — fungal disease risk HIGH. Preventive spray recommended."
    if tomorrow.rain_expected:
        return "Kal baarish ho sakti hai — spray na karein, dhul jayega."
    if tomorrow.disease_risk == DiseaseRiskLevel.HIGH:
        return "Kal nami zyada rahegi — fungal disease risk HIGH hai. Preventive spray kar sakte hain."
    return "Kal mausam theek rahega — kheti ke kaam ke liye accha samay hai."
