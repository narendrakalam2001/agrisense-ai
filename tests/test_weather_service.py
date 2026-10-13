"""Tests for src/weather_service.py. All HTTP calls are mocked — no network access."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.weather_service import (
    DailyForecast,
    DiseaseRiskLevel,
    WeatherServiceError,
    _compute_disease_risk,
    build_weather_alert_message,
    fetch_five_day_forecast,
    recommend_spray_days,
)


def test_compute_disease_risk_high_humidity_favorable_temp():
    risk = _compute_disease_risk(avg_temp_c=20.0, humidity_pct=85.0, rain_expected=False)
    assert risk == DiseaseRiskLevel.HIGH


def test_compute_disease_risk_low_when_dry_and_hot():
    risk = _compute_disease_risk(avg_temp_c=35.0, humidity_pct=30.0, rain_expected=False)
    assert risk == DiseaseRiskLevel.LOW


def test_compute_disease_risk_high_with_rain_and_moderate_humidity():
    risk = _compute_disease_risk(avg_temp_c=28.0, humidity_pct=70.0, rain_expected=True)
    assert risk == DiseaseRiskLevel.HIGH


def test_fetch_five_day_forecast_requires_api_key(monkeypatch):
    monkeypatch.setattr("src.weather_service.OPENWEATHERMAP_API_KEY", None)
    with pytest.raises(WeatherServiceError):
        fetch_five_day_forecast("Pune,IN", api_key=None)


@patch("src.weather_service.requests.get")
def test_fetch_five_day_forecast_parses_response(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "cod": "200",
        "list": [
            {
                "dt_txt": "2026-09-20 12:00:00",
                "main": {"temp": 22.0, "humidity": 88},
                "weather": [{"description": "light rain"}],
                "rain": {"3h": 1.2},
            },
            {
                "dt_txt": "2026-09-20 15:00:00",
                "main": {"temp": 24.0, "humidity": 90},
                "weather": [{"description": "moderate rain"}],
                "rain": {"3h": 2.0},
            },
        ],
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    forecasts = fetch_five_day_forecast("Pune,IN", api_key="fake-key")
    assert len(forecasts) == 1
    assert forecasts[0].rain_expected is True
    assert forecasts[0].disease_risk == DiseaseRiskLevel.HIGH


def test_recommend_spray_days_filters_correctly():
    forecasts = [
        DailyForecast("2026-09-20", 18, 24, 60, False, "clear", DiseaseRiskLevel.LOW, True),
        DailyForecast("2026-09-21", 18, 24, 90, True, "rain", DiseaseRiskLevel.HIGH, False),
    ]
    good_days = recommend_spray_days(forecasts)
    assert good_days == ["2026-09-20"]


def test_build_weather_alert_message_english_rain_high_risk():
    forecasts = [
        DailyForecast("2026-09-20", 18, 24, 60, False, "clear", DiseaseRiskLevel.LOW, True),
        DailyForecast("2026-09-21", 18, 22, 90, True, "rain", DiseaseRiskLevel.HIGH, False),
    ]
    message = build_weather_alert_message(forecasts, language="english")
    assert "HIGH" in message


def test_build_weather_alert_message_empty_forecast():
    message = build_weather_alert_message([], language="english")
    assert "unavailable" in message.lower()
