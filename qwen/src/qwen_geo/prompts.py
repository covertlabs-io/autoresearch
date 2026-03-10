from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPTS = {
    "strict_json": (
        "You are a visual geolocation assistant. Infer the most likely real-world location of the photo "
        "using visual cues only. Reply with JSON only and no markdown."
    ),
    "geohash_first": (
        "You are a precise geolocation assistant. Estimate the location from the image and emit JSON only. "
        "Prefer a correct country and geohash cell over overly specific coordinates."
    ),
    "coords_first": (
        "You geolocate outdoor photos. Predict the most likely latitude and longitude, then provide a coarse "
        "country and geohash summary in JSON only."
    ),
}


USER_PROMPTS = {
    "strict_json": (
        "Estimate where this image was taken. Return JSON with keys country_code, geohash, latitude, longitude."
    ),
    "geohash_first": (
        "Predict the photo location. Return JSON with keys geohash, country_code, latitude, longitude."
    ),
    "coords_first": (
        "Predict the photo location. Return JSON with keys latitude, longitude, country_code, geohash."
    ),
}


def render_target(record: dict[str, Any], label_style: str, geohash_precision: int) -> str:
    geohash = record.get(f"geohash_{geohash_precision}") or record.get("geohash")
    lat = round(float(record["latitude"]), 4)
    lon = round(float(record["longitude"]), 4)
    country_code = (record.get("country_code") or "UNK").upper()

    if label_style == "cell_json":
        payload = {"country_code": country_code, "geohash": geohash}
    elif label_style == "coords_json":
        payload = {"latitude": lat, "longitude": lon}
    else:
        payload = {
            "country_code": country_code,
            "geohash": geohash,
            "latitude": lat,
            "longitude": lon,
        }
    return json.dumps(payload, separators=(",", ":"))


def build_training_messages(
    record: dict[str, Any],
    *,
    prompt_style: str,
    label_style: str,
    geohash_precision: int,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPTS[prompt_style]}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": record["image"]},
                {"type": "text", "text": USER_PROMPTS[prompt_style]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": render_target(record, label_style, geohash_precision)}],
        },
    ]


def build_inference_messages(record: dict[str, Any], *, prompt_style: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPTS[prompt_style]}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": record["image"]},
                {"type": "text", "text": USER_PROMPTS[prompt_style]},
            ],
        },
    ]
