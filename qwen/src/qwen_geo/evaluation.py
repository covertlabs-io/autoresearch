from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from qwen_geo.config import EvaluationConfig, dump_json
from qwen_geo.geo import almost_same_cell, decode_geohash, haversine_km
from qwen_geo.prompts import build_inference_messages


def extract_json_blob(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_prediction(text: str) -> dict[str, Any]:
    blob = extract_json_blob(text)
    if not blob:
        return {}
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return {}

    parsed: dict[str, Any] = {}
    if isinstance(payload, dict):
        if "coordinates" in payload and isinstance(payload["coordinates"], dict):
            payload = {**payload, **payload["coordinates"]}
        if "lat" in payload and "latitude" not in payload:
            payload["latitude"] = payload["lat"]
        if "lon" in payload and "longitude" not in payload:
            payload["longitude"] = payload["lon"]

        for key in ("latitude", "longitude"):
            if key in payload:
                try:
                    parsed[key] = float(payload[key])
                except (TypeError, ValueError):
                    pass
        if payload.get("country_code"):
            parsed["country_code"] = str(payload["country_code"]).upper()[:2]
        if payload.get("geohash"):
            parsed["geohash"] = str(payload["geohash"]).lower()
    return parsed


def prediction_to_point(prediction: dict[str, Any]) -> tuple[float, float] | None:
    if "latitude" in prediction and "longitude" in prediction:
        return float(prediction["latitude"]), float(prediction["longitude"])
    if prediction.get("geohash"):
        center = decode_geohash(prediction["geohash"]).center
        return center.latitude, center.longitude
    return None


def compute_metrics(rows: list[dict[str, Any]], thresholds_km: list[int]) -> dict[str, Any]:
    distances = [row["distance_km"] for row in rows if row["distance_km"] is not None]
    parse_rate = sum(1 for row in rows if row["parsed_ok"]) / max(len(rows), 1)
    country_accuracy = (
        sum(1 for row in rows if row["country_match"] is True) / max(len(rows), 1)
        if rows
        else 0.0
    )
    geohash3_accuracy = sum(1 for row in rows if row["geohash3_match"]) / max(len(rows), 1)
    geohash4_accuracy = sum(1 for row in rows if row["geohash4_match"]) / max(len(rows), 1)

    metrics = {
        "examples": len(rows),
        "parse_rate": parse_rate,
        "country_accuracy": country_accuracy,
        "geohash3_accuracy": geohash3_accuracy,
        "geohash4_accuracy": geohash4_accuracy,
        "median_distance_km": float(sorted(distances)[len(distances) // 2]) if distances else math.inf,
        "mean_distance_km": float(sum(distances) / len(distances)) if distances else math.inf,
    }
    for threshold in thresholds_km:
        within = sum(1 for row in rows if row["distance_km"] is not None and row["distance_km"] <= threshold)
        metrics[f"acc_{threshold}km"] = within / max(len(rows), 1)
    metrics["judge_score"] = judge_score(metrics)
    return metrics


def judge_score(metrics: dict[str, Any]) -> float:
    median_distance = metrics.get("median_distance_km", math.inf)
    distance_term = 1.0 / (1.0 + math.log10(max(median_distance, 1.0)))
    return (
        0.30 * metrics.get("acc_25km", 0.0)
        + 0.20 * metrics.get("acc_200km", 0.0)
        + 0.15 * metrics.get("acc_750km", 0.0)
        + 0.15 * metrics.get("country_accuracy", 0.0)
        + 0.10 * metrics.get("parse_rate", 0.0)
        + 0.10 * distance_term
    )


def _move_inputs_to_device(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _prepare_generation_inputs(processor: Any, messages: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return _move_inputs_to_device(inputs, device)
    except TypeError:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = []
        for message in messages:
            for content in message.get("content", []):
                if content.get("type") == "image":
                    images.append(content["image"])
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
        return _move_inputs_to_device(inputs, device)


def evaluate_model(
    model: Any,
    processor: Any,
    eval_dataset: Any,
    evaluation_config: EvaluationConfig,
    *,
    prompt_style: str,
    run_dir: Path,
) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []
    predictions_path = run_dir / "predictions.jsonl"

    with predictions_path.open("w", encoding="utf-8") as sink:
        for index in range(min(len(eval_dataset), evaluation_config.eval_examples)):
            record = eval_dataset[index]
            messages = build_inference_messages(record, prompt_style=prompt_style)
            inputs = _prepare_generation_inputs(processor, messages, device)
            generation_kwargs = {
                "max_new_tokens": evaluation_config.max_new_tokens,
                "do_sample": evaluation_config.temperature > 0,
                "use_cache": True,
            }
            if generation_kwargs["do_sample"]:
                generation_kwargs["temperature"] = evaluation_config.temperature
                generation_kwargs["top_p"] = evaluation_config.top_p

            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation_kwargs)

            prompt_length = inputs["input_ids"].shape[-1]
            generated = output_ids[:, prompt_length:]
            text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            parsed = parse_prediction(text)
            point = prediction_to_point(parsed)
            distance = None
            if point is not None:
                distance = haversine_km(record["latitude"], record["longitude"], point[0], point[1])

            row = {
                "id": record["id"],
                "prediction_text": text,
                "parsed_ok": bool(parsed),
                "country_match": parsed.get("country_code") == record.get("country_code") if parsed else False,
                "geohash3_match": almost_same_cell(parsed.get("geohash"), record.get("geohash_3"), 3),
                "geohash4_match": almost_same_cell(parsed.get("geohash"), record.get("geohash_4"), 4),
                "distance_km": distance,
                "target_country_code": record.get("country_code"),
                "target_geohash_4": record.get("geohash_4"),
                "target_latitude": record["latitude"],
                "target_longitude": record["longitude"],
            }
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)

    metrics = compute_metrics(rows, evaluation_config.thresholds_km)
    dump_json(run_dir / "metrics.json", metrics)
    return metrics
