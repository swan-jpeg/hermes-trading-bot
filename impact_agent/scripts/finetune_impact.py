#!/usr/bin/env python3
"""Startpunt voor het fine-tunen van een impact-agent-model.

De gratis OpenRouter-modellen werken direct zonder training. Dit script is
voor als je een EIGEN impact-model wilt trainen op jouw historische
gebeurtenissen (events.jsonl).

Data-formaat (data/events.jsonl):
    {"event": "Trump prijst Jensen Huang", "source": "speech",
     "impacted": [{"entity": "NVDA", "asset_class": "stock", "sentiment": 0.8}]}

Gebruik:
    uv run python impact_agent/scripts/finetune_impact.py --data data/events.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def load_events(path: pathlib.Path) -> list[dict]:
    """Laad events.jsonl (één JSON-object per regel)."""
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def build_training_pairs(events: list[dict]) -> list[dict]:
    """Zet events om naar (prompt, completion) paren voor fine-tuning."""
    pairs = []
    for ev in events:
        impacted = ", ".join(
            f'{i["entity"]} ({i["asset_class"]}, sentiment {i["sentiment"]})'
            for i in ev.get("impacted", [])
        )
        prompt = (
            f"Gebeurtenis ({ev.get('source', 'news')}): {ev['event']}\n"
            "Welke instrumenten worden geraakt?"
        )
        completion = impacted or "geen"
        pairs.append({"prompt": prompt, "completion": completion})
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune impact-agent data prep")
    ap.add_argument("--data", default=str(DATA_DIR / "events.jsonl"),
                    help="pad naar events.jsonl")
    ap.add_argument("--out", default=str(DATA_DIR / "training_pairs.jsonl"),
                    help="output-pad voor training-paren")
    args = ap.parse_args()

    path = pathlib.Path(args.data)
    if not path.exists():
        print(f"⚠ Geen data gevonden op {path}. Maak eerst data/events.jsonl.")
        print("Formaat: {\"event\": \"...\", \"source\": \"speech\", "
              "\"impacted\": [{\"entity\": \"NVDA\", \"asset_class\": \"stock\", "
              "\"sentiment\": 0.8}]}")
        return 1

    events = load_events(path)
    pairs = build_training_pairs(events)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} training-paren geschreven naar {out}")
    print("Gebruik deze paren met een fine-tuning-API (OpenRouter / Nous) op "
          "een open model (bv. Llama-3.1-8B).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
