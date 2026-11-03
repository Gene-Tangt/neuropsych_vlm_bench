"""List every model available to the API keys of Anthropic, OpenAI and Google, with the metadata each API returns.

    python scripts/model_info_sheet.py                      # all providers, JSON
    python scripts/model_info_sheet.py anthropic google     # only these providers
    python scripts/model_info_sheet.py --jsonl              # one model per line instead

Output: docs/model_info_sheet_extracted_<dd_mm_yy>.json(l). API keys: OPENAI_KEY, ANTHROPIC_KEY, GEMINI_KEY in .env.
Each record has common fields (release date, token limits, reasoning, image input) plus `raw`, the full API entry.
The APIs differ in what they report, so fields a provider does not expose are null:
- Anthropic: creation date, input/output token limits, and capabilities (thinking types, effort levels, image input).
- OpenAI: only id, creation timestamp and owner; release date is the id's date suffix when present.
- Google: input/output token limits, sampling defaults and whether the model thinks; no release date.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

OUTPUT_DIR = Path("docs")
OPENAI_DATED = re.compile(r"-(\d{4}-\d{2}-\d{2})$")
ANTHROPIC_DATED = re.compile(r"-(\d{4})(\d{2})(\d{2})$")


def record(provider: str, model: str, raw: dict, **fields) -> dict:
    """Common fields first, then the full API entry."""
    common = {"display_name": None, "release_date": None, "max_input_tokens": None, "max_output_tokens": None,
              "reasoning": None, "reasoning_levels": None, "image_input": None}
    return {"provider": provider, "model": model, **common, **fields, "raw": raw}


def supported(capability) -> bool | None:
    """`supported` flag of an Anthropic capability entry."""
    return capability.get("supported") if isinstance(capability, dict) else None


def supported_keys(capability) -> list[str] | None:
    """Names of the supported sub-entries of an Anthropic capability, e.g. effort levels or thinking types."""
    if not isinstance(capability, dict):
        return None
    entries = capability.get("types", capability)
    return [name for name, value in entries.items() if isinstance(value, dict) and value.get("supported")]


def anthropic_models() -> list[dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
    rows = []
    for m in client.models.list(limit=1000):
        raw = m.model_dump(mode="json", exclude_none=True)
        caps = raw.get("capabilities", {})
        dated = ANTHROPIC_DATED.search(m.id)
        thinking, effort = caps.get("thinking"), caps.get("effort")
        rows.append(record(
            "anthropic", m.id, raw,
            display_name=m.display_name,
            release_date="-".join(dated.groups()) if dated else m.created_at.strftime("%Y-%m-%d"),
            max_input_tokens=raw.get("max_input_tokens"),
            max_output_tokens=raw.get("max_tokens"),
            reasoning=supported(thinking),
            reasoning_levels={"thinking_types": supported_keys(thinking), "effort": supported_keys(effort)}
            if thinking or effort else None,
            image_input=supported(caps.get("image_input"))))
    return rows


def openai_models() -> list[dict]:
    import openai

    client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY"))
    rows = []
    for m in client.models.list():
        dated = OPENAI_DATED.search(m.id)
        rows.append(record(
            "openai", m.id, m.model_dump(mode="json", exclude_none=True),
            release_date=dated.group(1) if dated
            else datetime.fromtimestamp(m.created, tz=timezone.utc).strftime("%Y-%m-%d")))
    return rows


def google_models() -> list[dict]:
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
    rows = []
    for m in client.models.list(config={"page_size": 1000}):
        rows.append(record(
            "google", m.name.removeprefix("models/"), m.model_dump(mode="json", exclude_none=True),
            display_name=m.display_name,
            max_input_tokens=m.input_token_limit,
            max_output_tokens=m.output_token_limit,
            reasoning=m.thinking))
    return rows


PROVIDERS = {"anthropic": anthropic_models, "openai": openai_models, "google": google_models}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("providers", nargs="*", help=f"any of {', '.join(PROVIDERS)} (default: all)")
    parser.add_argument("--jsonl", action="store_true", help="one model per line instead of a single JSON document")
    args = parser.parse_args()
    if unknown := set(args.providers) - set(PROVIDERS):
        parser.error(f"unknown providers: {', '.join(sorted(unknown))}")
    load_dotenv()

    now = datetime.now(timezone.utc)
    models, summary = [], {}
    for provider in args.providers or PROVIDERS:
        try:
            rows = PROVIDERS[provider]()
            summary[provider] = {"n_models": len(rows), "error": None}
            models += sorted(rows, key=lambda r: (r["release_date"] or "", r["model"]))
        except Exception as e:  # a missing key or SDK for one provider should not lose the others
            summary[provider] = {"n_models": 0, "error": f"{type(e).__name__}: {e}"}
        print(f"{provider}: {summary[provider]['n_models']} models"
              + (f" ({summary[provider]['error']})" if summary[provider]["error"] else ""))

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"model_info_sheet_extracted_{now:%d_%m_%y}.{'jsonl' if args.jsonl else 'json'}"
    with open(path, "w", encoding="utf-8") as f:
        if args.jsonl:
            f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in models)
        else:
            json.dump({"extracted_at": now.isoformat(timespec="seconds"), "providers": summary, "models": models},
                      f, indent=2, ensure_ascii=False)
    print(f"Saved {len(models)} models to {path}")
