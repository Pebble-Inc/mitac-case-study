"""
Download prompts from HuggingFace datasets for load testing.

Creates a prompts.json file with 20,000+ unique prompts across varying sequence lengths:
- Short (< 50 words): quick prefill, tests decode throughput
- Medium (50-200 words): balanced prefill/decode
- Long (200+ words): heavy prefill, tests chunked prefill

Datasets used:
- OpenAssistant/oasst1: conversational prompts (short-medium)
- HuggingFaceH4/ultrachat_200k: multi-turn chat prompts (medium-long)
- cnn_dailymail: summarization prompts (long)
"""

import json
import os
import sys

from datasets import load_dataset

HF_TOKEN = os.environ.get("HF_TOKEN")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")

TARGET_SHORT = 6000   # < 50 words
TARGET_MEDIUM = 8000  # 50-200 words
TARGET_LONG = 6000    # 200+ words


def classify_length(text: str) -> str:
    word_count = len(text.split())
    if word_count < 50:
        return "short"
    elif word_count < 200:
        return "medium"
    else:
        return "long"


def download_oasst1() -> list[str]:
    """Short-medium conversational prompts from OpenAssistant."""
    print("Downloading OpenAssistant/oasst1...")
    ds = load_dataset("OpenAssistant/oasst1", split="train", token=HF_TOKEN)
    prompts = []
    for row in ds:
        if row["role"] == "prompter" and row["parent_id"] is None:
            text = row["text"].strip()
            if len(text) > 10:
                prompts.append(text)
    print(f"  Got {len(prompts)} prompts from oasst1")
    return prompts


def download_ultrachat() -> list[str]:
    """Medium-long multi-turn chat prompts."""
    print("Downloading HuggingFaceH4/ultrachat_200k...")
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft", token=HF_TOKEN
    )
    prompts = []
    for row in ds:
        messages = row.get("messages", [])
        if messages and messages[0]["role"] == "user":
            text = messages[0]["content"].strip()
            if len(text) > 20:
                prompts.append(text)
        if len(prompts) >= 15000:
            break
    print(f"  Got {len(prompts)} prompts from ultrachat")
    return prompts


def download_cnn_dailymail() -> list[str]:
    """Long summarization prompts from CNN/DailyMail."""
    print("Downloading cnn_dailymail...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="train", token=HF_TOKEN)
    prompts = []
    for row in ds:
        article = row["article"].strip()
        if len(article.split()) > 150:
            prompt = f"Summarize the following article:\n\n{article}"
            prompts.append(prompt)
        if len(prompts) >= 10000:
            break
    print(f"  Got {len(prompts)} prompts from cnn_dailymail")
    return prompts


def build_prompt_set() -> list[dict]:
    oasst = download_oasst1()
    ultrachat = download_ultrachat()
    cnn = download_cnn_dailymail()

    all_prompts = oasst + ultrachat + cnn

    seen = set()
    unique = []
    for p in all_prompts:
        key = p[:200]
        if key not in seen:
            seen.add(key)
            unique.append(p)

    buckets = {"short": [], "medium": [], "long": []}
    for p in unique:
        buckets[classify_length(p)].append(p)

    print(
        f"\nAvailable: short={len(buckets['short'])}, "
        f"medium={len(buckets['medium'])}, long={len(buckets['long'])}"
    )

    selected = []
    for label, target in [
        ("short", TARGET_SHORT),
        ("medium", TARGET_MEDIUM),
        ("long", TARGET_LONG),
    ]:
        pool = buckets[label][:target]
        selected.extend({"prompt": p, "length_class": label} for p in pool)

    print(f"Selected {len(selected)} prompts total")
    return selected


def main():
    if not HF_TOKEN:
        sys.exit("HF_TOKEN is not set. Export a Hugging Face read token before running.")

    prompts = build_prompt_set()

    if len(prompts) < 20000:
        print(f"WARNING: Only got {len(prompts)} prompts, target was 20000+")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(prompts)} prompts to {OUTPUT_FILE}")

    from collections import Counter

    dist = Counter(p["length_class"] for p in prompts)
    for cls in ["short", "medium", "long"]:
        print(f"  {cls}: {dist.get(cls, 0)}")


if __name__ == "__main__":
    main()
