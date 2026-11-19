#!/usr/bin/env python3
"""
Live Verification of all 5 Configured Generator Adapters in CSHSM.
"""

import os
import sys
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.generation.adapters import CloudflareWorkersAIAdapter, GroqAdapter

load_dotenv()

sample_hate = "Hudugir ge driving baralla, road alli accident guarantee."
category = "Gender"

adapters = [
    ("Cloudflare Slot 1", CloudflareWorkersAIAdapter(model_id="@cf/google/gemma-4-26b-a4b-it")),
    ("Cloudflare Slot 2", CloudflareWorkersAIAdapter(model_id="@cf/zai-org/glm-4.7-flash")),
    ("Groq Slot 3", GroqAdapter(model_id="openai/gpt-oss-20b")),
    ("Groq Slot 4", GroqAdapter(model_id="qwen/qwen3.8-27b")),
    ("Groq Slot 5", GroqAdapter(model_id="openai/gpt-oss-120b")),
]

print("=" * 85)
print("CSHSM MILESTONE G3: LIVE ADAPTER EXECUTION VERIFICATION")
print("=" * 85)
print(f"Test Input Prompt: \"{sample_hate}\"")
print(f"Category: {category}\n")

results = []

for label, adapter in adapters:
    print(f"Executing: {label} [{adapter.model_id}]...")
    t0 = time.perf_counter()
    out = adapter.generate(prompt=sample_hate, category=category)
    lat_ms = out.latency_ms

    usable = bool(out.raw_text and len(out.raw_text.strip()) > 15)
    print(f"  Status: {out.status} | Latency: {lat_ms:.2f} ms | Usable: {usable}")
    if out.error_message:
        print(f"  Error: {out.error_message}")
    if out.raw_text:
        preview = out.raw_text.strip().replace("\n", " ")
        if len(preview) > 130:
            preview = preview[:130] + "..."
        print(f"  Output: \"{preview}\"")
    print("-" * 85)

    results.append({
        "label": label,
        "provider": adapter.provider,
        "model_id": adapter.model_id,
        "status": out.status,
        "latency_ms": lat_ms,
        "usable": usable,
        "error": out.error_message,
        "text": out.raw_text
    })

print("\n" + "=" * 85)
print("SUMMARY TABLE")
print("=" * 85)
print(f"{'Slot / Provider':<22} | {'Model ID':<32} | {'Status':<10} | {'Latency':<10} | {'Usable':<8}")
print("-" * 85)
for r in results:
    lat_str = f"{r['latency_ms']:.1f} ms"
    print(f"{r['label']:<22} | {r['model_id']:<32} | {r['status']:<10} | {lat_str:<10} | {str(r['usable']):<8}")
