#!/usr/bin/env python3
"""
CSHSM Manual Verification Script: Groq Cloud API.

Directly tests the Groq OpenAI-compatible Chat Completions endpoint for counter-speech generation.
Only executes when invoked directly as a standalone script.
Never executes network requests upon import.
"""

import os
import requests
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    model = "qwen/qwen3.8-27b"

    if not api_key:
        raise SystemExit("ERROR: Groq API key not configured.")

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Generate respectful counter-speech against "
                    "hateful social media comments. Challenge "
                    "harmful generalizations calmly. Do not insult "
                    "or attack the speaker. Output only the final "
                    "counter-speech, without analysis or reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Hateful comment: People from that community "
                    "are dishonest and nobody should trust them. "
                    "Generate a short, respectful counter-speech "
                    "in 2-3 sentences."
                ),
            },
        ],
        "max_completion_tokens": 200,
        "reasoning_effort": "low",
        "temperature": 0.7,
    }

    print("Sending one request to Groq...")
    print("Model:", model)

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60,
        )

        print("\nHTTP Status:", response.status_code)

        data = response.json()
        print("\nAPI Response:")
        print(data)

        if response.ok:
            choices = data.get("choices", [])
            if choices:
                print("\nGenerated Counter-Speech:")
                print(choices[0]["message"].get("content"))

            print("\nToken Usage:")
            print(data.get("usage", {}))
        else:
            print("\nRequest failed. Do not retry yet.")

    except requests.RequestException as error:
        print("Request error:", error)


if __name__ == "__main__":
    main()
