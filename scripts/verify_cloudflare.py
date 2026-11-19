#!/usr/bin/env python3
"""
CSHSM Manual Verification Script: Cloudflare Workers AI.

Directly tests the Cloudflare Workers AI endpoint for counter-speech generation.
Only executes when invoked directly as a standalone script.
Never executes network requests upon import.
"""

import os
import requests
from dotenv import load_dotenv


def main() -> None:
    # Load credentials from the project's .env file
    load_dotenv()

    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    model = "@cf/google/gemma-4-26b-a4b-it"

    if not account_id or not api_token:
        raise SystemExit("ERROR: Cloudflare credentials not loaded.")

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{model}"
    )

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a counter-speech generation model for "
                    "Kannada-English social media conversations.\n"
                    "Write in natural conversational Kannada using "
                    "English letters (Romanized Kannada).\n"
                    "Use approximately 70-80% Kannada vocabulary and "
                    "20-30% commonly used English words.\n"
                    "Never use Kannada script.\n"
                    "Avoid meaningless words, fabricated Kannada, "
                    "literal translations and repetitive phrases.\n"
                    "Use simple Kannada commonly spoken by people "
                    "in Karnataka.\n"
                    "Respond respectfully and empathetically. "
                    "Challenge hateful generalizations without "
                    "insulting anyone.\n"
                    "Generate only 2-4 meaningful sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Hateful comment: 'Hengasaru yavaglu weak irtare. "
                    "Avrige leadership mattu important decisions "
                    "maadoke ability illa.'\n"
                    "Generate a natural Romanized Kannada-English "
                    "counter-speech response."
                ),
            },
        ],
        "max_completion_tokens": 150,
        "chat_template_kwargs": {
            "enable_thinking": False
        },
        "temperature": 0.7,
    }

    print("Sending one request to Cloudflare Workers AI...")
    print("Model:", model)

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60,
        )

        print("\nHTTP Status:", response.status_code)

        try:
            result = response.json()
        except ValueError:
            print("Response was not valid JSON.")
            print(response.text[:2000])
            raise SystemExit(1)

        if response.ok:
            print("\nAPI Response:")
            print(result)
            print("\nCloudflare Workers AI API test completed.")
        else:
            print("\nAPI Error:")
            print(result)

    except requests.RequestException as error:
        print("\nRequest failed:", str(error))


if __name__ == "__main__":
    main()
