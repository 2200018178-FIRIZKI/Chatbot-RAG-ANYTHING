import requests, json

API_KEY = "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"

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    data=json.dumps({
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Test API"}]
    })
)

print(response.json())
