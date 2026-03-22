"""
vision.py — Image description via Ollama (LLaVA model).

Ollama must be running locally:
  ollama serve
  ollama pull llava        # or llava:13b for better quality
"""

import base64
import json
import logging
import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL  = "http://localhost:11434/api/generate"
VISION_MODEL = "llava"          # swap to "llava:13b" / "moondream" as needed

SYSTEM_PROMPT = """You are a concise image analyst.
Respond ONLY with valid JSON — no markdown, no prose outside JSON:
{
  "caption": "<one clear sentence describing the image>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"]
}"""

USER_PROMPT = "Describe this image. Return JSON only."


def describe_image(image_bytes: bytes) -> dict:
    """
    Send image bytes to Ollama LLaVA and return {"caption": ..., "tags": [...]}.
    Raises httpx.HTTPError on network problems.
    """
    b64 = base64.standard_b64encode(image_bytes).decode()

    payload = {
        "model":  VISION_MODEL,
        "prompt": USER_PROMPT,
        "system": SYSTEM_PROMPT,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    logger.info("Calling Ollama (%s) …", VISION_MODEL)
    with httpx.Client(timeout=120) as client:
        resp = client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()

    raw = resp.json().get("response", "").strip()
    # Strip accidental markdown fences
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLaVA returned non-JSON: %s", raw[:200])
        # Graceful fallback — return raw text as caption
        result = {"caption": raw, "tags": []}

    # Guarantee exactly 3 tags
    tags = result.get("tags", [])
    result["tags"] = (tags + ["", "", ""])[:3]
    return result
