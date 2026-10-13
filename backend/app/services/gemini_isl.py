import os
import json
import base64
import logging
import time
import requests

logger = logging.getLogger(__name__)


class GeminiISLService:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        self.timeout = int(os.getenv("GEMINI_TIMEOUT", "25"))
        self._refresh_endpoint()

    def _refresh_endpoint(self):
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            if self.api_key
            else ""
        )

    def is_available(self) -> bool:
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
            self._refresh_endpoint()
        return bool(self.api_key)

    def predict_from_frames(self, jpegs: list) -> dict:
        """
        Send sampled JPEG frames to Gemini 1.5 Flash Vision for zero-shot ISL translation.
        """
        if not self.is_available():
            return {"error": "GEMINI_API_KEY not configured"}

        if not jpegs:
            return {"error": "No frames provided"}

        t0 = time.time()
        try:
            # Sample up to 6 frames evenly across the 16-frame window
            n = len(jpegs)
            if n > 6:
                indices = [int(i * (n - 1) / 5) for i in range(6)]
                sampled = [jpegs[i] for i in indices]
            else:
                sampled = jpegs

            parts = [
                {
                    "text": (
                        "You are an expert Indian Sign Language (ISL) interpreter. "
                        "Analyze these consecutive frames of a signer performing Indian Sign Language. "
                        "Identify the gesture, word, or complete sentence they are communicating. "
                        "Return JSON with: "
                        "{\"sentence\": \"<complete English translation of what the person is communicating>\", "
                        "\"prediction\": \"<main sign or gloss in uppercase, e.g. HELLO, TIME, HELP, FREE>\", "
                        "\"confidence\": <float between 0.60 and 0.99>}. "
                        "If no sign or gesture is made (resting), return {\"sentence\": \"\", \"prediction\": \"\", \"confidence\": 0.0}."
                    )
                }
            ]

            for frame_bytes in sampled:
                b64_str = base64.b64encode(frame_bytes).decode("utf-8")
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_str
                    }
                })

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": 0.2
                }
            }

            headers = {"Content-Type": "application/json"}
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)

            if resp.status_code != 200:
                logger.warning("Gemini Vision request failed: %s %s", resp.status_code, resp.text)
                return {"error": f"Gemini API error {resp.status_code}"}

            data = resp.json()
            text_resp = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "{}")
            )

            result = json.loads(text_resp)
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            if not isinstance(result, dict):
                result = {}

            pred = str(result.get("prediction", "")).strip().upper()
            sentence = str(result.get("sentence", "")).strip()
            conf = float(result.get("confidence", 0.85 if sentence else 0.0))

            if not pred and sentence:
                words = [w.strip(".,!?") for w in sentence.split() if len(w) > 2]
                pred = words[0].upper() if words else "SIGN"

            latency = round((time.time() - t0) * 1000, 2)

            return {
                "prediction": pred,
                "sentence": sentence,
                "confidence": conf,
                "engine": "gemini-vision",
                "inference_time_ms": latency
            }

        except Exception as e:
            logger.exception("Gemini ISL prediction failed")
            return {"error": str(e)}
