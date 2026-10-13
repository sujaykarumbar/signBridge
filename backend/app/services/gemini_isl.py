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
            # Sample exactly 3 key frames (start, apex/middle, end) for fast 1-2s inference
            n = len(jpegs)
            if n >= 3:
                sampled = [jpegs[0], jpegs[n // 2], jpegs[-1]]
            else:
                sampled = jpegs

            parts = [
                {
                    "text": (
                        "You are a master Indian Sign Language (ISL) interpreter. "
                        "Carefully examine the signer's hand shapes, hand positions, palm orientation, and movement across these consecutive frames. "
                        "Accurately translate their Indian Sign Language into natural English. "
                        "Recognize common ISL signs such as: Namaste, Hello, Good morning, How are you, I am fine, Thank you, Please, Help, Time, Name, Father, Mother, Brother, Sister, Friend, House, Water, Food, Work, School, Where, What, Happy, Sad, Yes, No, etc. "
                        "Return a single JSON object: "
                        "{\"sentence\": \"<natural English sentence of what the person is communicating>\", "
                        "\"prediction\": \"<main ISL sign or gloss in UPPERCASE, e.g. HELLO, FINE, HELP, TIME, NAME>\", "
                        "\"confidence\": <float between 0.70 and 0.99>}. "
                        "If the person is resting, transition hands, or not performing any recognisable sign, return: {\"sentence\": \"\", \"prediction\": \"\", \"confidence\": 0.0}."
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
                    "temperature": 0.1
                }
            }

            headers = {"Content-Type": "application/json"}
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)

            # Retry once on temporary 503 spike or 429 rate limit
            if resp.status_code in (503, 429):
                time.sleep(0.5)
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
