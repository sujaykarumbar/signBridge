import os
import json
import base64
import logging
import math
import time
from io import BytesIO
from PIL import Image, ImageDraw
import requests

logger = logging.getLogger(__name__)


def _compress_frame(jpeg_bytes: bytes, max_dim: int = 720, quality: int = 85) -> bytes:
    """Resize and compress JPEG frame preserving hand and finger details."""
    try:
        img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return jpeg_bytes


def create_storyboard_grid(jpegs: list, max_tiles: int = 6) -> bytes:
    """
    Combines sequential motion frames into an ordered chronological storyboard grid
    (left-to-right, top-to-bottom) so Gemini interprets all movement phases simultaneously
    in a single high-definition image without multi-image rate limits or timeouts.
    """
    n = len(jpegs)
    if n == 0:
        return b""
    if n == 1:
        return _compress_frame(jpegs[0], max_dim=448, quality=80)

    count = min(max_tiles, n)
    indices = [int(i * (n - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
    selected_bytes = [jpegs[i] for i in indices]

    tile_w, tile_h = 320, 320
    cols = 3 if count >= 5 else (2 if count >= 2 else 1)
    rows = (count + cols - 1) // cols

    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(20, 24, 38))
    draw = ImageDraw.Draw(grid)

    for idx, raw_bytes in enumerate(selected_bytes):
        c = idx % cols
        r = idx // cols
        x = c * tile_w
        y = r * tile_h

        try:
            tile_img = Image.open(BytesIO(raw_bytes)).convert("RGB")
            tile_img.thumbnail((tile_w - 6, tile_h - 28), Image.Resampling.LANCZOS)
            ox = x + (tile_w - tile_img.width) // 2
            oy = y + 24 + (tile_h - 28 - tile_img.height) // 2
            grid.paste(tile_img, (ox, oy))
        except Exception:
            pass

        # Subtitle banner for chronological ordering
        draw.rectangle([x, y, x + tile_w, y + 22], fill=(35, 45, 75))
        draw.text((x + 8, y + 4), f"Step {idx+1}/{count}", fill=(255, 255, 255))
        draw.rectangle([x, y, x + tile_w, y + tile_h], outline=(60, 80, 130), width=1)

    buf = BytesIO()
    grid.save(buf, format="JPEG", quality=80, optimize=True)
    return buf.getvalue()


class GeminiISLService:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
        self.fallback_models = [
            candidate.strip()
            for candidate in os.getenv(
                "GEMINI_FALLBACK_MODELS",
                "gemini-3.5-flash",
            ).split(",")
            if candidate.strip()
        ]
        self.timeout = int(os.getenv("GEMINI_TIMEOUT", "35"))
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

    def predict_from_video_file(self, video_path: str) -> dict:
        """
        Upload full continuous video to Gemini Files API and analyze the complete
        temporal sequence of Indian Sign Language (ISL) gestures, signs, and sentences.
        """
        if not self.is_available():
            return {"error": "GEMINI_API_KEY not configured"}

        if not os.path.exists(video_path):
            return {"error": "Video file not found"}

        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return {"error": "Empty video file"}

        t0 = time.time()
        file_name = None
        try:
            ext = os.path.splitext(video_path)[1].lower()
            mime_type = "video/mp4"
            if ext == ".mov":
                mime_type = "video/quicktime"
            elif ext == ".webm":
                mime_type = "video/webm"
            elif ext == ".3gp":
                mime_type = "video/3gpp"
            elif ext == ".avi":
                mime_type = "video/x-msvideo"

            # 1. Initiate Resumable Upload
            start_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.api_key}"
            headers = {
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(file_size),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            }
            metadata = {"file": {"display_name": os.path.basename(video_path)}}
            r_start = requests.post(start_url, headers=headers, json=metadata, timeout=20)
            if r_start.status_code != 200:
                logger.warning("Gemini video upload start failed: %s %s", r_start.status_code, r_start.text[:120])
                return {"error": f"Upload start failed: {r_start.status_code}"}

            upload_url = r_start.headers.get("x-goog-upload-url")
            if not upload_url:
                return {"error": "Missing upload URL from Gemini Files API"}

            # 2. Upload video bytes
            with open(video_path, "rb") as f:
                video_bytes = f.read()

            upload_headers = {
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            }
            r_upload = requests.post(upload_url, headers=upload_headers, data=video_bytes, timeout=90)
            if r_upload.status_code != 200:
                logger.warning("Gemini video upload finalize failed: %s %s", r_upload.status_code, r_upload.text[:120])
                return {"error": f"Upload finalize failed: {r_upload.status_code}"}

            file_info = r_upload.json().get("file", {})
            file_name = file_info.get("name")
            file_uri = file_info.get("uri")

            if not file_uri:
                return {"error": "Failed to get file URI from Gemini"}

            # 3. Poll until ACTIVE (up to 30 seconds)
            for _ in range(25):
                state = file_info.get("state")
                if state == "ACTIVE":
                    break
                if state == "FAILED":
                    return {"error": "Gemini video processing failed on server"}
                time.sleep(2)
                r_check = requests.get(
                    f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={self.api_key}",
                    timeout=15,
                )
                if r_check.status_code == 200:
                    file_info = r_check.json()

            prompt = (
                "The uploaded clip has no audio and its lower caption area has been obscured.\n"
                "Interpret ONLY visible physical Indian Sign Language (ISL) hand movements, "
                "handshapes, signing locations, transitions, and non-manual facial grammar. "
                "Never infer signs or meaning from written text, subtitles, lip movements, "
                "or speech. Do not guess a sentence when the hand movements are unclear; "
                "return an empty sentence, no glosses, and confidence 0.0 instead.\n\n"
                "Transcribe only glosses supported by the visible signs, in chronological "
                "order, then translate those signs into a natural English sentence.\n\n"
                "Return JSON schema strictly adhering to:\n"
                "{\n"
                '  "hand_gestures_observed": "Chronological description of physical hand movements",\n'
                '  "detected_glosses": ["GLOSS1", "GLOSS2"],\n'
                '  "translated_sentence": "English translation derived purely from the physical hand signs",\n'
                '  "confidence": 0.0,\n'
                '  "sign_type": "continuous_sentence"\n'
                "}\n"
                "Return confidence from 0.0 to 1.0 based on visual evidence, not a default. "
                "Where 'sign_type' must be one of: 'continuous_sentence', 'isolated_word', "
                "'fingerspelling', or 'none'."
            )

            gen_payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}
                    ]
                }],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": 0.1
                }
            }

            candidate_models = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
            if self.model and self.model not in candidate_models:
                candidate_models.insert(0, self.model)

            resp = None
            last_err = None
            for m in candidate_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={self.api_key}"
                for attempt in range(3):
                    try:
                        logger.info("Calling Gemini (%s) attempt %d/3...", m, attempt + 1)
                        r = requests.post(url, json=gen_payload, headers={"Content-Type": "application/json"}, timeout=85)
                        if r.status_code == 200:
                            resp = r
                            break
                        elif r.status_code in (503, 429):
                            logger.warning("Gemini %s attempt %d returned %d, retrying after backoff...", m, attempt + 1, r.status_code)
                            time.sleep(2.5 * (attempt + 1))
                        else:
                            logger.warning("Gemini %s returned %d: %s", m, r.status_code, r.text[:120])
                            break
                    except Exception as e:
                        logger.warning("Gemini %s attempt %d error: %s", m, attempt + 1, e)
                        last_err = e
                        time.sleep(2.0)
                if resp is not None:
                    break

            if resp is None or resp.status_code != 200:
                return {"error": f"Gemini video translation unavailable: {last_err or (resp.status_code if resp else 'no response')}"}

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
                return {"error": "Gemini returned an invalid translation response"}

            raw_glosses = result.get("detected_glosses", [])
            if isinstance(raw_glosses, str):
                raw_glosses = [g.strip().upper() for g in raw_glosses.split() if g.strip()]
            elif isinstance(raw_glosses, list):
                raw_glosses = [str(g).strip().upper() for g in raw_glosses if str(g).strip()]
            else:
                raw_glosses = []

            translated_sentence = str(result.get("translated_sentence") or result.get("sentence", "")).strip()
            try:
                conf = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
                conf = 0.0
            sign_type = str(result.get("sign_type", "continuous_sentence")).strip().lower()
            if sign_type not in ("continuous_sentence", "isolated_word", "fingerspelling", "none"):
                sign_type = "none"
            hand_actions = str(result.get("hand_gestures_observed") or result.get("physical_hand_actions", "")).strip()

            primary_pred = raw_glosses[0] if raw_glosses else str(result.get("prediction", "")).strip().upper()
            if not translated_sentence or not raw_glosses or not hand_actions:
                return {
                    "error": (
                        "Gemini could not provide a translation supported by visible hand "
                        "gestures and glosses"
                    )
                }

            latency = round((time.time() - t0) * 1000, 2)
            return {
                "detected_glosses": raw_glosses,
                "translated_sentence": translated_sentence,
                "confidence": conf,
                "confidence_source": "unvalidated_model_estimate",
                "requires_review": True,
                "sign_type": sign_type,
                "hand_gestures_observed": hand_actions,
                "prediction": primary_pred,
                "sentence": translated_sentence,
                "engine": "gemini-files-video",
                "inference_time_ms": latency,
            }
        except Exception as e:
            logger.exception("Gemini video file translation failed")
            return {"error": str(e)}
        finally:
            # Clean up uploaded file from Gemini storage
            if file_name and self.api_key:
                try:
                    requests.delete(
                        f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={self.api_key}",
                        timeout=10,
                    )
                except Exception:
                    pass


    def predict_from_frames(self, jpegs: list) -> dict:
        """
        Send chronological storyboard or single keyframe to Gemini Multimodal Vision
        for accurate Indian Sign Language (ISL) sentence, word, and fingerspelling translation.
        """
        if not self.is_available():
            return {"error": "GEMINI_API_KEY not configured"}

        if not jpegs:
            return {"error": "No frames provided"}

        t0 = time.time()
        try:
            selected_count = min(6, len(jpegs))
            frame_indices = [
                round(index * (len(jpegs) - 1) / max(1, selected_count - 1))
                for index in range(selected_count)
            ]
            selected_frames = [jpegs[index] for index in frame_indices]
            instruction_intro = (
                f"You are given {len(selected_frames)} separate video frames in chronological order. "
                "Compare the frames to follow the signer’s hand movements over time; do not treat "
                "them as unrelated static images.\n"
            )

            system_instruction = (
                "You are an expert Indian Sign Language (ISL) linguist and certified interpreter.\n"
                f"{instruction_intro}"
                "Indian Sign Language (ISL) operates with Subject-Object-Verb (SOV) sentence structure, topic-comment "
                "pragmatics, non-manual facial cues, and spatial grammar, differing fundamentally from English Subject-Verb-Object (SVO).\n\n"
                "CRITICAL INTERPRETATION RULES:\n"
                "1. STRICT PHYSICAL GESTURE INTERPRETATION:\n"
                "   - Carefully examine the signer's handshape (open palm, fist, index point, thumb, claw, pinch, etc.).\n"
                "   - Track whether one or both hands are signing, their location (head, chin, chest, neutral space), "
                "     and the movement trajectory across the chronological steps.\n"
                "   - The lower caption area is obscured. Ignore any remaining text, subtitles, logos, or UI overlays.\n"
                "   - There is no audio. Never infer the translation from speech or lip movements.\n"
                "2. ACCURATE TRANSLATION & CLASSIFICATION:\n"
                "   - If a single sign or word is performed (e.g., HELLO, THANK YOU, NAME, PLEASE, HELP, WATER, HOME, YES, NO, etc.), "
                "     output that sign and set sign_type: 'isolated_word'.\n"
                "   - If fingerspelling or letters/numbers are shown (e.g. A-Z, 0-9), set sign_type: 'fingerspelling'.\n"
                "   - If a continuous multi-sign sentence or phrase is signed, transcribe all distinct glosses in chronological order "
                "     and translate into a natural, grammatically correct English sentence (SVO), setting sign_type: 'continuous_sentence'.\n"
                "3. HONESTY & NO HALLUCINATIONS:\n"
                "   - Base your interpretation strictly on the physical gestures shown in the frames.\n"
                "   - DO NOT hallucinate default phrases like 'I am reading a storybook' or 'Congratulations' unless those "
                "     exact signs are clearly performed.\n"
                "   - If hands are at rest, occluded, or unclear, return an empty translated_sentence, "
                "     confidence 0.0, and empty detected_glosses.\n"
                "4. STRICT JSON OUTPUT:\n"
                "   You MUST return a single valid JSON object adhering strictly to this schema:\n"
                "   {\n"
                '     "hand_gestures_observed": "Chronological physical hand movements visible in the frames",\n'
                '     "detected_glosses": ["GLOSS1", "GLOSS2"],\n'
                '     "translated_sentence": "Natural English sentence translation",\n'
                '     "confidence": 0.0,\n'
                '     "sign_type": "continuous_sentence"\n'
                "   }\n"
                "   Confidence must reflect visible evidence, not a default. Where 'sign_type' must be "
                "strictly one of: 'continuous_sentence', 'isolated_word', 'fingerspelling', or 'none'."
            )

            parts = [{"text": system_instruction}]
            for index, jpeg in enumerate(selected_frames, start=1):
                image_bytes = _compress_frame(jpeg, max_dim=720, quality=85)
                parts.append({"text": f"Frame {index} of {len(selected_frames)}"})
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
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

            # Only use fallbacks that the operator has explicitly configured.
            candidate_models = [self.model, *self.fallback_models]
            candidate_models = list(dict.fromkeys(candidate_models))

            resp = None
            last_err = "no response"
            for m in candidate_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={self.api_key}"
                for attempt in range(2):
                    try:
                        r = requests.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=self.timeout,
                        )
                        if r.status_code == 200:
                            resp = r
                            break
                        last_err = self._describe_api_error(r.status_code)
                        if (r.status_code == 429 or 500 <= r.status_code <= 599) and attempt == 0:
                            retry_after = r.headers.get("Retry-After", "")
                            try:
                                delay = min(5.0, max(1.0, float(retry_after)))
                            except (TypeError, ValueError):
                                delay = 1.0
                            logger.warning(
                                "Gemini model %s returned HTTP %s; retrying once in %.1fs",
                                m,
                                r.status_code,
                                delay,
                            )
                            time.sleep(delay)
                            continue
                        logger.warning(
                            "Gemini model %s returned HTTP %s",
                            m,
                            r.status_code,
                        )
                        break
                    except requests.RequestException as e:
                        error_type = type(e).__name__
                        logger.warning("Gemini model %s request failed (%s)", m, error_type)
                        last_err = f"connection error ({error_type})"
                        break
                if resp is not None:
                    break

            if resp is None or resp.status_code != 200:
                attempted = ", ".join(candidate_models)
                return {
                    "error": (
                        f"Gemini API unavailable: {last_err}. "
                        f"Models tried: {attempted}. "
                        "Check the API key, model access, quota, and service status."
                    )
                }

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
                return {"error": "Gemini returned an invalid translation response"}

            # Strict schema field extraction
            raw_glosses = result.get("detected_glosses", [])
            if isinstance(raw_glosses, str):
                raw_glosses = [g.strip().upper() for g in raw_glosses.split() if g.strip()]
            elif isinstance(raw_glosses, list):
                raw_glosses = [str(g).strip().upper() for g in raw_glosses if str(g).strip()]
            else:
                raw_glosses = []

            translated_sentence = str(result.get("translated_sentence") or result.get("sentence", "")).strip()
            try:
                conf = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
                conf = 0.0

            sign_type = str(result.get("sign_type", "none")).strip().lower()
            if sign_type not in ("continuous_sentence", "isolated_word", "fingerspelling", "none"):
                sign_type = "none"
            if not translated_sentence:
                return {"error": "Gemini could not identify a translation from visible signs"}

            primary_pred = raw_glosses[0] if raw_glosses else str(result.get("prediction", "")).strip().upper()
            hand_actions = str(
                result.get("hand_gestures_observed")
                or result.get("physical_hand_actions", "")
            ).strip()
            if not translated_sentence or not raw_glosses or not hand_actions:
                return {
                    "error": (
                        "Gemini could not provide a translation supported by visible hand "
                        "gestures and glosses"
                    )
                }

            latency = round((time.time() - t0) * 1000, 2)

            return {
                "detected_glosses": raw_glosses,
                "translated_sentence": translated_sentence,
                "confidence": conf,
                "confidence_source": "unvalidated_model_estimate",
                "requires_review": True,
                "sign_type": sign_type,
                "hand_gestures_observed": hand_actions,
                # Backwards-compatible fields
                "prediction": primary_pred,
                "sentence": translated_sentence,
                "engine": "gemini-vision-continuous",
                "inference_time_ms": latency,
            }

        except Exception as e:
            logger.exception("Gemini ISL prediction failed")
            return {"error": str(e)}

    @staticmethod
    def _describe_api_error(status_code: int) -> str:
        if status_code in (401, 403):
            return f"HTTP {status_code}: API key or model permission issue"
        if status_code == 404:
            return "HTTP 404: configured model was not found"
        if status_code == 429:
            return "HTTP 429: rate limit or quota exceeded"
        if 500 <= status_code <= 599:
            return f"HTTP {status_code}: Gemini service is temporarily unavailable"
        return f"HTTP {status_code}: Gemini rejected the request"
