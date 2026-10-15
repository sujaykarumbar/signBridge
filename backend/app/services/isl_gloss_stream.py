import time
import logging
from app.services.gemini_grammar_decoder import GeminiGrammarDecoder

logger = logging.getLogger(__name__)


class ISLGlossStreamTracker:
    """
    Stage 3: Cross-Modal Alignment & Gloss Stream Tracker.
    
    Accumulates continuous ISL gloss detections from temporal classifiers (Stage 2),
    filters out consecutive repetitive duplicate detections (debouncing),
    and triggers Stage 4 (Gemini Grammar Reconstruction) when a pause occurs or
    the sequence length reaches a threshold.
    """

    def __init__(
        self,
        dwell_seconds: float = 0.8,
        pause_timeout_seconds: float = 1.0,
        min_confidence: float = 0.40,
        decoder: GeminiGrammarDecoder = None
    ):
        self.dwell_seconds = dwell_seconds
        self.pause_timeout_seconds = pause_timeout_seconds
        self.min_confidence = min_confidence
        self.decoder = decoder or GeminiGrammarDecoder()

        self.gloss_sequence = []
        self.last_gloss = None
        self.last_gloss_time = 0.0
        self.last_activity_time = time.time()

    def get_current_glosses(self) -> list[str]:
        return list(self.gloss_sequence)

    def add_gloss(self, gloss: str, confidence: float = 1.0) -> dict:
        """
        Add a detected sign gloss from Stage 2.
        Filters out low-confidence noise (< min_confidence).
        """
        now = time.time()
        gloss = str(gloss).strip().upper()
        if not gloss or confidence < self.min_confidence:
            return {"accepted": False, "current_glosses": list(self.gloss_sequence), "sentence": None}

        # Check debouncing: if it's the same sign within dwell_seconds, ignore repetition
        if gloss == self.last_gloss and (now - self.last_gloss_time) < self.dwell_seconds:
            self.last_activity_time = now
            return {"accepted": False, "current_glosses": list(self.gloss_sequence), "sentence": None}

        # New distinct gloss detected
        self.gloss_sequence.append(gloss)
        self.last_gloss = gloss
        self.last_gloss_time = now
        self.last_activity_time = now

        logger.info("[Gloss Stream] Added gloss: %s (conf: %.2f) (Total in buffer: %s)", gloss, confidence, self.gloss_sequence)

        return {
            "accepted": True,
            "new_gloss": gloss,
            "current_glosses": list(self.gloss_sequence),
            "sentence": None
        }

    def add_glosses(self, glosses: list[str], confidence: float = 1.0) -> list[dict]:
        """
        Add multiple glosses from continuous sequence segmentation or multimodal inference.
        """
        results = []
        for g in glosses:
            res = self.add_gloss(g, confidence)
            if res.get("accepted"):
                results.append(res)
        return results

    def check_pause_and_reconstruct(self, min_glosses: int = 1) -> dict:
        """
        Check if the signer paused, indicating the end of a sign sequence/sentence.
        If paused and at least `min_glosses` are buffered, reconstruct sentence.
        """
        now = time.time()
        if not self.gloss_sequence:
            return {"reconstructed": False}

        if len(self.gloss_sequence) >= min_glosses and (now - self.last_activity_time) >= self.pause_timeout_seconds:
            glosses_to_translate = list(self.gloss_sequence)
            recon_res = self.decoder.reconstruct_sentence(glosses_to_translate)
            
            # Reset buffer
            self.gloss_sequence = []
            self.last_gloss = None

            return {
                "reconstructed": True,
                "sentence": recon_res.get("sentence", ""),
                "original_glosses": glosses_to_translate,
                "latency_ms": recon_res.get("latency_ms", 0.0)
            }

        return {"reconstructed": False}

    def flush_and_reconstruct(self) -> dict:
        """Force flush current buffer and decode to sentence."""
        if not self.gloss_sequence:
            return {"reconstructed": False, "sentence": "", "original_glosses": []}

        glosses_to_translate = list(self.gloss_sequence)
        recon_res = self.decoder.reconstruct_sentence(glosses_to_translate)
        self.gloss_sequence = []
        self.last_gloss = None

        return {
            "reconstructed": True,
            "sentence": recon_res.get("sentence", ""),
            "original_glosses": glosses_to_translate,
            "latency_ms": recon_res.get("latency_ms", 0.0)
        }

    def clear(self):
        self.gloss_sequence = []
        self.last_gloss = None
        self.last_activity_time = time.time()
