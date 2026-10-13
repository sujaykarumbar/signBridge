import os
import json
import logging
import time
import requests

logger = logging.getLogger(__name__)


class GeminiGrammarDecoder:
    """
    Stage 4: Transformer Grammar Decoder for Indian Sign Language (ISL).
    
    Transforms ISL gloss token sequences (which follow Subject-Object-Verb /
    topic-comment syntax without English tense/auxiliary inflections) into
    natural, grammatically correct spoken English sentences using Google Gemini.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        self.timeout = int(os.getenv("GEMINI_TIMEOUT", "15"))
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

    def reconstruct_sentence(self, glosses: list[str]) -> dict:
        """
        Reconstruct a sequence of ISL glosses into a fluent English sentence.
        
        Args:
            glosses: List of detected sign words/glosses (e.g. ['COLLEGE', 'I', 'GO', 'FINISH']).
            
        Returns:
            dict containing:
                - sentence: Reconstructed English sentence
                - original_glosses: Original gloss list
                - explanation: Grammatical breakdown (if available)
                - latency_ms: Execution time in milliseconds
        """
        if not glosses:
            return {
                "sentence": "",
                "original_glosses": [],
                "latency_ms": 0.0
            }

        # Filter out empty or whitespace tokens
        cleaned_glosses = [str(g).strip().upper() for g in glosses if str(g).strip()]
        if not cleaned_glosses:
            return {
                "sentence": "",
                "original_glosses": [],
                "latency_ms": 0.0
            }

        # If only a single word, capitalize and return directly unless grammar context is needed
        if len(cleaned_glosses) == 1:
            return {
                "sentence": cleaned_glosses[0].capitalize() + ".",
                "original_glosses": cleaned_glosses,
                "latency_ms": 0.0
            }

        # Deduplicate consecutive tokens: ['HELLO', 'HELLO'] -> ['HELLO']
        deduped = []
        for g in cleaned_glosses:
            if not deduped or deduped[-1] != g:
                deduped.append(g)

        rule_sentence = self._rule_based_reconstruct(deduped)

        if not self.is_available():
            return {
                "sentence": rule_sentence,
                "original_glosses": deduped,
                "engine": "isl-rule-grammar",
                "latency_ms": 0.0,
            }

        t0 = time.time()
        gloss_str = json.dumps(deduped)
        prompt = (
            "You are an expert Indian Sign Language (ISL) linguist and interpreter.\n"
            "Indian Sign Language uses Subject-Object-Verb (SOV) and topic-comment grammatical structures, "
            "without English auxiliary verbs (e.g., 'is', 'have') or morphological tense inflections.\n\n"
            f"Given this sequence of detected ISL gloss tokens in order:\n{gloss_str}\n\n"
            "Translate and reconstruct these glosses into a single natural, fluent English sentence with proper grammar and punctuation.\n"
            "Return JSON matching this schema:\n"
            "{\n"
            '  "sentence": "<fluent English translation>",\n'
            f'  "original_glosses": {gloss_str}\n'
            "}"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        }

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            latency = round((time.time() - t0) * 1000, 2)

            if resp.status_code == 200:
                data = resp.json()
                candidate_text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "{}")
                )
                result = json.loads(candidate_text)
                sentence = result.get("sentence", "").strip()
                if sentence:
                    return {
                        "sentence": sentence,
                        "original_glosses": deduped,
                        "engine": "gemini-transformer-decoder",
                        "latency_ms": latency
                    }
        except Exception:
            pass

        return {
            "sentence": rule_sentence,
            "original_glosses": deduped,
            "engine": "isl-rule-grammar",
            "latency_ms": round((time.time() - t0) * 1000, 2)
        }

    @staticmethod
    def _rule_based_reconstruct(glosses: list[str]) -> str:
        """
        Rule-based ISL syntax converter (SOV/Topic-Comment -> Fluent English).
        Handles greetings, pronouns, question structures, and multi-word signs.
        """
        if not glosses:
            return ""

        vocab_map = {
            "GOODMORNING": "good morning",
            "THANKYOU": "thank you",
            "BIGLARGE": "big",
            "SMALLLITTLE": "small",
            "YOUPLURAL": "you all",
            "TRAINTICKET": "train ticket",
            "STOREORSHOP": "store",
            "T_SHIRT": "t-shirt",
            "TSHIRT": "t-shirt",
            "CELLPHONE": "phone",
            "DEAF": "deaf",
        }

        words = [vocab_map.get(g, g.lower()) for g in glosses]

        # Single word expressions
        if len(words) == 1:
            w = words[0]
            if w in ("hello", "good morning", "thank you"):
                return f"{w.capitalize()}!"
            return f"{w.capitalize()}."

        # Common 2-word and 3-word idiomatic patterns in ISL
        joined_upper = " ".join(glosses)
        patterns = {
            "HELLO GOODMORNING": "Hello, good morning!",
            "HELLO HOW YOU": "Hello, how are you?",
            "HOW YOU": "How are you?",
            "YOU HOW": "How are you?",
            "WHAT NAME YOU": "What is your name?",
            "YOU NAME WHAT": "What is your name?",
            "NAME MY": "My name is...",
            "WHAT TIME": "What time is it?",
            "TIME WHAT": "What time is it?",
            "TIME HOUR": "The time is one hour.",
            "I HAPPY": "I am happy.",
            "I SAD": "I am sad.",
            "I TIRED": "I am tired.",
            "I SICK": "I am sick.",
            "I DEAF": "I am deaf.",
            "I GO HOUSE": "I am going home.",
            "HOUSE I GO": "I am going home.",
            "MEET TOMORROW": "See you tomorrow.",
            "TOMORROW MEET": "See you tomorrow.",
            "TODAY FRIDAY": "Today is Friday.",
            "WEATHER COLD": "The weather is cold.",
            "WEATHER HOT": "The weather is hot.",
        }

        if joined_upper in patterns:
            return patterns[joined_upper]

        # Question handling (if contains what, how, where, when, why)
        q_words = {"what", "how", "where", "when", "why"}
        has_q = any(w in q_words for w in words)
        if has_q:
            # Place question word first
            q = next(w for w in words if w in q_words)
            rest = [w for w in words if w != q]
            if "name" in rest and "you" in rest:
                return "What is your name?"
            if "time" in rest:
                return "What time is it?"
            return f"{q.capitalize()} {' '.join(rest)}?"

        # Subject + Adjective / Copula (e.g. ['dog', 'big'] -> "The dog is big.")
        adjectives = {"big", "small", "good", "bad", "happy", "sad", "cold", "hot", "fast", "slow", "old", "new", "beautiful", "ugly", "loud", "quiet", "cheap", "expensive", "tall", "short"}
        if len(words) == 2 and words[1] in adjectives:
            subj = words[0]
            if subj == "i":
                return f"I am {words[1]}."
            elif subj == "you":
                return f"You are {words[1]}."
            else:
                return f"The {subj} is {words[1]}."

        # General sentence fallback with capitalized first word and period
        clean_sentence = " ".join(words).strip()
        # Clean up 'i' to uppercase 'I'
        parts = clean_sentence.split()
        parts = ["I" if p == "i" else p for p in parts]
        res = " ".join(parts)
        return res[0].upper() + res[1:] + "."
