"""Spam classification + behavioral analysis for Discord AutoMod.

Exposes SpamBehavioralAnalyzer, which scores a message on two axes:
  1. Spam content  — pretrained text classifier (HF transformers)
  2. Behavioral    — per-user history: repetition, similarity, flood rate,
                     mention spam, link spam, character spam

analyze(user_id, message) returns a dict of scores + an overall verdict,
ready to be fused with the toxicity/sentiment modules downstream.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Deque, Dict, Optional

try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


SPAM_MODEL = "mshenoda/roberta-spam"

URL_RE = re.compile(r"https?://\S+|www\.\S+|discord\.gg/\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"<@!?\d+>|@everyone|@here")


@dataclass
class UserState:
    messages: Deque[tuple[float, str]] = field(default_factory=lambda: deque(maxlen=20))
    warnings: int = 0


@dataclass
class AnalysisResult:
    spam_score: float
    repetition_score: float
    similarity_score: float
    flood_score: float
    mention_score: float
    link_score: float
    char_spam_score: float
    behavioral_score: float
    overall_score: float
    verdict: str
    reasons: list[str]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class SpamBehavioralAnalyzer:
    def __init__(
        self,
        history_size: int = 10,
        flood_window_sec: float = 10.0,
        flood_threshold: int = 6,
        similarity_threshold: float = 0.85,
        use_model: bool = True,
    ):
        self.history_size = history_size
        self.flood_window = flood_window_sec
        self.flood_threshold = flood_threshold
        self.similarity_threshold = similarity_threshold
        self._users: Dict[str, UserState] = defaultdict(
            lambda: UserState(messages=deque(maxlen=history_size))
        )

        self._classifier = None
        if use_model and _HAS_TRANSFORMERS:
            try:
                self._classifier = pipeline("text-classification", model=SPAM_MODEL)
            except Exception:
                self._classifier = None

    def analyze(
        self, user_id: str, message: str, timestamp: Optional[float] = None
    ) -> AnalysisResult:
        ts = timestamp if timestamp is not None else time.time()
        state = self._users[user_id]

        spam_score = self._spam_score(message)
        repetition_score = self._repetition_score(state, message)
        similarity_score = self._similarity_score(state, message)
        flood_score = self._flood_score(state, ts)
        mention_score = self._mention_score(message)
        link_score = self._link_score(message)
        char_spam_score = self._char_spam_score(message)

        behavioral_score = max(
            repetition_score,
            similarity_score,
            flood_score,
            mention_score,
            link_score,
            char_spam_score,
        )
        overall_score = max(spam_score, behavioral_score)

        reasons: list[str] = []
        if spam_score >= 0.7:
            reasons.append(f"classifier flagged spam ({spam_score:.2f})")
        if repetition_score >= 0.9:
            reasons.append("exact duplicate of recent message")
        elif similarity_score >= self.similarity_threshold:
            reasons.append(f"near-duplicate of recent message ({similarity_score:.2f})")
        if flood_score >= 0.8:
            reasons.append("flooding channel")
        if mention_score >= 0.7:
            reasons.append("excessive mentions")
        if link_score >= 0.7:
            reasons.append("link spam")
        if char_spam_score >= 0.7:
            reasons.append("character/caps spam")

        if overall_score >= 0.85:
            verdict = "block"
        elif overall_score >= 0.6:
            verdict = "warn"
        else:
            verdict = "allow"

        state.messages.append((ts, message))
        if verdict != "allow":
            state.warnings += 1

        return AnalysisResult(
            spam_score=spam_score,
            repetition_score=repetition_score,
            similarity_score=similarity_score,
            flood_score=flood_score,
            mention_score=mention_score,
            link_score=link_score,
            char_spam_score=char_spam_score,
            behavioral_score=behavioral_score,
            overall_score=overall_score,
            verdict=verdict,
            reasons=reasons,
        )

    def reset_user(self, user_id: str) -> None:
        self._users.pop(user_id, None)

    def _spam_score(self, message: str) -> float:
        if not self._classifier or not message.strip():
            return self._heuristic_spam(message)
        try:
            out = self._classifier(message[:512])[0]
            label = str(out.get("label", "")).lower()
            score = float(out.get("score", 0.0))
            return score if "spam" in label and "not" not in label else 1.0 - score
        except Exception:
            return self._heuristic_spam(message)

    @staticmethod
    def _heuristic_spam(message: str) -> float:
        if not message:
            return 0.0
        triggers = ("free", "click here", "buy now", "http", "discord.gg/", "nitro", "giveaway")
        hits = sum(1 for t in triggers if t in message.lower())
        return min(1.0, hits / 3.0)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def _repetition_score(self, state: UserState, message: str) -> float:
        norm = self._normalize(message)
        if not norm:
            return 0.0
        for _, prev in state.messages:
            if self._normalize(prev) == norm:
                return 1.0
        return 0.0

    def _similarity_score(self, state: UserState, message: str) -> float:
        norm = self._normalize(message)
        if not norm or not state.messages:
            return 0.0
        best = 0.0
        for _, prev in state.messages:
            ratio = SequenceMatcher(None, norm, self._normalize(prev)).ratio()
            if ratio > best:
                best = ratio
        return best

    def _flood_score(self, state: UserState, ts: float) -> float:
        recent = 1 + sum(1 for t, _ in state.messages if ts - t <= self.flood_window)
        return min(1.0, recent / self.flood_threshold)

    @staticmethod
    def _mention_score(message: str) -> float:
        mentions = MENTION_RE.findall(message)
        if "@everyone" in message or "@here" in message:
            return 1.0
        return min(1.0, len(mentions) / 5.0)

    @staticmethod
    def _link_score(message: str) -> float:
        links = URL_RE.findall(message)
        if not links:
            return 0.0
        tokens = max(1, len(message.split()))
        density = len(links) / tokens
        invite_bonus = 0.3 if "discord.gg/" in message.lower() else 0.0
        return min(1.0, density * 2 + invite_bonus + (0.5 if len(links) >= 2 else 0.0))

    @staticmethod
    def _char_spam_score(message: str) -> float:
        if len(message) < 10:
            return 0.0
        letters = [c for c in message if c.isalpha()]
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
        longest_run = 1
        run = 1
        for i in range(1, len(message)):
            if message[i] == message[i - 1] and not message[i].isspace():
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 1
        run_score = min(1.0, longest_run / 8.0)
        caps_score = caps_ratio if caps_ratio > 0.7 and len(letters) > 15 else 0.0
        return max(run_score, caps_score)


if __name__ == "__main__":
    analyzer = SpamBehavioralAnalyzer(use_model=False)
    samples = [
        ("alice", "hey everyone, how's it going?"),
        ("alice", "anyone want to play later?"),
        ("bob",   "FREE NITRO CLICK HERE discord.gg/scamlink"),
        ("carl",  "spam spam spam spam spam"),
        ("carl",  "spam spam spam spam spam"),
        ("carl",  "spam spam spam spam spam!"),
        ("dave",  "AAAAAAAAAAAAAAAAAAA"),
        ("eve",   "@everyone check this out http://a.co http://b.co http://c.co"),
    ]
    for user, msg in samples:
        r = analyzer.analyze(user, msg)
        print(f"[{r.verdict.upper():5}] {user}: {msg}")
        print(f"        overall={r.overall_score:.2f} spam={r.spam_score:.2f} "
              f"beh={r.behavioral_score:.2f} reasons={r.reasons}")
