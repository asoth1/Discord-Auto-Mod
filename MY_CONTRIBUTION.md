# My contribution to Discord AutoMod

CMSC473 team project (Spring 2026). My slice was **spam and behavioral
analysis** — the layer that catches abusive *patterns* the toxicity
classifier alone wouldn't see, like flooding, repeated messages, link
spam, mass mentions, and bot-style copypasta.

## Files I authored

- [`spam_behavioral.py`](spam_behavioral.py) — the `SpamBehavioralAnalyzer`
  class, the `UserState` and `AnalysisResult` dataclasses, all detection
  signals, and the verdict logic.
- [`test_spam_behavioral.py`](test_spam_behavioral.py) — 26 tests covering
  every detection signal, plus per-user isolation, history reset, empty
  input, short-caps false-positive guards, verdict thresholds, the
  overall-score aggregation rule, the warnings counter, history-size
  bounds, and false-positive guards on the heuristic fallback.

The integration glue in `main.py` (lines 14, 20, 44–46) calls into my
module — the `spam_analyzer` is held at module scope so per-user history
persists across messages, which the rest of the pipeline relies on.

## What `SpamBehavioralAnalyzer` actually does

For each incoming message it computes seven independent signals and folds
them into one verdict:

| Signal | What it catches | How |
|---|---|---|
| `spam_score` | Promo / scam / generic spam content | Hugging Face `mshenoda/roberta-spam` text classifier, with a regex-pattern fallback (`click here`, `buy now`, `free {nitro\|gift\|giveaway\|robux\|vbucks}`, `nitro {free\|gift\|giveaway}`, `giveaway`, `discord.gg/`) when the transformer isn't available. The patterns require *combinations* — bare `free` or `nitrogen` won't trigger |
| `repetition_score` | Exact-duplicate messages | Normalize whitespace + casing, check against the user's recent history |
| `similarity_score` | Near-duplicates / paraphrased copypasta | `difflib.SequenceMatcher.ratio()` against each recent message |
| `flood_score` | Channel flooding | Count messages within a sliding time window (default 10s, threshold 6) |
| `mention_score` | Mass-ping abuse | Regex over `<@id>`, `<@!id>`, `@everyone`, `@here`; the latter two are an instant 1.0 |
| `link_score` | Link spam, especially Discord invites | URL count + density-per-token + a `+0.3` Discord-invite bonus + a `+0.5` multi-link bonus; a single non-invite link scores 0 |
| `char_spam_score` | All-caps yelling, key-mashing | Caps ratio above 0.7 on messages with `> 15` letters, or any `≥ 8`-char repeated-character run |

These are combined into a single `behavioral_score` (max of the six
behavioral signals) and an `overall_score` (max of `spam_score` and
`behavioral_score`). The verdict tiers are `block` (`overall ≥ 0.85`),
`warn` (`overall ≥ 0.6`), and `allow` otherwise. The analyzer also
produces a human-readable `reasons: list[str]` explaining what tripped
(e.g. `"near-duplicate of recent message (0.91)"`, `"flooding channel"`),
and bumps a per-user `warnings` counter on every non-`allow` verdict.

The class is designed to be **stateful and per-user**: an internal
`defaultdict[str, UserState]` holds each user's bounded message history
(`deque(maxlen=…)`) and a warning counter, so the `repetition`,
`similarity`, and `flood` signals look at *that user's* recent activity
and aren't confused by cross-user noise.

## What the rest of the pipeline does with my output

`AnalysisResult.to_dict()` returns a flat dict that
[`fusion_update.py`](fusion_update.py) (Ying's module) consumes via
`adapt_spam_output(...)`, which surfaces seven of my fields into the
fused feature vector: `spam`, `repetition`, `url_risk` (← `link_score`),
`behavioral`, `flood`, `mention`, `char_spam`. The decision system in
[`decision_system.py`](decision_system.py) then weights `behavioral`,
`url_risk`, and `repetition` heavily and adds hard rules — for example,
`url_risk ≥ 0.85 and behavioral ≥ 0.75 → mute` (aggressive spam), and
`url_risk ≥ 0.65 or repetition ≥ 0.85 → delete` (link spam / copypasta).

## Test coverage

`test_spam_behavioral.py` has 26 tests:

- **Allow path:** clean conversational messages all return `allow`,
  short caps messages don't false-positive as char-spam, empty messages
  don't crash, and a single non-invite link is left alone.
- **Repetition + similarity:** exact duplicates score 1.0 and block;
  near-duplicates with one extra punctuation mark hit `≥ 0.85` and
  warn/block.
- **Flood:** six messages within the window drives `flood_score` to 1.0
  and triggers a warn/block; the same six messages spread over time
  expire out of the window and the score drops back below 0.5.
- **Mentions:** `@everyone` is an instant 1.0 / block; five `<@id>`
  mentions reach the threshold.
- **Links:** three URLs in one message hits the block tier; a
  `discord.gg/` invite alone produces a non-zero link score.
- **Char spam:** all-caps long messages flag, long repeated-character
  runs (`aaaaaaaa…`) hit 1.0 / block.
- **Heuristic fallback:** with `use_model=False`, classic spam phrases
  (`FREE NITRO click here buy now`) still get `spam_score > 0.5`, and
  known combos like `free nitro` or a `discord.gg/` invite each clear
  `≥ 0.5` on their own. False-positive guards: bare URLs, `"feel free
  to ask"`, and `"nitrogen is an element"` all score `0.0`.
- **Per-user isolation:** Alice posting `"hello world"` does not cause
  Bob's `"hello world"` to flag as a repeat.
- **State management:** `reset_user(...)` clears history; `to_dict()`
  exposes every score field with the right types (floats for scores,
  enum-string verdict, list reasons); `history_size` truly bounds the
  per-user deque; the `warnings` counter increments on every non-`allow`
  verdict and is untouched on `allow`.
- **Aggregation + thresholds:** `overall_score == max(spam_score,
  behavioral_score)` exactly; low-score messages return `allow` with
  `overall < 0.6`, and a sub-flood burst of distinct messages stays
  under the warn threshold.

Run the tests with:

```bash
python test_spam_behavioral.py
```

The module also has a `__main__` demo at the bottom of
`spam_behavioral.py` that walks several archetypal messages through
the analyzer.

## Design choices worth flagging

- **Two-mode operation.** The transformer dependency is *optional*. If
  `transformers` isn't installed (or fails to load the model — common in
  CI / grader environments without network), the `_spam_score` path
  falls back to a small keyword heuristic and the rest of the analyzer
  works unchanged. This was important so my teammates could run the
  pipeline without a 1.5GB model download.
- **Bounded per-user state.** `UserState.messages` is a `deque(maxlen=20)`,
  so memory is O(users × 20) regardless of channel volume. Old messages
  age out cleanly without explicit cleanup.
- **Max-aggregation, not sum.** `behavioral_score = max(...)` rather than
  a sum or weighted average — one strong signal (e.g. flooding alone) is
  enough to act on, and stacking weaker signals shouldn't produce false
  positives by accumulation. The decision-system layer handles weighting
  across modules.

## Commits

- [`59fbda2`](https://github.com/spotlur2/Discord-Auto-Mod/commit/59fbda2)
  — *"Add SpamBehavioralAnalyzer and tests for spam detection and
  behavioral analysis"*
- [`b40690c`](https://github.com/spotlur2/Discord-Auto-Mod/commit/b40690c)
  — *"Enhance spam detection with heuristic patterns and expand test
  coverage"* (regex patterns replacing the bare-keyword heuristic, plus
  the 9 new tests bringing the suite to 26)
