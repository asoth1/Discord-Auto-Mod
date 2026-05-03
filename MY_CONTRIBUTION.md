# My contribution to Discord AutoMod

CMSC473 team project (Spring 2026). My slice was **spam and behavioral
analysis** — the layer that catches abusive *patterns* the toxicity
classifier alone wouldn't see, like flooding, repeated messages, link
spam, mass mentions, and bot-style copypasta.

## Files I authored

- [`spam_behavioral.py`](spam_behavioral.py) — the `SpamBehavioralAnalyzer`
  class, the `UserState` and `AnalysisResult` dataclasses, all detection
  signals, and the verdict logic.
- [`test_spam_behavioral.py`](test_spam_behavioral.py) — 17 tests covering
  every detection signal, plus per-user isolation, history reset, empty
  input, and short-caps false-positive guards.

The integration glue in `main.py` (lines 14, 20, 44–46) calls into my
module — the `spam_analyzer` is held at module scope so per-user history
persists across messages, which the rest of the pipeline relies on.

## What `SpamBehavioralAnalyzer` actually does

For each incoming message it computes seven independent signals and folds
them into one verdict:

| Signal | What it catches | How |
|---|---|---|
| `spam_score` | Promo / scam / generic spam content | Hugging Face `mshenoda/roberta-spam` text classifier, with a heuristic-keyword fallback (`free`, `click here`, `discord.gg/`, …) when the transformer isn't available |
| `repetition_score` | Exact-duplicate messages | Normalize whitespace + casing, check against the user's recent history |
| `similarity_score` | Near-duplicates / paraphrased copypasta | `difflib.SequenceMatcher.ratio()` against each recent message |
| `flood_score` | Channel flooding | Count messages within a sliding time window (default 10s, threshold 6) |
| `mention_score` | Mass-ping abuse | Regex over `<@id>`, `<@!id>`, `@everyone`, `@here`; the latter two are an instant 1.0 |
| `link_score` | Link spam, especially Discord invites | URL count + density-per-token + invite bonus |
| `char_spam_score` | All-caps yelling, key-mashing | Caps ratio above 0.7 on long messages, or any `≥ 8`-char repeated-character run |

These are combined into a single `behavioral_score` (max of the six
behavioral signals) and an `overall_score` (max of `spam_score` and
`behavioral_score`). The verdict tiers are `allow` / `warn` / `block`
based on the overall score, and the analyzer also produces a
human-readable `reasons: list[str]` explaining what tripped (e.g.
`"near-duplicate of recent message (0.91)"`, `"flooding channel"`).

The class is designed to be **stateful and per-user**: an internal
`defaultdict[str, UserState]` holds each user's bounded message history
(`deque(maxlen=…)`) and a warning counter, so the `repetition`,
`similarity`, and `flood` signals look at *that user's* recent activity
and aren't confused by cross-user noise.

## What the rest of the pipeline does with my output

`AnalysisResult.to_dict()` returns a flat dict that
[`fusion_update.py`](fusion_update.py) (Ying's module) consumes via
`adapt_spam_output(...)`, which surfaces five of my fields into the
fused feature vector: `spam`, `repetition`, `url_risk`, `behavioral`,
`flood`, `mention`, `char_spam`. The decision system in
[`decision_system.py`](decision_system.py) then weights `behavioral`,
`url_risk`, and `repetition` heavily and adds hard rules — for example,
`url_risk ≥ 0.85 and behavioral ≥ 0.75 → mute` (aggressive spam),
`repetition ≥ 0.85 → delete` (copypasta).

## Test coverage

`test_spam_behavioral.py` has 17 tests:

- **Allow path:** clean conversational messages all return `allow`,
  short caps messages don't false-positive as char-spam, empty messages
  don't crash.
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
  (`FREE NITRO click here buy now`) still get `spam_score > 0.5`.
- **Per-user isolation:** Alice posting `"hello world"` does not cause
  Bob's `"hello world"` to flag as a repeat.
- **State management:** `reset_user(...)` clears history; `to_dict()`
  exposes every score field.

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

## Commit

[`59fbda2`](https://github.com/spotlur2/Discord-Auto-Mod/commit/59fbda2)
— *"Add SpamBehavioralAnalyzer and tests for spam detection and
behavioral analysis"*
