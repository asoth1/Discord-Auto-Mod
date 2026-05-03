"""Tests for SpamBehavioralAnalyzer."""

import sys

from spam_behavioral import SpamBehavioralAnalyzer


def make_analyzer():
    return SpamBehavioralAnalyzer(use_model=False)


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    return cond


def test_clean_messages():
    print("test_clean_messages")
    a = make_analyzer()
    results = [
        a.analyze("u1", "hey everyone how's your day"),
        a.analyze("u1", "anyone playing valorant later?"),
        a.analyze("u1", "just finished my homework"),
    ]
    ok = all(r.verdict == "allow" for r in results)
    return check("all clean messages allowed", ok)


def test_exact_repetition():
    print("test_exact_repetition")
    a = make_analyzer()
    a.analyze("u1", "hello world", timestamp=1000)
    r = a.analyze("u1", "hello world", timestamp=1001)
    return check("exact duplicate blocked", r.verdict == "block" and r.repetition_score == 1.0)


def test_near_duplicate():
    print("test_near_duplicate")
    a = make_analyzer()
    a.analyze("u1", "check out this cool link please", timestamp=1000)
    r = a.analyze("u1", "check out this cool link please!", timestamp=1001)
    return check(
        "near-duplicate detected",
        r.similarity_score >= 0.85 and r.verdict in ("warn", "block"),
    )


def test_flood():
    print("test_flood")
    a = make_analyzer(); passed = True
    for i in range(6):
        r = a.analyze("flooder", f"message number {i}", timestamp=1000 + i)
    passed &= check("flood_score reaches 1.0 after 6 messages", r.flood_score >= 1.0)
    passed &= check("flood blocked/warned", r.verdict in ("warn", "block"))
    return passed


def test_flood_window_expires():
    print("test_flood_window_expires")
    a = make_analyzer()
    for i in range(4):
        a.analyze("u1", f"msg {i}", timestamp=1000 + i)
    r = a.analyze("u1", "much later message", timestamp=2000)
    return check("old messages don't count toward flood", r.flood_score < 0.5)


def test_mention_everyone():
    print("test_mention_everyone")
    a = make_analyzer()
    r = a.analyze("u1", "hey @everyone join now")
    return check("@everyone flagged", r.mention_score == 1.0 and r.verdict == "block")


def test_many_mentions():
    print("test_many_mentions")
    a = make_analyzer()
    r = a.analyze("u1", "<@111> <@222> <@333> <@444> <@555> come here")
    return check("5 mentions flagged", r.mention_score >= 1.0)


def test_link_spam():
    print("test_link_spam")
    a = make_analyzer()
    r = a.analyze("u1", "http://a.co http://b.co http://c.co")
    return check("multiple links flagged", r.link_score >= 0.7 and r.verdict == "block")


def test_discord_invite():
    print("test_discord_invite")
    a = make_analyzer()
    r = a.analyze("u1", "join discord.gg/freenitro")
    return check("discord invite flagged", r.link_score > 0.0)


def test_caps_spam():
    print("test_caps_spam")
    a = make_analyzer()
    r = a.analyze("u1", "STOP DOING THAT RIGHT NOW PLEASE")
    return check("all caps flagged", r.char_spam_score > 0.0)


def test_char_runs():
    print("test_char_runs")
    a = make_analyzer()
    r = a.analyze("u1", "aaaaaaaaaaaaaaaaa")
    return check("long char run flagged", r.char_spam_score >= 1.0 and r.verdict == "block")


def test_heuristic_spam():
    print("test_heuristic_spam")
    a = make_analyzer()
    r = a.analyze("u1", "FREE NITRO click here buy now")
    return check("heuristic catches spam phrases", r.spam_score > 0.5)


def test_per_user_isolation():
    print("test_per_user_isolation")
    a = make_analyzer()
    a.analyze("alice", "hello world", timestamp=1000)
    r = a.analyze("bob", "hello world", timestamp=1001)
    return check("different user not penalized for duplicate", r.repetition_score == 0.0)


def test_empty_message():
    print("test_empty_message")
    a = make_analyzer()
    r = a.analyze("u1", "")
    return check("empty message doesn't crash and is allowed", r.verdict == "allow")


def test_reset_user():
    print("test_reset_user")
    a = make_analyzer()
    a.analyze("u1", "hello", timestamp=1000)
    a.reset_user("u1")
    r = a.analyze("u1", "hello", timestamp=1001)
    return check("reset clears history", r.repetition_score == 0.0)


def test_short_message_not_caps_spam():
    print("test_short_message_not_caps_spam")
    a = make_analyzer()
    r = a.analyze("u1", "OK")
    return check("short caps message allowed", r.char_spam_score == 0.0 and r.verdict == "allow")


def test_result_to_dict():
    print("test_result_to_dict")
    a = make_analyzer()
    r = a.analyze("u1", "hello")
    d = r.to_dict()
    keys = {"spam_score", "repetition_score", "similarity_score", "flood_score",
            "mention_score", "link_score", "char_spam_score", "behavioral_score",
            "overall_score", "verdict", "reasons"}
    return check("to_dict has all fields", keys.issubset(d.keys()))


def test_single_benign_link():
    print("test_single_benign_link")
    a = make_analyzer()
    r = a.analyze("u1", "check out https://github.com")
    return check(
        "single non-invite link not flagged",
        r.link_score == 0.0 and r.verdict == "allow",
    )


def test_verdict_thresholds():
    print("test_verdict_thresholds")
    passed = True
    a = make_analyzer()
    r = a.analyze("u1", "hi", timestamp=1000)
    passed &= check("low score → allow", r.verdict == "allow" and r.overall_score < 0.6)

    a2 = make_analyzer()
    distinct = ["hello there friend", "what's up today", "good morning all"]
    r2 = None
    for i, msg in enumerate(distinct):
        r2 = a2.analyze("u2", msg, timestamp=1000 + i)
    passed &= check(
        "flood under threshold → allow",
        r2.flood_score < 0.6 and r2.verdict == "allow",
    )
    return passed


def test_overall_is_max_of_spam_and_behavioral():
    print("test_overall_is_max_of_spam_and_behavioral")
    a = make_analyzer()
    r = a.analyze("u1", "FREE NITRO giveaway click here")
    expected = max(r.spam_score, r.behavioral_score)
    return check(
        "overall_score == max(spam, behavioral)",
        abs(r.overall_score - expected) < 1e-9,
    )


def test_warnings_counter():
    print("test_warnings_counter")
    passed = True
    a = make_analyzer()
    a.analyze("u1", "hi there", timestamp=1000)
    passed &= check("allow doesn't bump warnings", a._users["u1"].warnings == 0)
    a.analyze("u1", "hi there", timestamp=1001)
    passed &= check("block bumps warnings", a._users["u1"].warnings == 1)
    a.analyze("u1", "hi there", timestamp=1002)
    passed &= check("warnings accumulate", a._users["u1"].warnings == 2)
    return passed


def test_history_size_bounds():
    print("test_history_size_bounds")
    a = SpamBehavioralAnalyzer(use_model=False, history_size=3)
    for i in range(10):
        a.analyze("u1", f"unique message number {i}", timestamp=1000 + i * 100)
    return check("deque bounded by history_size", len(a._users["u1"].messages) == 3)


def test_heuristic_ignores_bare_url():
    print("test_heuristic_ignores_bare_url")
    a = make_analyzer()
    r = a.analyze("u1", "https://github.com/anthropics/claude-code")
    return check("bare URL doesn't trigger heuristic", r.spam_score == 0.0)


def test_heuristic_ignores_benign_free():
    print("test_heuristic_ignores_benign_free")
    a = make_analyzer()
    passed = True
    r = a.analyze("u1", "feel free to ask any questions", timestamp=1000)
    passed &= check("'feel free' doesn't trigger heuristic", r.spam_score == 0.0)
    r = a.analyze("u2", "nitrogen is an element on the periodic table", timestamp=1001)
    passed &= check("'nitrogen' doesn't match nitro pattern", r.spam_score == 0.0)
    return passed


def test_heuristic_catches_known_combos():
    print("test_heuristic_catches_known_combos")
    a = make_analyzer()
    passed = True
    r = a.analyze("u1", "FREE NITRO right here", timestamp=1000)
    passed &= check("'free nitro' flagged", r.spam_score >= 0.5)
    r = a.analyze("u2", "join discord.gg/scamlink for more", timestamp=1001)
    passed &= check("discord invite flagged by heuristic", r.spam_score >= 0.5)
    return passed


def test_to_dict_value_types():
    print("test_to_dict_value_types")
    a = make_analyzer()
    r = a.analyze("u1", "hello")
    d = r.to_dict()
    score_keys = ["spam_score", "repetition_score", "similarity_score", "flood_score",
                  "mention_score", "link_score", "char_spam_score", "behavioral_score",
                  "overall_score"]
    floats_ok = all(isinstance(d[k], float) for k in score_keys)
    verdict_ok = isinstance(d["verdict"], str) and d["verdict"] in {"allow", "warn", "block"}
    reasons_ok = isinstance(d["reasons"], list)
    return check(
        "score fields are floats; verdict/reasons typed",
        floats_ok and verdict_ok and reasons_ok,
    )


if __name__ == "__main__":
    tests = [
        test_clean_messages,
        test_exact_repetition,
        test_near_duplicate,
        test_flood,
        test_flood_window_expires,
        test_mention_everyone,
        test_many_mentions,
        test_link_spam,
        test_discord_invite,
        test_caps_spam,
        test_char_runs,
        test_heuristic_spam,
        test_per_user_isolation,
        test_empty_message,
        test_reset_user,
        test_short_message_not_caps_spam,
        test_result_to_dict,
        test_single_benign_link,
        test_verdict_thresholds,
        test_overall_is_max_of_spam_and_behavioral,
        test_warnings_counter,
        test_history_size_bounds,
        test_heuristic_ignores_bare_url,
        test_heuristic_ignores_benign_free,
        test_heuristic_catches_known_combos,
        test_to_dict_value_types,
    ]
    passed = sum(1 for t in tests if t())
    total = len(tests)
    print(f"\n{passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)
