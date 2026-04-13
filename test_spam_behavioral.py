"""Tests for SpamBehavioralAnalyzer."""

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
    ]
    passed = sum(1 for t in tests if t())
    total = len(tests)
    print(f"\n{passed}/{total} tests passed")
