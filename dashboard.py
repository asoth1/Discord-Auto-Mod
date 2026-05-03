import streamlit as st
import pandas as pd
from main import run_pipeline

# page title display
st.set_page_config(page_title="Discord AutoMod Dashboard", layout="wide")

# Header of the dashboard with Discord icon
st.markdown("""
<h1 style="display:flex; align-items:center; gap:12px;">
    <img src="https://pngimg.com/uploads/discord/discord_PNG27.png"
         width="50"
         style="background-color:transparent;">
    Discord AutoMod Interactive Dashboard
</h1>
""", unsafe_allow_html = True)

st.write(
    "Analyze Discord messages and view moderation decisions, risk scores, and feature explanations."
)

# for keeping message history
if "history" not in st.session_state:
    st.session_state.history = []

# sidebar (left panel)
st.sidebar.title("⚙️ Moderator Controls")

user_id = st.sidebar.text_input("User ID", "sunshine")

context_text = st.sidebar.text_area(
    "Context messages",
    "hey!"
)

context_messages = [
    line.strip()
    for line in context_text.split("\n")
    if line.strip()
]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Context messages are previous chat messages. "
    "The bottom chat box is the new message being analyzed."
)

if st.sidebar.button("Clear History"):
    st.session_state.history = []

# chat input 
message = st.chat_input("Type a Discord message to analyze...")

if message:
    result = run_pipeline(user_id, message, context_messages)
    st.session_state.history.append(result)

# main (body) display
if st.session_state.history:
    result = st.session_state.history[-1]

    fused = result["fused_output"]
    decision = result["decision"]

    action = decision["action"]
    risk_score = float(decision["risk_score"])

    # showing lastes message
    st.markdown('<div class="discord-card">', unsafe_allow_html = True)
    st.markdown("## 💬 Latest Message")
    st.markdown(
        f'<div class="message-box"><b>{result["user_id"]}</b>: {result["message"]}</div>',
        unsafe_allow_html = True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # set how to show the action after adding message
    if action == "allow":
        st.markdown(
            f'<div class="allow">✅ Action: {action.upper()}</div>',
            unsafe_allow_html = True
        )
    elif action == "warn":
        st.markdown(
            f'<div class="warn">⚠️ Action: {action.upper()}</div>',
            unsafe_allow_html = True
        )
    else:
        st.markdown(
            f'<div class="danger">🚫 Action: {action.upper()}</div>',
            unsafe_allow_html = True
        )

    st.markdown("<br>", unsafe_allow_html = True)

    # for metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{risk_score:.3f}</div>
            </div>
            """,
            unsafe_allow_html = True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">User ID</div>
                <div class="metric-value">{result["user_id"]}</div>
            </div>
            """,
            unsafe_allow_html = True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Decision</div>
                <div class="metric-value">{action}</div>
            </div>
            """,
            unsafe_allow_html = True
        )

    # for risk features
    risk_features = {
        "toxicity": fused.get("toxicity", 0),
        "insult": fused.get("insult", 0),
        "threat": fused.get("threat", 0),
        "anger": fused.get("anger", 0),
        "disgust": fused.get("disgust", 0),
        "sarcasm": fused.get("sarcasm", 0),
        "spam": fused.get("spam", 0),
        "url_risk": fused.get("url_risk", 0),
        "flood": fused.get("flood", 0),
        "mention": fused.get("mention", 0),
        "char_spam": fused.get("char_spam", 0),
    }

    # explaination section
    st.markdown("### 🧠 Explanation")
    st.write("**Reason:**", decision["reason"])

    st.markdown("### 🔍 Why this decision?")

    important = sorted(
        risk_features.items(),
        key = lambda x: x[1],
        reverse = True
    )[:3]

    top_feature = max(risk_features, key=risk_features.get)

    for name, value in important:
        if name == top_feature:
            st.write(f"🔥 **{name}**: {round(value, 3)} (main driver)")
        else:
            st.write(f"- **{name}**: {round(value, 3)}")

    top_feature = max(risk_features, key = risk_features.get)

    st.write(
        f"🔥 **Main Risk Driver:** `{top_feature}` = `{risk_features[top_feature]:.3f}`"
    )

    # showing feature bar chart
    st.markdown("### 📊 Feature Scores")

    feature_df = pd.DataFrame(
        list(risk_features.items()),
        columns = ["Feature", "Score"]
    ).sort_values(by = "Score", ascending = False)

    st.bar_chart(feature_df.set_index("Feature"), color="#5865F2")

    st.write("Higher values indicate stronger influence on the moderation decision.")

    # score breakdown section
    st.markdown("### 🔎 Score Breakdown")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### ☠️ Toxicity")
        for key in ["toxicity", "insult", "threat"]:
            value = float(fused.get(key, 0))
            st.write(key, round(value, 3))
            st.progress(value)

    with col2:
        st.markdown("#### 😡 Emotion")
        for key in ["anger", "disgust", "sarcasm"]:
            value = float(fused.get(key, 0))
            st.write(key, round(value, 3))
            st.progress(value)

    with col3:
        st.markdown("#### 🚫 Spam / Behavior")
        for key in ["spam", "url_risk", "flood"]:
            value = float(fused.get(key, 0))
            st.write(key, round(value, 3))
            st.progress(value)

    # showing raw outputs
    with st.expander("Raw Model Outputs"):
        st.json(result["raw_outputs"])

    # showing moderation history table
    st.markdown("### 📝 Moderation History")
    
    st.write("This table shows how the system responds to different types of messages over time.")

    history_rows = []

    action_display = {
        "allow": "🟢 allow",
        "warn": "🟡 warn",
        "delete": "🔴 delete"
    }

    for item in st.session_state.history:
        history_rows.append({
            "User": item["user_id"],
            "Message": item["message"],
            "Action": action_display.get(item["decision"]["action"], item["decision"]["action"]),
            "Risk Score": round(float(item["decision"]["risk_score"]), 3),
            "Reason": item["decision"]["reason"]
        })

    st.dataframe(
        pd.DataFrame(history_rows[::-1]),
        use_container_width = True
    )

else:
    st.info("Type a Discord message in the bottom chat box to start.")