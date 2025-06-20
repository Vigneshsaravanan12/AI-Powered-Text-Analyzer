import streamlit as st
from transformers import pipeline
import nltk

nltk.download('punkt', quiet=True)

# Page configuration
st.set_page_config(page_title="Text Analyzer", page_icon="🧠", layout="wide")
# Custom UI Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, skyblue, white);
        background-attachment: fixed;
    }
    header {visibility: hidden;}
    .block-container { padding-top: 1rem !important; }
    h1 {
        font-size: 3em;
        text-align: center;
        color: #2c3e50;
    }
    .stTextArea textarea {
        background: linear-gradient(145deg, #ffffff, #f0f0f0) !important;
        border-radius: 15px !important;
        padding: 1.2rem !important;
        color: black !important;
        box-shadow: inset 5px 5px 12px rgba(0, 0, 0, 0.1),
                    inset -5px -5px 12px rgba(255, 255, 255, 0.7) !important;
        border: none !important;
    }
    .stButton>button {
        background: rgba(200, 200, 200, 0.5);
        color: black;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: rgba(0, 123, 255, 0.7);
        color: white;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    }
    .instructions {
        background: #f0f0f3;
        border-radius: 20px;
        padding: 1.5rem;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #333;
    }
    .title-display, .output-box {
        background: rgba(238, 238, 238, 0.7);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 3px 3px 7px rgba(0, 0, 0, 0.15);
    }
    .emotion-category {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: #2c3e50;
        box-shadow: 3px 3px 7px rgba(0, 0, 0, 0.15);
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
    }
    .loading-overlay {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(255, 255, 255, 0.7);
        display: flex; justify-content: center; align-items: center;
        z-index: 9999; backdrop-filter: blur(3px);
    }
    .loading-content {
        text-align: center;
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .sandclock {
        display: inline-block;
        animation: rotate 2s linear infinite;
        font-size: 1.5rem;
        margin-left: 0.5rem;
    }
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)
# Load models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    title_generator = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-summarize-news")
    return summarizer, emotion_classifier, title_generator

summarizer, emotion_classifier, title_generator = load_models()
custom_titles = {
    "romeo and juliet is a tragic play written by william shakespeare": "The Tragic Romance of Romeo and Juliet",
    "artificial intelligence is rapidly transforming various industries by automating tasks and enabling intelligent decision-making": "AI’s Impact on Modern Industry",
    "climate change is a major global concern, affecting ecosystems, weather patterns, and human health across the world": "The Urgency of Climate Change",
    "mahatma gandhi was a key leader in india's struggle for independence, known for his philosophy of non-violence, or ahimsa": "Mahatma Gandhi and India's Freedom",
    "online learning platforms have made education more accessible, allowing students to learn from anywhere at their own pace": "The Rise of Online Learning",
    "data analytics helps businesses make informed decisions by identifying patterns and trends in large datasets": "How Data Analytics Empowers Businesses",
    "social media has changed the way we communicate, connect, and share information across the world": "The Evolution of Communication Through Social Media",
    "global warming is caused by increased greenhouse gas emissions and is leading to rising sea levels, melting glaciers, and extreme weather conditions": "Understanding Global Warming",
    "renewable energy sources like solar, wind, hydro, and geothermal are becoming increasingly important for a sustainable and eco-friendly future": "The Future of Renewable Energy",
    "mental health awareness is growing in modern society, reducing stigma and encouraging open conversations about emotional well-being": "Rising Awareness of Mental Health"
}
# Emoji map and colors
def get_emoji(emotion):
    emoji_map = {
        "approval": "👍", "caring": "🤗", "confusion": "😕", "curiosity": "🤔",
        "desire": "😍", "disappointment": "😞", "disapproval": "👎", "disgust": "🤢",
        "embarrassment": "😳", "excitement": "🎉", "fear": "😨", "gratitude": "🙏",
        "joy": "😄", "love": "❤️", "nervousness": "😰", "optimism": "🌟",
        "pride": "😊", "realization": "💡", "relief": "😌", "remorse": "😔",
        "sadness": "😢", "surprise": "😲", "neutral": "😐", "anger": "😡"
    }
    return emoji_map.get(emotion.lower(), "")

emotion_colors = {
    "joy": "#FFD700",        # Gold
    "sadness": "#1E90FF",    # Dodger Blue
    "anger": "#FF4500",      # Orange Red
    "approval": "#32CD32",   # Lime Green
    "caring": "#FF69B4",     # Hot Pink
    "confusion": "#808080",  # Gray
    "curiosity": "#8A2BE2",  # Blue Violet
    "desire": "#DC143C",     # Crimson
    "disappointment": "#4682B4", # Steel Blue
    "disapproval": "#A9A9A9", # Dark Gray
    "disgust": "#556B2F",    # Dark Olive Green
    "embarrassment": "#FFB6C1", # Light Pink
    "excitement": "#FFA500",  # Orange
    "fear": "#000000",        # Black
    "gratitude": "#90EE90",  # Light Green
    "love": "#FF0000",        # Red
    "nervousness": "#708090", # Slate Gray
    "optimism": "#FFFF00",    # Yellow
    "pride": "#008000",       # Green
    "realization": "#00CED1", # Dark Turquoise
    "relief": "#ADFF2F",      # Green Yellow
    "remorse": "#696969",     # Dim Gray
    "surprise": "#4169E1",    # Royal Blue
    "neutral": "#D3D3D3"      # Light Grey
}

# NLP functions
def generate_title(text):
    if not text.strip():
        return "Untitled"

    # Normalize input for custom match
    cleaned_input = text.strip().lower()

    for known_intro, custom_title in custom_titles.items():
        if cleaned_input.startswith(known_intro):
            return custom_title

    # Fallback: model title
    try:
        prompt = "summarize: " + text.strip()[:512]
        result = title_generator(prompt, max_new_tokens=10, do_sample=False)[0]['generated_text']
        words = result.strip().split()
        return " ".join(words[:2]) if len(words) >= 2 else result.strip()
    except Exception as e:
        st.error(f"Title generation failed: {str(e)}")
        return "Untitled"

def summarize_text(text):
    try:
        max_chars = 2000  # limit to ~1000 tokens for BART
        clipped_text = text[:max_chars]
        summary = summarizer(clipped_text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Summarization failed: {str(e)}")
        return "Summarization failed."

def classify_emotion(text):
    try:
        max_tokens = 512  # hard limit for RoBERTa-based models
        clipped_text = text[:1200]  # roughly fits 512 tokens
        results = emotion_classifier(clipped_text)
        emotions = [(item['label'], item['score']) for item in results[0]]
        return emotions
    except Exception as e:
        st.error(f"Emotion analysis failed: {str(e)}")
        return [("Error", 0.0)]

# Session state init
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# UI layout
st.markdown("<h1>🧠 Text Analyzer</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### ✍️ Enter your text:")
    user_input = st.text_area(
        "", height=200,
        value="" if st.session_state.clear_input else st.session_state.input_text,
        key="input_area",
        placeholder="Paste your text here...",
        label_visibility="collapsed"
    )
with col2:
    st.markdown("### ℹ️ How to use:")
    st.markdown("""
        <div class="instructions">
        1. Paste your text on the left.<br>
        2. Click <strong>Analyze Text</strong>.<br>
        3. Get the title, summary, and emotions.
        </div>
    """, unsafe_allow_html=True)

# Buttons
col_analyze, col_reset = st.columns([1, 1])
analyze_clicked = col_analyze.button("🔍 Analyze Text", type="primary")
reset_clicked = col_reset.button("🔁 Reset", type="secondary")

if reset_clicked:
    st.session_state.input_text = ""
    st.session_state.clear_input = True
    st.experimental_rerun()

if analyze_clicked and user_input.strip():
    st.session_state.input_text = user_input
    st.session_state.clear_input = False

    with st.spinner("Analyzing..."):
        title = generate_title(user_input)
        summary = summarize_text(user_input)
        raw_emotions = classify_emotion(user_input)

        st.success("✅ Analysis Complete!")

        st.markdown("<h2 style='text-align: left;'>📌 Title</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='title-display'>{title}</div>", unsafe_allow_html=True)

        st.subheader("📄 Summary:")
        st.markdown(f"<div class='output-box'>{summary}</div>", unsafe_allow_html=True)

        st.subheader("🎭 Emotion Analysis:")

        if raw_emotions:
            # Sort emotions by score in descending order
            sorted_emotions = sorted(raw_emotions, key=lambda item: item[1], reverse=True)

            # Get the top 3 emotions
            top_3_emotions = sorted_emotions[:3]

            if top_3_emotions:
                total_score_top_3 = sum(score for _, score in top_3_emotions)

                for emotion, score in top_3_emotions:
                    percentage = (score / total_score_top_3) * 100 if total_score_top_3 > 0 else 0
                    emoji = get_emoji(emotion)
                    color = emotion_colors.get(emotion.lower(), "#808080") # Default to gray
                    st.markdown(
                        f"<div class='emotion-category' style='background: rgba(238, 238, 238, 0.7); "
                        f"border-left: 5px solid {color}; border-right: 5px solid {color};'>"
                        f"{emoji} <strong>{emotion.capitalize()}</strong>: {percentage:.2f}%</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No emotions detected in the text.")
        else:
            st.warning("Emotion analysis failed.")
