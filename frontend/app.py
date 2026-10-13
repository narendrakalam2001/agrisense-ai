"""AgriSense AI — Streamlit Farmer Portal.

Run with:
    streamlit run frontend/app.py
or:
    python scripts/run_app.py

Five screens (sidebar navigation):
    Home | Disease Detection | Kisan Saathi Chatbot | Weather Advisory | Disease Encyclopedia
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import KNOWLEDGE_BASE_DIR, MODEL_H5_PATH, configure_logging
from src.chatbot import ChatbotError, KisanSaathiChatbot
from src.llm_advisor import LLMAdvisorError, build_fallback_explanation, explain_treatment_plan
from src.rag_engine import RAGEngine, RAGEngineError
from src.weather_service import WeatherServiceError, fetch_five_day_forecast, recommend_spray_days
from services.detection_service import DetectionService, DetectionServiceError

logger = configure_logging(__name__, log_filename="streamlit_app.log")

# ---------------------------------------------------------------------------
# Page config + color scheme (per project brief)
# ---------------------------------------------------------------------------
COLOR_PRIMARY = "#2E7D32"    # Green — agriculture feel
COLOR_SECONDARY = "#5D4037"  # Earth brown
COLOR_BACKGROUND = "#FAFAF0"  # Light cream
COLOR_ACCENT = "#FF6F00"     # Orange — call to action

st.set_page_config(page_title="AgriSense AI — Kisan Portal", page_icon="🌱", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {COLOR_BACKGROUND}; }}
    .agrisense-header {{
        background-color: {COLOR_PRIMARY}; color: white; padding: 1.2rem;
        border-radius: 10px; margin-bottom: 1rem;
    }}
    .agrisense-card {{
        background-color: white; border-radius: 10px; padding: 1rem;
        border-left: 6px solid {COLOR_PRIMARY}; margin-bottom: 1rem;
    }}
    .agrisense-card-urgent {{ border-left: 6px solid #C62828; }}
    .agrisense-card-organic {{ border-left: 6px solid {COLOR_PRIMARY}; }}
    .agrisense-card-prevent {{ border-left: 6px solid #F9A825; }}
    .agrisense-badge {{
        background-color: {COLOR_ACCENT}; color: white; padding: 0.2rem 0.6rem;
        border-radius: 6px; font-size: 0.8rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------
@st.cache_resource
def get_detection_service() -> DetectionService | None:
    try:
        return DetectionService()
    except DetectionServiceError as exc:
        logger.error("Could not load detection model: %s", exc)
        return None


@st.cache_resource
def get_rag_engine() -> RAGEngine | None:
    try:
        return RAGEngine()
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load RAG engine: %s", exc)
        return None


@st.cache_data
def load_disease_info() -> dict:
    path = KNOWLEDGE_BASE_DIR / "disease_info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if "language" not in st.session_state:
    st.session_state.language = "English"
if "chatbot_session" not in st.session_state:
    st.session_state.chatbot_session = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_detection" not in st.session_state:
    st.session_state.last_detection = None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🌱 AgriSense AI")
st.sidebar.caption("Apni fasal bachao, AI ki madad se")
st.session_state.language = st.sidebar.radio("भाषा / Language", ["English", "हिंदी"], horizontal=True)
screen = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📸 Disease Detection", "💬 Kisan Saathi Chatbot", "🌤️ Weather Advisory", "📖 Disease Encyclopedia"],
)

LANG_CODE = "hindi" if st.session_state.language == "हिंदी" else "english"


def t(en: str, hi: str) -> str:
    """Tiny inline translation helper for static UI strings."""
    return hi if LANG_CODE == "hindi" else en


# ---------------------------------------------------------------------------
# Screen 1 — Home
# ---------------------------------------------------------------------------
def render_home() -> None:
    st.markdown(
        f'<div class="agrisense-header"><h1>🌱 AgriSense AI</h1>'
        f'<p>{t("Save your crop with the power of AI", "Apni fasal bachao, AI ki madad se")}</p></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="agrisense-card"><h3>📸 {t("Disease Detection", "रोग पहचान")}</h3>'
            f'<p>{t("Upload a leaf photo for instant AI diagnosis.", "पत्ती की फोटो अपलोड करें, तुरंत AI निदान पाएं।")}</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="agrisense-card"><h3>💊 {t("Treatment Guide", "उपचार गाइड")}</h3>'
            f'<p>{t("Get chemical + organic treatment recommendations.", "रासायनिक और जैविक उपचार सुझाव पाएं।")}</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="agrisense-card"><h3>🌤️ {t("Weather Advisory", "मौसम सलाह")}</h3>'
            f'<p>{t("Know the best days to spray based on disease risk.", "रोग जोखिम के आधार पर छिड़काव के सही दिन जानें।")}</p></div>',
            unsafe_allow_html=True,
        )

    st.info(
        t(
            "120 million Indian farmers lose ~30% of crop yield to disease every year. "
            "Early detection can save ₹15,000-50,000 per acre.",
            "12 करोड़ भारतीय किसान हर साल बीमारी से ~30% फसल उपज खो देते हैं। "
            "जल्दी पहचान से प्रति एकड़ ₹15,000-50,000 तक बचाए जा सकते हैं।",
        )
    )


# ---------------------------------------------------------------------------
# Screen 2 — Disease Detection
# ---------------------------------------------------------------------------
def render_detection() -> None:
    st.header(f"📸 {t('Disease Detection', 'रोग पहचान')}")
    detection_service = get_detection_service()
    rag_engine = get_rag_engine()

    if detection_service is None:
        st.error(
            t(
                f"Model not found at {MODEL_H5_PATH}. Run scripts/train_model.py first.",
                f"मॉडल नहीं मिला ({MODEL_H5_PATH})। पहले scripts/train_model.py चलाएँ।",
            )
        )
        return

    uploaded_file = st.file_uploader(
        t("Upload or take a photo of the affected leaf", "प्रभावित पत्ती की फोटो अपलोड करें या लें"),
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info(t("👆 Upload a leaf image to get started.", "👆 शुरू करने के लिए पत्ती की फोटो अपलोड करें।"))
        return

    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption=t("Uploaded image", "अपलोड की गई फोटो"), width=300)

    with st.spinner(t("Analyzing image...", "फोटो का विश्लेषण हो रहा है...")):
        try:
            result = detection_service.detect(image_bytes)
        except DetectionServiceError as exc:
            st.error(str(exc))
            return

    st.session_state.last_detection = result

    st.subheader(t("Detection Result", "पहचान परिणाम"))
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric(t("Detected Disease", "पहचानी गई बीमारी"), result.predicted_class.replace("___", " — ").replace("_", " "))
        st.progress(min(result.confidence, 1.0), text=f"{t('Confidence', 'कॉन्फिडेंस')}: {result.confidence * 100:.1f}%")

        st.write(f"**{t('Top 3 possibilities', 'शीर्ष 3 संभावनाएँ')}:**")
        for pred in result.top_3:
            st.write(f"- {pred.class_name.replace('___', ' — ').replace('_', ' ')}: {pred.confidence * 100:.1f}%")

    with col2:
        if result.gradcam_image_base64:
            st.image(
                base64.b64decode(result.gradcam_image_base64),
                caption=t("Grad-CAM: red area = affected zone", "ग्रैड-कैम: लाल क्षेत्र = प्रभावित क्षेत्र"),
                width=300,
            )

    if rag_engine is None:
        st.warning(t("Treatment engine unavailable.", "उपचार इंजन उपलब्ध नहीं है।"))
        return

    try:
        plan = rag_engine.get_treatment_plan(result.predicted_class)
    except RAGEngineError as exc:
        st.warning(str(exc))
        return

    st.subheader(t("Treatment Plan", "उपचार योजना"))

    st.markdown(
        f'<div class="agrisense-card agrisense-card-urgent"><b>🚨 {t("Immediate Action", "तुरंत करें")}</b><br>'
        f'{plan.immediate_action_hi if LANG_CODE == "hindi" else plan.immediate_action_en}</div>',
        unsafe_allow_html=True,
    )

    chem = plan.chemical_treatment or {}
    if chem.get("pesticide_name") and chem.get("pesticide_name") != "Not applicable":
        st.markdown(
            f'<div class="agrisense-card"><b>💊 {t("Chemical Treatment", "रासायनिक उपचार")}</b><br>'
            f'{t("Pesticide", "दवाई")}: {chem.get("pesticide_name")} ({chem.get("brand_examples", "")})<br>'
            f'{t("Dosage", "मात्रा")}: {chem.get("dosage")}<br>'
            f'{t("Frequency", "आवृत्ति")}: {chem.get("frequency")}<br>'
            f'{t("Safety", "सुरक्षा")}: {chem.get("safety_hi") if LANG_CODE == "hindi" else chem.get("safety_en")}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="agrisense-card agrisense-card-organic"><b>🌿 {t("Organic Alternative", "जैविक विकल्प")}</b><br>'
        f'{plan.organic_alternative_hi if LANG_CODE == "hindi" else plan.organic_alternative_en}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="agrisense-card agrisense-card-prevent"><b>🛡️ {t("Prevention", "रोकथाम")}</b><br>'
        f'{plan.prevention_hi if LANG_CODE == "hindi" else plan.prevention_en}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{t('Estimated recovery', 'अनुमानित रिकवरी')}: {plan.estimated_recovery}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(f"💬 {t('Ask Chatbot', 'चैटबॉट से पूछें')}"):
            st.session_state.chat_history.append(
                ("assistant", f"{t('I see you just diagnosed', 'मैंने देखा आपने अभी पहचाना')}: {plan.disease_name_en}. {t('Ask me anything about treating it.', 'इसके उपचार के बारे में कुछ भी पूछें।')}")
            )
            st.info(t("Switch to the Kisan Saathi Chatbot tab to continue.", "जारी रखने के लिए Kisan Saathi Chatbot टैब पर जाएँ।"))
    with col_b:
        report_text = (
            f"AgriSense AI Report\nDisease: {plan.disease_name_en}\n"
            f"Confidence: {result.confidence * 100:.1f}%\n"
            f"Immediate action: {plan.immediate_action_en}\n"
            f"Chemical: {chem.get('pesticide_name')} - {chem.get('dosage')}\n"
            f"Organic: {plan.organic_alternative_en}\nPrevention: {plan.prevention_en}\n"
        )
        st.download_button(
            f"📄 {t('Save Report', 'रिपोर्ट सेव करें')}",
            data=report_text,
            file_name="agrisense_report.txt",
            mime="text/plain",
        )


# ---------------------------------------------------------------------------
# Screen 3 — Chatbot
# ---------------------------------------------------------------------------
def render_chatbot() -> None:
    st.header(f"💬 {t('Kisan Saathi Chatbot', 'किसान साथी चैटबॉट')}")
    rag_engine = get_rag_engine()

    if rag_engine is None:
        st.error(t("Chatbot backend unavailable.", "चैटबॉट बैकएंड उपलब्ध नहीं है।"))
        return

    if st.session_state.chatbot_session is None:
        try:
            st.session_state.chatbot_session = KisanSaathiChatbot(rag_engine=rag_engine)
        except Exception as exc:  # noqa: BLE001 - likely a missing GEMINI_API_KEY
            st.error(
                t(
                    f"Could not start chatbot: {exc}. Check GEMINI_API_KEY in your .env file.",
                    f"चैटबॉट शुरू नहीं हो सका: {exc}। अपनी .env फ़ाइल में GEMINI_API_KEY जाँचें।",
                )
            )
            return

    suggested_questions = [
        "Tomato mein kaun si bimari common hai?",
        "Organic spray kaise banayein?",
        "Aaj spray karna chahiye?",
    ]
    st.write(f"**{t('Try asking', 'यह पूछकर देखें')}:**")
    cols = st.columns(len(suggested_questions))
    for col, q in zip(cols, suggested_questions):
        if col.button(q):
            st.session_state.chat_history.append(("user", q))

    for role, content in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant", avatar="🧑‍🌾" if role == "user" else "🌱"):
            st.write(content)

    user_input = st.chat_input(t("Type your question...", "अपना सवाल लिखें..."))
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user", avatar="🧑‍🌾"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="🌱"):
            with st.spinner("..."):
                try:
                    reply = st.session_state.chatbot_session.send_message(user_input)
                except ChatbotError as exc:
                    reply = t(f"Sorry, something went wrong: {exc}", f"माफ करें, कुछ गलत हुआ: {exc}")
            st.write(reply)
        st.session_state.chat_history.append(("assistant", reply))


# ---------------------------------------------------------------------------
# Screen 4 — Weather Advisory
# ---------------------------------------------------------------------------
def render_weather() -> None:
    st.header(f"🌤️ {t('Weather Advisory', 'मौसम सलाह')}")
    city = st.text_input(t("Enter your city", "अपना शहर दर्ज करें"), value="Pune,IN")

    if st.button(t("Get Forecast", "पूर्वानुमान पाएं")):
        with st.spinner(t("Fetching forecast...", "पूर्वानुमान लाया जा रहा है...")):
            try:
                forecasts = fetch_five_day_forecast(city)
            except WeatherServiceError as exc:
                st.error(str(exc))
                return

        spray_days = recommend_spray_days(forecasts)
        if spray_days:
            st.success(f"{t('Best spray days this week', 'इस हफ्ते छिड़काव के सबसे अच्छे दिन')}: {', '.join(spray_days)}")
        else:
            st.warning(t("No ideal spray days in the next 5 days — high risk or rain throughout.", "अगले 5 दिनों में छिड़काव के लिए कोई आदर्श दिन नहीं — उच्च जोखिम या बारिश।"))

        risk_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        cols = st.columns(len(forecasts))
        for col, f in zip(cols, forecasts):
            with col:
                st.markdown(f"**{f.date}**")
                st.write(f"{risk_colors.get(f.disease_risk.value, '⚪')} {f.disease_risk.value.upper()}")
                st.write(f"{f.temp_min_c:.0f}–{f.temp_max_c:.0f}°C")
                st.write(f"💧 {f.humidity_pct:.0f}%")
                if f.rain_expected:
                    st.write(f"🌧️ {t('Rain expected', 'बारिश संभावित')}")


# ---------------------------------------------------------------------------
# Screen 5 — Disease Encyclopedia
# ---------------------------------------------------------------------------
def render_encyclopedia() -> None:
    st.header(f"📖 {t('Disease Encyclopedia', 'रोग विश्वकोश')}")
    disease_info = load_disease_info()

    if not disease_info:
        st.warning(t("Knowledge base not found.", "ज्ञान आधार नहीं मिला।"))
        return

    crops = sorted({info["crop"] for info in disease_info.values()})
    selected_crop = st.selectbox(t("Filter by crop", "फसल के अनुसार फ़िल्टर करें"), ["All"] + crops)
    search_term = st.text_input(t("Search by symptom keyword", "लक्षण से खोजें"))

    for class_name, info in sorted(disease_info.items()):
        if selected_crop != "All" and info["crop"] != selected_crop:
            continue
        symptoms = info["symptoms_hi"] if LANG_CODE == "hindi" else info["symptoms_en"]
        if search_term and search_term.lower() not in symptoms.lower() and search_term.lower() not in info["disease_name_en"].lower():
            continue

        name = info["disease_name_hi"] if LANG_CODE == "hindi" else info["disease_name_en"]
        with st.expander(f"{info['crop']} — {name} (severity: {info['severity']})"):
            st.write(f"**{t('Pathogen', 'रोगजनक')}:** {info['pathogen']}")
            st.write(f"**{t('Symptoms', 'लक्षण')}:** {symptoms}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if screen == "🏠 Home":
    render_home()
elif screen == "📸 Disease Detection":
    render_detection()
elif screen == "💬 Kisan Saathi Chatbot":
    render_chatbot()
elif screen == "🌤️ Weather Advisory":
    render_weather()
elif screen == "📖 Disease Encyclopedia":
    render_encyclopedia()
