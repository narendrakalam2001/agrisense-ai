"""FastAPI serving layer for AgriSense AI.

Endpoints:
    POST /detect              — image upload -> disease detection + treatment plan
    POST /chat                — chatbot message -> Kisan Saathi reply
    GET  /disease/{name}      — disease encyclopedia lookup by class name
    GET  /health              — liveness/readiness probe

Run locally:
    uvicorn serving.api:app --reload --port 8000

Run in Docker: see Dockerfile (uses the same command, no --reload).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import KNOWLEDGE_BASE_DIR, MODEL_H5_PATH, configure_logging
from src.chatbot import ChatbotError, KisanSaathiChatbot
from src.llm_advisor import LLMAdvisorError, build_fallback_explanation, explain_treatment_plan
from src.rag_engine import RAGEngine, RAGEngineError
from services.detection_service import DetectionService, DetectionServiceError

logger = configure_logging(__name__, log_filename="api.log")

app = FastAPI(
    title="AgriSense AI API",
    description="Intelligent Crop Disease Detection + Treatment Advisory Platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed Streamlit origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons — loaded once at process startup, reused across requests.
# Chatbot sessions are per-conversation-id so multiple farmers can chat
# concurrently without sharing memory.
# ---------------------------------------------------------------------------
_detection_service: Optional[DetectionService] = None
_rag_engine: Optional[RAGEngine] = None
_chatbot_sessions: dict[str, KisanSaathiChatbot] = {}


@app.on_event("startup")
def load_models() -> None:
    """Load the CV model and RAG engine once when the API process starts.

    Failures here are logged but do not crash the process — ``/health``
    reports degraded status and endpoints that need the missing component
    return a 503 instead of the whole API going down.
    """
    global _detection_service, _rag_engine

    try:
        _detection_service = DetectionService()
    except DetectionServiceError as exc:
        logger.error("Detection service failed to load: %s", exc)

    try:
        _rag_engine = RAGEngine()
    except Exception as exc:  # noqa: BLE001 - keep API up even if RAG init fails
        logger.error("RAG engine failed to load: %s", exc)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------
class ClassPredictionResponse(BaseModel):
    class_name: str
    confidence: float


class DetectResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_3: list[ClassPredictionResponse]
    gradcam_image_base64: Optional[str] = None
    treatment_plan: dict
    explanation_text: str


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., description="Stable ID for this farmer's chat session.")
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    rag_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness probe for Docker/Render/K8s health checks."""
    model_loaded = _detection_service is not None
    rag_loaded = _rag_engine is not None
    status = "ok" if (model_loaded and rag_loaded) else "degraded"
    return HealthResponse(status=status, model_loaded=model_loaded, rag_loaded=rag_loaded)


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...), language: str = "hinglish") -> DetectResponse:
    """Detect crop disease from an uploaded leaf image and return a treatment plan.

    Args:
        file: Uploaded JPG/PNG image of a crop leaf.
        language: Explanation language — ``"hindi"``, ``"english"``, or
            ``"hinglish"`` (default).

    Returns:
        Full detection + treatment response.

    Raises:
        HTTPException: 503 if the model isn't loaded, 400 for a bad image.
    """
    if _detection_service is None:
        raise HTTPException(status_code=503, detail="Detection model is not loaded.")

    image_bytes = await file.read()
    try:
        result = _detection_service.detect(image_bytes)
    except DetectionServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    treatment_plan_dict: dict = {}
    explanation_text = ""

    if _rag_engine is not None:
        try:
            plan = _rag_engine.get_treatment_plan(result.predicted_class)
            treatment_plan_dict = plan.to_dict()
            try:
                explained = explain_treatment_plan(plan, language=language)
                explanation_text = explained.explanation_text
            except LLMAdvisorError as exc:
                logger.warning("LLM explanation failed, using fallback: %s", exc)
                explanation_text = build_fallback_explanation(plan, language="english").explanation_text
        except RAGEngineError as exc:
            logger.warning("No treatment plan found for '%s': %s", result.predicted_class, exc)

    return DetectResponse(
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        top_3=[ClassPredictionResponse(**p.to_dict()) for p in result.top_3],
        gradcam_image_base64=result.gradcam_image_base64,
        treatment_plan=treatment_plan_dict,
        explanation_text=explanation_text,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the Kisan Saathi chatbot for a given conversation.

    Args:
        request: Conversation ID + farmer message.

    Returns:
        The chatbot's reply.

    Raises:
        HTTPException: 503 if the RAG engine isn't loaded, 500 on chat failure.
    """
    if _rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine is not loaded.")

    session = _chatbot_sessions.get(request.conversation_id)
    if session is None:
        session = KisanSaathiChatbot(rag_engine=_rag_engine)
        _chatbot_sessions[request.conversation_id] = session

    try:
        reply = session.send_message(request.message)
    except ChatbotError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(reply=reply)


@app.get("/disease/{class_name}")
def get_disease(class_name: str) -> dict:
    """Look up encyclopedia details for a single disease class.

    Args:
        class_name: Exact class name (matches ``CLASS_NAMES``), e.g.
            ``Tomato___Late_blight``.

    Returns:
        Combined disease_info + pesticide_database record.

    Raises:
        HTTPException: 404 if the class name isn't found.
    """
    info_path = KNOWLEDGE_BASE_DIR / "disease_info.json"
    pesticide_path = KNOWLEDGE_BASE_DIR / "pesticide_database.json"

    if not info_path.exists() or not pesticide_path.exists():
        raise HTTPException(status_code=503, detail="Knowledge base files not found.")

    disease_info = json.loads(info_path.read_text(encoding="utf-8"))
    pesticide_data = json.loads(pesticide_path.read_text(encoding="utf-8"))

    if class_name not in disease_info:
        raise HTTPException(status_code=404, detail=f"Unknown disease class '{class_name}'.")

    return {**disease_info[class_name], **pesticide_data.get(class_name, {})}


@app.get("/diseases")
def list_diseases() -> list[str]:
    """List all known disease class names (for the encyclopedia screen)."""
    info_path = KNOWLEDGE_BASE_DIR / "disease_info.json"
    if not info_path.exists():
        raise HTTPException(status_code=503, detail="Knowledge base files not found.")
    disease_info = json.loads(info_path.read_text(encoding="utf-8"))
    return sorted(disease_info.keys())
