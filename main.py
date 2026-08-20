from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from part3_support_agent.graph.workflow import support_graph
from part3_support_agent.tools.classify_product_image import (
    classify_product_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_PATH = PROJECT_ROOT / "frontend" / "index.html"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


app = FastAPI(title="Flipkart Order Intelligence")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


conversations = {}
conversation_order_features = {}


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    order_features: dict | None = None


@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH)


@app.post("/api/chat")
def chat(data: ChatRequest):
    message = data.message.strip()
    conversation_id = data.conversation_id

    if not message:
        return {
            "answer": "Please enter a message.",
            "source": "system",
            "confidence": 0.0,
        }

    if not conversation_id:
        conversation_id = str(uuid4())

    history = conversations.get(conversation_id, [])

    if data.order_features:
        conversation_order_features[conversation_id] = data.order_features

    order_features = conversation_order_features.get(
        conversation_id
    )

    state = {
        "query": message,
        "conversation_history": history,
    }

    if order_features:
        state["order_features"] = order_features

    message_lower = message.lower()

    if (
        "return risk" in message_lower
        or "return probability" in message_lower
    ) and not order_features:
        return {
            "conversation_id": conversation_id,
            "answer": (
                "Please provide the order details before checking "
                "return risk."
            ),
            "source": "return_risk",
            "confidence": 0.0,
        }

    result = support_graph.invoke(state)

    conversations[conversation_id] = result.get(
        "conversation_history",
        history,
    )

    return {
        "conversation_id": conversation_id,
        "answer": result.get("answer", ""),
        "source": result.get("source", ""),
        "confidence": result.get("confidence", 0.0),
    }


@app.post("/api/classify-image")
async def classify_image(file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "image.png").name
    file_name = f"{uuid4()}_{original_name}"
    image_path = UPLOAD_DIR / file_name

    content = await file.read()

    with open(image_path, "wb") as output_file:
        output_file.write(content)

    result = classify_product_image(str(image_path))

    return {
        "image_path": result["image_path"],
        "predicted_class": result["class"],
        "confidence": result["confidence"],
    }