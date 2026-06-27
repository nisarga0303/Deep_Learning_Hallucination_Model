import io
import os
import pickle
import threading
import hashlib
from dataclasses import dataclass

import open_clip
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../ml"))
from dataset import build_convnext_transform, canonicalize_question
from models import BiomedCLIPHybridModel, ConvNeXtTraditionalModel


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
ML_DIR = os.path.join(BASE_DIR, "../ml")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_state_lock = threading.Lock()


@dataclass
class AppState:
    vocabs: dict
    knowledge_graph: object
    traditional_model: torch.nn.Module
    hybrid_model: torch.nn.Module
    tokenizer: object
    preprocess_val: object
    traditional_transform: object


_state: AppState | None = None
_state_error: str | None = None


def _load_state():
    global _state, _state_error
    if _state is not None:
        return _state

    with _state_lock:
        if _state is not None:
            return _state

        try:
            with open(os.path.join(ML_DIR, "vocabs.pkl"), "rb") as handle:
                vocabs = pickle.load(handle)

            with open(os.path.join(ML_DIR, "knowledge_graph.pkl"), "rb") as handle:
                knowledge_graph = pickle.load(handle)

            answers = vocabs["answers"]
            organs = vocabs["organs"]
            modalities = vocabs["modalities"]
            biomedclip_model_name = vocabs["biomedclip_model_name"]

            traditional_model = ConvNeXtTraditionalModel(len(organs), len(modalities)).to(DEVICE)
            traditional_model.load_state_dict(torch.load(os.path.join(ML_DIR, "traditional_model.pth"), map_location=DEVICE))
            traditional_model.eval()

            clip_model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(biomedclip_model_name)
            clip_model = clip_model.to(DEVICE)
            hybrid_model = BiomedCLIPHybridModel(
                clip_model=clip_model,
                num_answers=len(answers),
                num_organs=len(organs),
                num_modalities=len(modalities),
                num_entities=len(knowledge_graph.idx2entity),
                num_relations=len(knowledge_graph.idx2relation),
            ).to(DEVICE)
            hybrid_model.load_state_dict(torch.load(os.path.join(ML_DIR, "hybrid_model.pth"), map_location=DEVICE), strict=False)
            hybrid_model.eval()

            _state = AppState(
                vocabs=vocabs,
                knowledge_graph=knowledge_graph,
                traditional_model=traditional_model,
                hybrid_model=hybrid_model,
                tokenizer=open_clip.get_tokenizer(biomedclip_model_name),
                preprocess_val=preprocess_val,
                traditional_transform=build_convnext_transform("test"),
            )
            _state_error = None
            return _state
        except Exception as exc:
            _state_error = str(exc)
            raise


def choose_answer(question_intent, answer_label, organ_label, modality_label):
    if question_intent == "organ":
        return organ_label
    if question_intent == "modality":
        return modality_label
    return answer_label


def normalize_binary_disease_answer(answer_label):
    normalized = str(answer_label or "").strip()
    if not normalized:
        return "Unknown"

    lowered = normalized.lower()
    if lowered in {"no", "none", "normal", "0"}:
        return "No"
    if normalized.isdigit():
        return "No" if int(normalized) == 0 else "Yes"
    return "Yes"


def traditional_tta_logits(model, image_tensor):
    variants = [
        image_tensor,
        torch.flip(image_tensor, dims=[3]),
    ]
    organ_logits = []
    modality_logits = []
    with torch.no_grad():
        for variant in variants:
            organ_out, modality_out = model(variant)
            organ_logits.append(organ_out)
            modality_logits.append(modality_out)
    return torch.stack(organ_logits).mean(dim=0), torch.stack(modality_logits).mean(dim=0)


def apply_traditional_sanity_check(state, image, question_intent, organ_idx, modality_idx):
    organs = state.vocabs["organs"]
    modalities = state.vocabs["modalities"]

    predicted_organ = organs[organ_idx]
    predicted_modality = modalities[modality_idx]
    known_confusion_pairs = {
        ("Lung", "MRI"),
        ("Abdomen", "CT"),
    }

    if (predicted_organ, predicted_modality) not in known_confusion_pairs:
        return organ_idx, modality_idx

    image_tensor = state.preprocess_val(image).unsqueeze(0).to(DEVICE)
    canonical_question = (
        "Which part of the body does this image belong to?"
        if question_intent == "organ"
        else "What modality is used to take this image?"
    )
    text_tokens = state.tokenizer([canonical_question]).to(DEVICE)

    with torch.no_grad():
        _image_features, _text_features, hybrid_organ_logits, hybrid_modality_logits = state.hybrid_model.encode_base(
            image_tensor,
            text_tokens,
        )
        hybrid_organ_idx = hybrid_organ_logits.argmax(dim=1).item()
        hybrid_modality_idx = hybrid_modality_logits.argmax(dim=1).item()

    if modalities[hybrid_modality_idx] == predicted_modality and organs[hybrid_organ_idx] != predicted_organ:
        return hybrid_organ_idx, hybrid_modality_idx

    return organ_idx, modality_idx


@app.get("/health")
async def health():
    loaded = _state is not None
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models_loaded": loaded,
        "last_error": _state_error,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), question: str = Form(""), modelType: str = Form(...)):
    try:
        state = _load_state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}") from exc

    img_bytes = await file.read()
    image_sha256 = hashlib.sha256(img_bytes).hexdigest()
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    canonical_question, question_intent = canonicalize_question(question or "")

    organs = state.vocabs["organs"]
    modalities = state.vocabs["modalities"]
    answers = state.vocabs["answers"]

    if modelType == "traditional":
        image_tensor = state.traditional_transform(image).unsqueeze(0).to(DEVICE)
        organ_logits, modality_logits = traditional_tta_logits(state.traditional_model, image_tensor)
        organ_idx = organ_logits.argmax(dim=1).item()
        modality_idx = modality_logits.argmax(dim=1).item()
        organ_idx, modality_idx = apply_traditional_sanity_check(state, image, question_intent, organ_idx, modality_idx)
        organ_confidence = F.softmax(organ_logits, dim=1)[0, organ_idx].item()
        modality_confidence = F.softmax(modality_logits, dim=1)[0, modality_idx].item()

        answer = choose_answer(question_intent, "", organs[organ_idx], modalities[modality_idx])
        return {
            "answer": answer,
            "organ": organs[organ_idx],
            "modality": modalities[modality_idx],
            "confidence": float((organ_confidence + modality_confidence) / 2.0),
            "risk": "LOW" if min(organ_confidence, modality_confidence) >= 0.65 else "ELEVATED",
            "evidence": "ConvNeXt-Tiny image-only baseline using pretrained visual features.",
            "debug": {
                "filename": file.filename,
                "image_sha256": image_sha256,
                "question_intent": question_intent,
                "model_type": modelType,
            },
        }

    image_tensor = state.preprocess_val(image).unsqueeze(0).to(DEVICE)
    text_tokens = state.tokenizer([canonical_question]).to(DEVICE)

    with torch.no_grad():
        image_features, text_features, organ_logits, modality_logits = state.hybrid_model.encode_base(image_tensor, text_tokens)
        organ_idx = organ_logits.argmax(dim=1).item()
        modality_idx = modality_logits.argmax(dim=1).item()
        graph_nodes, graph_adjacency = state.knowledge_graph.build_batch_subgraphs(
            intents=[question_intent],
            organs=[organs[organ_idx]],
            modalities=[modalities[modality_idx]],
        )
        answer_logits = state.hybrid_model.answer_with_graph(
            image_features,
            text_features,
            graph_nodes.to(DEVICE),
            graph_adjacency.to(DEVICE),
        )
        answer_idx = answer_logits.argmax(dim=1).item()

        answer_confidence = F.softmax(answer_logits, dim=1)[0, answer_idx].item()
        organ_confidence = F.softmax(organ_logits, dim=1)[0, organ_idx].item()
        modality_confidence = F.softmax(modality_logits, dim=1)[0, modality_idx].item()

    answer = choose_answer(question_intent, answers[answer_idx], organs[organ_idx], modalities[modality_idx])
    if question_intent == "abnormality_presence":
        answer = normalize_binary_disease_answer(answers[answer_idx])
    return {
        "answer": answer,
        "organ": organs[organ_idx],
        "modality": modalities[modality_idx],
        "confidence": float((answer_confidence + organ_confidence + modality_confidence) / 3.0),
        "risk": "LOW" if min(answer_confidence, organ_confidence, modality_confidence) >= 0.65 else "ELEVATED",
        "evidence": "BiomedCLIP encoder fused with RotatE knowledge embeddings and GAT-based subgraph reasoning.",
        "debug": {
            "filename": file.filename,
            "image_sha256": image_sha256,
            "question_intent": question_intent,
            "model_type": modelType,
        },
    }
