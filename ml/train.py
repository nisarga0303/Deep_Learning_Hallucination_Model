import json
import os
import pickle

import numpy as np
import open_clip
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

try:
    from .dataset import MedicalVQADataset, build_convnext_transform
    from .knowledge_graph import MedicalKnowledgeGraph
    from .models import BiomedCLIPHybridModel, ConvNeXtTraditionalModel
except ImportError:
    from dataset import MedicalVQADataset, build_convnext_transform
    from knowledge_graph import MedicalKnowledgeGraph
    from models import BiomedCLIPHybridModel, ConvNeXtTraditionalModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
TRADITIONAL_MODEL_PATH = os.path.join(BASE_DIR, "traditional_model.pth")
HYBRID_MODEL_PATH = os.path.join(BASE_DIR, "hybrid_model.pth")
VOCABS_PATH = os.path.join(BASE_DIR, "vocabs.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
KNOWLEDGE_GRAPH_PATH = os.path.join(BASE_DIR, "knowledge_graph.pkl")
BIOMEDCLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_with_strings(batch):
    images = torch.stack([item["image"] for item in batch])
    result = {
        "image": images,
        "question": [item["question"] for item in batch],
        "raw_question": [item["raw_question"] for item in batch],
        "question_intent": [item["question_intent"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "organ": [item["organ"] for item in batch],
        "modality": [item["modality"] for item in batch],
        "answer_idx": torch.tensor([item["answer_idx"] for item in batch], dtype=torch.long),
        "organ_idx": torch.tensor([item["organ_idx"] for item in batch], dtype=torch.long),
        "modality_idx": torch.tensor([item["modality_idx"] for item in batch], dtype=torch.long),
        "intent_idx": torch.tensor([item["intent_idx"] for item in batch], dtype=torch.long),
    }
    return result


def trainable_state_dict(model):
    return {
        key: value.cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("clip_model.")
    }


def evaluate_traditional(model, dataloader, device):
    model.eval()
    organ_true, organ_pred = [], []
    modality_true, modality_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            organ_logits, modality_logits = model(images)
            organ_pred.extend(organ_logits.argmax(dim=1).cpu().tolist())
            modality_pred.extend(modality_logits.argmax(dim=1).cpu().tolist())
            organ_true.extend(batch["organ_idx"].tolist())
            modality_true.extend(batch["modality_idx"].tolist())

    return {
        "organ_accuracy": accuracy_score(organ_true, organ_pred),
        "modality_accuracy": accuracy_score(modality_true, modality_pred),
    }


def evaluate_hybrid(model, dataloader, tokenizer, knowledge_graph, device, use_ground_truth_graph=False):
    model.eval()
    answer_true, answer_pred = [], []
    organ_true, organ_pred = [], []
    modality_true, modality_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            text_tokens = tokenizer(batch["question"]).to(device)
            image_features, text_features, organ_logits, modality_logits = model.encode_base(images, text_tokens)

            organ_indices = organ_logits.argmax(dim=1).cpu().tolist()
            modality_indices = modality_logits.argmax(dim=1).cpu().tolist()

            if use_ground_truth_graph:
                organ_labels = batch["organ"]
                modality_labels = batch["modality"]
            else:
                organ_labels = [dataloader.dataset.organs[idx] for idx in organ_indices]
                modality_labels = [dataloader.dataset.modalities[idx] for idx in modality_indices]

            graph_nodes, graph_adj = knowledge_graph.build_batch_subgraphs(
                intents=batch["question_intent"],
                organs=organ_labels,
                modalities=modality_labels,
            )

            answer_logits = model.answer_with_graph(
                image_features,
                text_features,
                graph_nodes.to(device),
                graph_adj.to(device),
            )

            answer_pred.extend(answer_logits.argmax(dim=1).cpu().tolist())
            organ_pred.extend(organ_indices)
            modality_pred.extend(modality_indices)
            answer_true.extend(batch["answer_idx"].tolist())
            organ_true.extend(batch["organ_idx"].tolist())
            modality_true.extend(batch["modality_idx"].tolist())

    return {
        "answer_accuracy": accuracy_score(answer_true, answer_pred),
        "organ_accuracy": accuracy_score(organ_true, organ_pred),
        "modality_accuracy": accuracy_score(modality_true, modality_pred),
    }


def train_traditional(train_dataset, test_dataset, device):
    print("--- Training Traditional ConvNeXt-Tiny Model ---")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_with_strings)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_with_strings)

    model = ConvNeXtTraditionalModel(
        num_organs=len(train_dataset.organs),
        num_modalities=len(train_dataset.modalities),
    ).to(device)

    organ_counts = np.bincount(
        [train_dataset.organ2idx[sample["organ"]] for sample in train_dataset.samples],
        minlength=len(train_dataset.organs),
    )
    modality_counts = np.bincount(
        [train_dataset.modality2idx[sample["modality"]] for sample in train_dataset.samples],
        minlength=len(train_dataset.modalities),
    )

    organ_weights = torch.tensor(1.0 / np.maximum(organ_counts, 1), dtype=torch.float32, device=device)
    modality_weights = torch.tensor(1.0 / np.maximum(modality_counts, 1), dtype=torch.float32, device=device)
    organ_criterion = nn.CrossEntropyLoss(weight=organ_weights)
    modality_criterion = nn.CrossEntropyLoss(weight=modality_weights)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    best_metrics = {"organ_accuracy": 0.0, "modality_accuracy": 0.0}
    best_state = None

    for epoch in range(6):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            organ_targets = batch["organ_idx"].to(device)
            modality_targets = batch["modality_idx"].to(device)

            optimizer.zero_grad()
            organ_logits, modality_logits = model(images)
            loss = organ_criterion(organ_logits, organ_targets) + modality_criterion(modality_logits, modality_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate_traditional(model, test_loader, device)
        print(
            f"Traditional Epoch {epoch + 1}/6 | Loss: {running_loss / len(train_loader):.4f} | "
            f"Organ Acc: {metrics['organ_accuracy']:.4f} | Modality Acc: {metrics['modality_accuracy']:.4f}"
        )
        if metrics["organ_accuracy"] + metrics["modality_accuracy"] >= best_metrics["organ_accuracy"] + best_metrics["modality_accuracy"]:
            best_metrics = metrics
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    torch.save(best_state or model.state_dict(), TRADITIONAL_MODEL_PATH)
    return best_metrics


def train_hybrid(train_dataset, test_dataset, device):
    print("--- Training Hybrid BiomedCLIP + RotatE + GAT Model ---")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_with_strings)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_with_strings)

    clip_model, _clip_preprocess_train, _clip_preprocess_val = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL_NAME)
    clip_model = clip_model.to(device)

    knowledge_graph = MedicalKnowledgeGraph(train_dataset.samples)
    model = BiomedCLIPHybridModel(
        clip_model=clip_model,
        num_answers=len(train_dataset.answers),
        num_organs=len(train_dataset.organs),
        num_modalities=len(train_dataset.modalities),
        num_entities=len(knowledge_graph.idx2entity),
        num_relations=len(knowledge_graph.idx2relation),
    ).to(device)

    organ_counts = np.bincount(
        [train_dataset.organ2idx[sample["organ"]] for sample in train_dataset.samples],
        minlength=len(train_dataset.organs),
    )
    modality_counts = np.bincount(
        [train_dataset.modality2idx[sample["modality"]] for sample in train_dataset.samples],
        minlength=len(train_dataset.modalities),
    )

    organ_weights = torch.tensor(1.0 / np.maximum(organ_counts, 1), dtype=torch.float32, device=device)
    modality_weights = torch.tensor(1.0 / np.maximum(modality_counts, 1), dtype=torch.float32, device=device)
    answer_criterion = nn.CrossEntropyLoss()
    organ_criterion = nn.CrossEntropyLoss(weight=organ_weights)
    modality_criterion = nn.CrossEntropyLoss(weight=modality_weights)
    optimizer = optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4, weight_decay=1e-4)

    best_metrics = {"answer_accuracy": 0.0, "organ_accuracy": 0.0, "modality_accuracy": 0.0}
    best_state = None

    for epoch in range(4):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            text_tokens = tokenizer(batch["question"]).to(device)
            answer_targets = batch["answer_idx"].to(device)
            organ_targets = batch["organ_idx"].to(device)
            modality_targets = batch["modality_idx"].to(device)
            graph_nodes, graph_adj = knowledge_graph.build_batch_subgraphs(
                intents=batch["question_intent"],
                organs=batch["organ"],
                modalities=batch["modality"],
            )

            optimizer.zero_grad()
            image_features, text_features, organ_logits, modality_logits = model.encode_base(images, text_tokens)
            answer_logits = model.answer_with_graph(
                image_features,
                text_features,
                graph_nodes.to(device),
                graph_adj.to(device),
            )

            classification_loss = (
                0.8 * answer_criterion(answer_logits, answer_targets) +
                0.6 * organ_criterion(organ_logits, organ_targets) +
                0.6 * modality_criterion(modality_logits, modality_targets)
            )

            kg_sample = knowledge_graph.sample_triples(batch_size=128)
            if kg_sample is not None:
                heads, relations, tails, negative_tails = [tensor.to(device) for tensor in kg_sample]
                positive_scores = model.kg_encoder.score_triples(heads, relations, tails)
                negative_scores = model.kg_encoder.score_triples(heads, relations, negative_tails)
                kg_loss = torch.relu(1.0 - positive_scores + negative_scores).mean()
            else:
                kg_loss = torch.tensor(0.0, device=device)

            loss = classification_loss + 0.1 * kg_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate_hybrid(model, test_loader, tokenizer, knowledge_graph, device)
        print(
            f"Hybrid Epoch {epoch + 1}/4 | Loss: {running_loss / len(train_loader):.4f} | "
            f"Answer Acc: {metrics['answer_accuracy']:.4f} | Organ Acc: {metrics['organ_accuracy']:.4f} | "
            f"Modality Acc: {metrics['modality_accuracy']:.4f}"
        )
        if (
            metrics["answer_accuracy"] + metrics["organ_accuracy"] + metrics["modality_accuracy"]
            >= best_metrics["answer_accuracy"] + best_metrics["organ_accuracy"] + best_metrics["modality_accuracy"]
        ):
            best_metrics = metrics
            best_state = trainable_state_dict(model)

    torch.save(best_state or trainable_state_dict(model), HYBRID_MODEL_PATH)
    with open(KNOWLEDGE_GRAPH_PATH, "wb") as handle:
        pickle.dump(knowledge_graph, handle)

    return best_metrics


def extract_traditional_features(image_source):
    if isinstance(image_source, Image.Image):
        image = image_source.convert("RGB").resize((224, 224))
    else:
        image = Image.open(image_source).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)


def save_vocabs(train_dataset):
    payload = {
        "answers": train_dataset.answers,
        "organs": train_dataset.organs,
        "modalities": train_dataset.modalities,
        "intents": train_dataset.intents,
        "biomedclip_model_name": BIOMEDCLIP_MODEL_NAME,
    }
    with open(VOCABS_PATH, "wb") as handle:
        pickle.dump(payload, handle)


def main():
    device = choose_device()
    print(f"Using device: {device}")

    convnext_train = MedicalVQADataset(DATA_DIR, split="train", transform=build_convnext_transform("train"))
    convnext_test = MedicalVQADataset(DATA_DIR, split="test", transform=build_convnext_transform("test"), augment_templates=False)

    print(f"Train samples: {len(convnext_train)}")
    print(f"Test samples: {len(convnext_test)}")
    save_vocabs(convnext_train)

    traditional_metrics = train_traditional(convnext_train, convnext_test, device)

    biomedclip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_NAME)
    del biomedclip_model
    hybrid_train = MedicalVQADataset(DATA_DIR, split="train", transform=preprocess_train)
    hybrid_test = MedicalVQADataset(DATA_DIR, split="test", transform=preprocess_val, augment_templates=False)
    hybrid_metrics = train_hybrid(hybrid_train, hybrid_test, device)

    metrics = {
        "traditional": traditional_metrics,
        "hybrid": hybrid_metrics,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
