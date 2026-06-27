import json
import os
import random
from collections import Counter

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


SLAKE_IMAGE_MEAN = [0.485, 0.456, 0.406]
SLAKE_IMAGE_STD = [0.229, 0.224, 0.225]

QUESTION_TEMPLATES = {
    "organ": [
        "Which part of the body does this image belong to?",
        "What is the main organ in the image?",
        "Which organ is shown in this medical image?",
    ],
    "modality": [
        "What modality is used to take this image?",
        "What type of scan is this image?",
        "Which imaging modality produced this image?",
    ],
    "weighting": [
        "What is the mr weighting in this image?",
        "Which MR weighting is shown in this image?",
    ],
    "plane": [
        "What is the scanning plane of this image?",
        "Which scanning plane is shown in this image?",
    ],
    "abnormality_presence": [
        "Are there abnormalities in this image?",
        "Does this image show any abnormality?",
        "Does this image contain any disease?",
    ],
    "disease_list": [
        "What diseases are included in the picture?",
        "What disease is shown in this image?",
    ],
    "abnormality_location": [
        "Where is the abnormality located?",
        "Where is the abnormality in this image?",
    ],
    "count": [
        "How many structures are visible in this image?",
        "How many relevant structures are shown in this image?",
    ],
    "binary_organ": [
        "Does the picture contain this organ?",
        "Is this organ present in the image?",
    ],
    "open": [
        "Describe the most relevant medical finding in this image.",
        "What is the correct answer for this medical image question?",
    ],
}


def normalize_label(value):
    return str(value or "Unknown").replace("_", " ").strip() or "Unknown"


def normalize_organ(value):
    organ = normalize_label(value)
    organ_lower = organ.lower()
    if organ_lower.startswith("brain"):
        return "Brain"
    if organ_lower == "pelvic cavity":
        return "Pelvis"
    if organ_lower in {"chest heart", "chest mediastinal"}:
        return "Chest"
    if organ_lower == "chest lung":
        return "Lung"
    return organ


def normalize_modality(value):
    modality = normalize_label(value).upper()
    alias_map = {
        "XRAY": "X-Ray",
        "X RAY": "X-Ray",
        "X-RAY": "X-Ray",
        "CT": "CT",
        "MRI": "MRI",
    }
    return alias_map.get(modality, modality.title())


def detect_question_intent(question):
    normalized = str(question or "").strip().lower()
    if not normalized:
        return "open"

    if "modality" in normalized or "scan type" in normalized or "imaging modality" in normalized:
        return "modality"
    if "weighting" in normalized:
        return "weighting"
    if "plane" in normalized:
        return "plane"
    if "abnormalit" in normalized and "where" in normalized:
        return "abnormality_location"
    if "what disease" in normalized or "what diseases" in normalized or "which disease" in normalized:
        return "disease_list"
    if "contain any disease" in normalized or "does this contain any disease" in normalized:
        return "abnormality_presence"
    if "abnormalit" in normalized or "look abnormal" in normalized:
        return "abnormality_presence"
    if "how many" in normalized or "count" in normalized:
        return "count"
    if "does the picture contain" in normalized or "is the" in normalized and "present" in normalized:
        return "binary_organ"
    if "organ" in normalized or "body" in normalized or "part of the body" in normalized:
        return "organ"
    return "open"


def canonicalize_question(question):
    intent = detect_question_intent(question)
    return QUESTION_TEMPLATES[intent][0], intent


def augment_question(question, split):
    canonical, intent = canonicalize_question(question)
    if split != "train":
        return canonical, intent
    templates = QUESTION_TEMPLATES.get(intent, QUESTION_TEMPLATES["open"])
    return random.choice(templates), intent


def build_convnext_transform(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop((224, 224), scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=SLAKE_IMAGE_MEAN, std=SLAKE_IMAGE_STD),
        ])

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=SLAKE_IMAGE_MEAN, std=SLAKE_IMAGE_STD),
    ])


class MedicalVQADataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None, seed=42, augment_templates=True):
        self.data_dir = data_dir
        self.split = split
        self.seed = seed
        self.transform = transform
        self.augment_templates = augment_templates

        self.samples = []
        self.answers = set()
        self.organs = set()
        self.modalities = set()
        self.intents = set()
        self.question_counter = Counter()

        grouped_samples = self._load_grouped_samples()

        self.answers = sorted(self.answers)
        self.organs = sorted(self.organs)
        self.modalities = sorted(self.modalities)
        self.intents = sorted(self.intents)

        self.answer2idx = {answer: idx for idx, answer in enumerate(self.answers)}
        self.organ2idx = {organ: idx for idx, organ in enumerate(self.organs)}
        self.modality2idx = {modality: idx for idx, modality in enumerate(self.modalities)}
        self.intent2idx = {intent: idx for idx, intent in enumerate(self.intents)}

        rng = random.Random(self.seed)
        rng.shuffle(grouped_samples)

        split_idx = int(len(grouped_samples) * 0.8)
        if split == "train":
            selected_groups = grouped_samples[:split_idx]
        else:
            selected_groups = grouped_samples[split_idx:]

        self.samples = [sample for group in selected_groups for sample in group]

    def _load_grouped_samples(self):
        imgs_dir = os.path.join(self.data_dir, "imgs")
        grouped_samples = []
        if not os.path.exists(imgs_dir):
            return grouped_samples

        folders = [folder for folder in os.listdir(imgs_dir) if folder.startswith("xmlab")]
        for folder in folders:
            folder_path = os.path.join(imgs_dir, folder)
            json_path = os.path.join(folder_path, "question.json")
            img_path = os.path.join(folder_path, "source.jpg")
            if not os.path.exists(json_path) or not os.path.exists(img_path):
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as handle:
                    questions = json.load(handle)
            except Exception:
                continue

            folder_samples = []
            for question in questions:
                if question.get("q_lang") != "en":
                    continue

                raw_question = str(question.get("question", "")).strip()
                canonical_question, question_intent = canonicalize_question(raw_question)
                answer = normalize_label(question.get("answer"))
                organ = normalize_organ(question.get("location"))
                modality = normalize_modality(question.get("modality"))

                self.answers.add(answer)
                self.organs.add(organ)
                self.modalities.add(modality)
                self.intents.add(question_intent)
                self.question_counter[canonical_question.lower()] += 1

                folder_samples.append({
                    "img_path": img_path,
                    "question": raw_question,
                    "canonical_question": canonical_question,
                    "question_intent": question_intent,
                    "answer": answer,
                    "organ": organ,
                    "modality": modality,
                })

            if folder_samples:
                grouped_samples.append(folder_samples)

        return grouped_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample["img_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        question_text = sample["canonical_question"]
        if self.augment_templates:
            question_text, _ = augment_question(sample["question"], self.split)

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = image

        return {
            "image": image_tensor,
            "image_path": sample["img_path"],
            "question": question_text,
            "raw_question": sample["question"],
            "question_intent": sample["question_intent"],
            "answer": sample["answer"],
            "organ": sample["organ"],
            "modality": sample["modality"],
            "answer_idx": self.answer2idx[sample["answer"]],
            "organ_idx": self.organ2idx[sample["organ"]],
            "modality_idx": self.modality2idx[sample["modality"]],
            "intent_idx": self.intent2idx[sample["question_intent"]],
        }
