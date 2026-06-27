import random
from collections import defaultdict

import torch


def make_entity_key(entity_type, value):
    return f"{entity_type}::{value}"


class MedicalKnowledgeGraph:
    def __init__(self, samples):
        self.entity2idx = {}
        self.idx2entity = []
        self.relation2idx = {}
        self.idx2relation = []
        self.entity_display = {}
        self.adjacency = defaultdict(set)
        self.triples = []

        for sample in samples:
            answer_key = make_entity_key("answer", sample["answer"])
            organ_key = make_entity_key("organ", sample["organ"])
            modality_key = make_entity_key("modality", sample["modality"])
            intent_key = make_entity_key("intent", sample["question_intent"])

            self._ensure_entity(answer_key, sample["answer"])
            self._ensure_entity(organ_key, sample["organ"])
            self._ensure_entity(modality_key, sample["modality"])
            self._ensure_entity(intent_key, sample["question_intent"])

            self._add_triple(answer_key, "located_in", organ_key)
            self._add_triple(answer_key, "captured_with", modality_key)
            self._add_triple(intent_key, "expects_answer", answer_key)
            self._add_triple(intent_key, "focuses_on", organ_key)
            self._add_triple(organ_key, "compatible_with", modality_key)

    def _ensure_entity(self, entity_key, display_value):
        if entity_key not in self.entity2idx:
            self.entity2idx[entity_key] = len(self.idx2entity)
            self.idx2entity.append(entity_key)
            self.entity_display[entity_key] = display_value
        return self.entity2idx[entity_key]

    def _ensure_relation(self, relation_name):
        if relation_name not in self.relation2idx:
            self.relation2idx[relation_name] = len(self.idx2relation)
            self.idx2relation.append(relation_name)
        return self.relation2idx[relation_name]

    def _add_triple(self, head_key, relation_name, tail_key):
        head_idx = self._ensure_entity(head_key, head_key.split("::", 1)[1])
        relation_idx = self._ensure_relation(relation_name)
        tail_idx = self._ensure_entity(tail_key, tail_key.split("::", 1)[1])
        triple = (head_idx, relation_idx, tail_idx)
        self.triples.append(triple)
        self.adjacency[head_idx].add(tail_idx)
        self.adjacency[tail_idx].add(head_idx)

    def entity_index(self, entity_type, value):
        return self.entity2idx.get(make_entity_key(entity_type, value))

    def build_subgraph(self, intent, organ=None, modality=None, top_k=12):
        seed_indices = []
        for entity_type, value in (("intent", intent), ("organ", organ), ("modality", modality)):
            if value is None:
                continue
            idx = self.entity_index(entity_type, value)
            if idx is not None:
                seed_indices.append(idx)

        if not seed_indices:
            seed_indices = [0]

        collected = []
        seen = set()
        queue = list(seed_indices)
        while queue and len(collected) < top_k:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            collected.append(current)
            for neighbor in sorted(self.adjacency[current]):
                if neighbor not in seen and len(collected) + len(queue) < top_k * 2:
                    queue.append(neighbor)

        node_count = len(collected)
        adjacency = torch.eye(node_count, dtype=torch.float32)
        for row, src in enumerate(collected):
            for col, dst in enumerate(collected):
                if row != col and dst in self.adjacency[src]:
                    adjacency[row, col] = 1.0

        return torch.tensor(collected, dtype=torch.long), adjacency

    def build_batch_subgraphs(self, intents, organs, modalities, top_k=12):
        node_index_tensors = []
        adjacency_tensors = []
        max_nodes = 1

        subgraphs = []
        for intent, organ, modality in zip(intents, organs, modalities):
            node_indices, adjacency = self.build_subgraph(intent, organ, modality, top_k=top_k)
            subgraphs.append((node_indices, adjacency))
            max_nodes = max(max_nodes, node_indices.size(0))

        for node_indices, adjacency in subgraphs:
            pad_nodes = max_nodes - node_indices.size(0)
            if pad_nodes > 0:
                node_indices = torch.cat([node_indices, node_indices.new_full((pad_nodes,), node_indices[0])])
                padded_adjacency = torch.zeros((max_nodes, max_nodes), dtype=torch.float32)
                padded_adjacency[:adjacency.size(0), :adjacency.size(1)] = adjacency
                padded_adjacency.fill_diagonal_(1.0)
                adjacency = padded_adjacency
            node_index_tensors.append(node_indices)
            adjacency_tensors.append(adjacency)

        return torch.stack(node_index_tensors), torch.stack(adjacency_tensors)

    def sample_triples(self, batch_size):
        if not self.triples:
            return None

        positives = random.sample(self.triples, k=min(batch_size, len(self.triples)))
        heads, relations, tails, negative_tails = [], [], [], []
        entity_count = len(self.idx2entity)
        for head_idx, relation_idx, tail_idx in positives:
            negative_tail = random.randrange(entity_count)
            while negative_tail == tail_idx:
                negative_tail = random.randrange(entity_count)
            heads.append(head_idx)
            relations.append(relation_idx)
            tails.append(tail_idx)
            negative_tails.append(negative_tail)

        return (
            torch.tensor(heads, dtype=torch.long),
            torch.tensor(relations, dtype=torch.long),
            torch.tensor(tails, dtype=torch.long),
            torch.tensor(negative_tails, dtype=torch.long),
        )
