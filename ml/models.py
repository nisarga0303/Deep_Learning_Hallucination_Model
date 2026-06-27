import math

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtTraditionalModel(nn.Module):
    def __init__(self, num_organs, num_modalities):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        backbone = convnext_tiny(weights=weights)
        in_features = backbone.classifier[2].in_features
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(0.2)
        self.organ_head = nn.Linear(in_features, num_organs)
        self.modality_head = nn.Linear(in_features, num_modalities)

    def forward(self, images):
        features = self.features(images)
        features = self.avgpool(features)
        features = self.flatten(features)
        features = self.dropout(features)
        organ_logits = self.organ_head(features)
        modality_logits = self.modality_head(features)
        return organ_logits, modality_logits


class RotatEEncoder(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)
        self.relation_phase = nn.Embedding(num_relations, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_re.weight)
        nn.init.xavier_uniform_(self.entity_im.weight)
        nn.init.uniform_(self.relation_phase.weight, a=-math.pi, b=math.pi)

    def entity_features(self, entity_indices):
        real = self.entity_re(entity_indices)
        imag = self.entity_im(entity_indices)
        return self.output_projection(torch.cat([real, imag], dim=-1))

    def score_triples(self, head_indices, relation_indices, tail_indices):
        head_re = self.entity_re(head_indices)
        head_im = self.entity_im(head_indices)
        tail_re = self.entity_re(tail_indices)
        tail_im = self.entity_im(tail_indices)

        phase = self.relation_phase(relation_indices)
        relation_re = torch.cos(phase)
        relation_im = torch.sin(phase)

        rotated_re = head_re * relation_re - head_im * relation_im
        rotated_im = head_re * relation_im + head_im * relation_re
        score = (rotated_re - tail_re).pow(2) + (rotated_im - tail_im).pow(2)
        return -score.sum(dim=-1)


class QuestionGuidedGAT(nn.Module):
    def __init__(self, node_dim, query_dim, hidden_dim):
        super().__init__()
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features, adjacency, query_features):
        projected_nodes = self.node_projection(node_features)
        projected_query = self.query_projection(query_features).unsqueeze(1)
        guided_nodes = projected_nodes + projected_query

        scores = torch.matmul(guided_nodes, projected_nodes.transpose(1, 2)) / math.sqrt(projected_nodes.size(-1))
        scores = scores.masked_fill(adjacency <= 0, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attended_nodes = torch.matmul(attention, projected_nodes)
        pooled = attended_nodes.mean(dim=1)
        return self.output_projection(pooled)


class BiomedCLIPHybridModel(nn.Module):
    def __init__(self, clip_model, num_answers, num_organs, num_modalities, num_entities, num_relations):
        super().__init__()
        self.clip_model = clip_model
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        if hasattr(self.clip_model, "text_projection") and hasattr(self.clip_model.text_projection, "shape"):
            clip_dim = self.clip_model.text_projection.shape[-1]
        else:
            clip_dim = 512

        hidden_dim = 256
        self.image_projection = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.organ_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_organs),
        )
        self.modality_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_modalities),
        )

        self.kg_encoder = RotatEEncoder(num_entities, num_relations, embedding_dim=128)
        self.graph_attention = QuestionGuidedGAT(node_dim=128, query_dim=hidden_dim * 2, hidden_dim=128)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 128, 256),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.answer_head = nn.Linear(256, num_answers)

    def encode_base(self, images, text_tokens):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images, normalize=True)
            text_features = self.clip_model.encode_text(text_tokens, normalize=True)

        image_features = self.image_projection(image_features.float())
        text_features = self.text_projection(text_features.float())
        organ_logits = self.organ_head(image_features)
        modality_logits = self.modality_head(image_features)
        return image_features, text_features, organ_logits, modality_logits

    def answer_with_graph(self, image_features, text_features, graph_node_indices, graph_adjacency):
        node_features = self.kg_encoder.entity_features(graph_node_indices)
        query_features = torch.cat([image_features, text_features], dim=1)
        kg_context = self.graph_attention(node_features, graph_adjacency, query_features)
        fused = self.fusion(torch.cat([image_features, text_features, kg_context], dim=1))
        return self.answer_head(fused)
