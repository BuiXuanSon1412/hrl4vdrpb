import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reshape_by_heads(qkv, head_num):
    """Reshape tensor for multi-head attention"""
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    """Multi-head attention mechanism with masking support"""
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(
            batch_s, head_num, n, input_s
        )
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(
            batch_s, head_num, n, input_s
        )

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


class AddAndNormalization(nn.Module):
    """Add and normalize with layer normalization (more stable than instance norm)"""

    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, input1, input2):
        added = input1 + input2
        normalized = self.norm(added)
        return normalized


class FeedForward(nn.Module):
    """Feed-forward network"""

    def __init__(self, embedding_dim, ff_hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))


class EncoderLayer(nn.Module):
    """Enhanced encoder using attention"""

    def __init__(self, embedding_dim=128, head_num=8, qkv_dim=16, ff_hidden_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndNormalization(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ff_hidden_dim)
        self.add_n_normalization_2 = AddAndNormalization(embedding_dim)

    def forward(self, input1):
        """
        Args:
            input1: (batch, seq_len, embedding_dim)
        Returns:
            output: (batch, seq_len, embedding_dim)
        """
        q = reshape_by_heads(self.Wq(input1), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=self.head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3


class CrossAttentionLayer(nn.Module):
    """Cross-attention between two sequences (e.g., vehicle-customer)"""

    def __init__(self, embedding_dim=128, head_num=8, qkv_dim=16, ff_hidden_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndNormalization(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ff_hidden_dim)
        self.add_n_normalization_2 = AddAndNormalization(embedding_dim)

    def forward(self, query_input, key_value_input):
        """
        Args:
            query_input: (batch, query_len, embedding_dim)
            key_value_input: (batch, kv_len, embedding_dim)
        Returns:
            output: (batch, query_len, embedding_dim)
        """
        q = reshape_by_heads(self.Wq(query_input), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(key_value_input), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(key_value_input), head_num=self.head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.add_n_normalization_1(query_input, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3


class CustomerEncoder(nn.Module):
    """Enhanced customer encoder using attention layers"""

    def __init__(
        self,
        input_dim=7,
        embedding_dim=128,
        num_layers=3,
        head_num=8,
        qkv_dim=16,
        ff_hidden_dim=512,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Initial embedding
        self.input_proj = nn.Linear(input_dim, embedding_dim)

        # Stack of attention layers
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embedding_dim, head_num, qkv_dim, ff_hidden_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        Args:
            x: (batch, num_customers, input_dim)
        Returns:
            encoded: (batch, num_customers, embedding_dim)
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        return x


class HighLevelPolicy(nn.Module):
    """
    Enhanced high-level policy with attention-based state encoding
    Decides which vehicle to plan next
    """

    def __init__(
        self,
        state_dim,
        embedding_dim=128,
        num_vehicles=3,
        head_num=8,
        qkv_dim=16,
        ff_hidden_dim=512,
    ):
        super().__init__()
        self.num_vehicles = num_vehicles
        self.embedding_dim = embedding_dim

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Attention layer for state refinement
        self.attention_layer = EncoderLayer(
            embedding_dim, head_num, qkv_dim, ff_hidden_dim
        )

        # Vehicle selection head
        self.vehicle_selector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_vehicles),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim)
        Returns:
            vehicle_logits: (batch, num_vehicles)
            value: (batch, 1)
        """
        # Encode state
        features = self.state_encoder(state)  # (batch, embedding_dim)

        # Apply attention (expand to sequence for attention mechanism)
        features_expanded = features.unsqueeze(1)  # (batch, 1, embedding_dim)
        features_refined = self.attention_layer(
            features_expanded
        )  # (batch, 1, embedding_dim)
        features_refined = features_refined.squeeze(1)  # (batch, embedding_dim)

        # Vehicle selection
        vehicle_logits = self.vehicle_selector(features_refined)

        # Value estimation
        value = self.value_head(features_refined)

        return vehicle_logits, value


class LowLevelPolicy(nn.Module):
    """
    Enhanced low-level policy using CVRP-style pointer network
    Decides which customer to visit and whether to use drone
    """

    def __init__(
        self,
        customer_dim=7,
        vehicle_dim=6,
        embedding_dim=128,
        num_customers=20,
        num_layers=3,
        head_num=8,
        qkv_dim=16,
        ff_hidden_dim=512,
        logit_clipping=10.0,
    ):
        super().__init__()
        self.num_customers = num_customers
        self.embedding_dim = embedding_dim
        self.sqrt_embedding_dim = np.sqrt(embedding_dim)
        self.logit_clipping = logit_clipping

        # customer encoder
        self.customer_encoder = CustomerEncoder(
            customer_dim, embedding_dim, num_layers, head_num, qkv_dim, ff_hidden_dim
        )

        # Vehicle context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(vehicle_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Cross-attention between context and customers
        self.cross_attention = CrossAttentionLayer(
            embedding_dim, head_num, qkv_dim, ff_hidden_dim
        )

        # Pointer mechanism components (CVRP-style)
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.head_num = head_num
        self.qkv_dim = qkv_dim

        # Single-head key for final scoring
        self.single_head_key = None

        # Drone decision network
        self.drone_decision = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 2),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, customer_features, vehicle_context, mask=None):
        """
        Args:
            customer_features: (batch, num_customers, customer_dim)
            vehicle_context: (batch, vehicle_dim)
            mask: (batch, num_customers) - True for invalid customers
        Returns:
            customer_logits: (batch, num_customers)
            drone_logits: (batch, 2)
            value: (batch, 1)
        """
        batch_size = customer_features.size(0)

        # Encode customers with attention
        customer_encoded = self.customer_encoder(customer_features)
        # (batch, num_customers, embedding_dim)

        # Encode vehicle context
        context_encoded = self.context_encoder(vehicle_context)
        # (batch, embedding_dim)

        # Cross-attention: let context attend to customers
        context_expanded = context_encoded.unsqueeze(1)  # (batch, 1, embedding_dim)
        context_attended = self.cross_attention(context_expanded, customer_encoded)
        # (batch, 1, embedding_dim)
        context_attended = context_attended.squeeze(1)  # (batch, embedding_dim)

        # Pointer mechanism for customer selection (CVRP-style)
        # Multi-head attention
        q = reshape_by_heads(
            self.Wq(context_attended.unsqueeze(1)), head_num=self.head_num
        )  # (batch, head_num, 1, qkv_dim)

        k = reshape_by_heads(
            self.Wk(customer_encoded), head_num=self.head_num
        )  # (batch, head_num, num_customers, qkv_dim)

        v = reshape_by_heads(
            self.Wv(customer_encoded), head_num=self.head_num
        )  # (batch, head_num, num_customers, qkv_dim)

        # Create ninf_mask for attention
        ninf_mask = None
        if mask is not None:
            # Expand mask from (batch, num_customers) to (batch, 1, num_customers)
            ninf_mask = (mask.float() * float("-inf")).unsqueeze(1)

        out_concat = multi_head_attention(q, k, v, rank3_ninf_mask=ninf_mask)
        # (batch, 1, head_num * qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat).squeeze(1)
        # (batch, embedding_dim)

        # Single-head attention for probability calculation
        single_head_key = customer_encoded.transpose(1, 2)
        # (batch, embedding_dim, num_customers)

        score = torch.matmul(mh_atten_out.unsqueeze(1), single_head_key).squeeze(1)
        # (batch, num_customers)

        # Scale and clip logits (CVRP-style)
        score_scaled = score / self.sqrt_embedding_dim
        customer_logits = self.logit_clipping * torch.tanh(score_scaled)

        # Apply mask (use original 2D mask here)
        if mask is not None:
            customer_logits = customer_logits.masked_fill(mask, float("-inf"))

        # Drone decision
        customer_agg = customer_encoded.mean(dim=1)  # (batch, embedding_dim)
        drone_input = torch.cat([customer_agg, context_attended], dim=1)
        drone_logits = self.drone_decision(drone_input)  # (batch, 2)

        # Value estimation
        value = self.value_head(context_attended)

        return customer_logits, drone_logits, value


########################################
# FACTORY FUNCTIONS
########################################


def create_high_level_policy(
    state_dim,
    num_vehicles=3,
    embedding_dim=128,
    head_num=8,
    qkv_dim=16,
    ff_hidden_dim=512,
):
    """Factory function for creating enhanced high-level policy"""
    return HighLevelPolicy(
        state_dim=state_dim,
        embedding_dim=embedding_dim,
        num_vehicles=num_vehicles,
        head_num=head_num,
        qkv_dim=qkv_dim,
        ff_hidden_dim=ff_hidden_dim,
    )


def create_low_level_policy(
    customer_dim=7,
    vehicle_dim=6,
    num_customers=20,
    embedding_dim=128,
    num_layers=3,
    head_num=8,
    qkv_dim=16,
    ff_hidden_dim=512,
    logit_clipping=10.0,
):
    """Factory function for creating enhanced low-level policy"""
    return LowLevelPolicy(
        customer_dim=customer_dim,
        vehicle_dim=vehicle_dim,
        embedding_dim=embedding_dim,
        num_customers=num_customers,
        num_layers=num_layers,
        head_num=head_num,
        qkv_dim=qkv_dim,
        ff_hidden_dim=ff_hidden_dim,
        logit_clipping=logit_clipping,
    )
