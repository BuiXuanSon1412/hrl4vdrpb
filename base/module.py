import torch
import torch.nn as nn
import numpy as np


class CustomerEncoder(nn.Module):
    """Encode customer features using attention mechanism"""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: (batch, num_customers, input_dim)
        x = self.input_proj(x)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class HighLevelPolicy(nn.Module):
    """
    High-level policy: decides which vehicle to plan next and strategic decisions
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, num_vehicles: int = 3):
        super().__init__()

        self.num_vehicles = num_vehicles

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Vehicle selection head
        self.vehicle_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_vehicles),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state):
        features = self.encoder(state)

        # Vehicle selection logits
        vehicle_logits = self.vehicle_selector(features)

        # State value
        value = self.value_head(features)

        return vehicle_logits, value


class LowLevelPolicy(nn.Module):
    """
    Low-level policy: decides which customer to visit next and whether to use drone
    Uses pointer network architecture for auto-regressive generation
    """

    def __init__(
        self,
        customer_dim: int = 7,
        vehicle_dim: int = 6,
        hidden_dim: int = 128,
        num_customers: int = 20,
    ):
        super().__init__()

        self.num_customers = num_customers
        self.hidden_dim = hidden_dim

        # Customer encoder
        self.customer_encoder = CustomerEncoder(customer_dim, hidden_dim)

        # Vehicle/context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(vehicle_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pointer mechanism for customer selection
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        self.pointing_scale = np.sqrt(hidden_dim)

        # Drone usage decision
        self.drone_decision = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Use drone or not
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, customer_features, vehicle_context, mask=None):
        """
        Args:
            customer_features: (batch, num_customers, customer_dim)
            vehicle_context: (batch, vehicle_dim)
            mask: (batch, num_customers) - mask for served customers
        """
        # Encode customers
        customer_encoded = self.customer_encoder(
            customer_features
        )  # (batch, num_customers, hidden)

        # Encode context
        context_encoded = self.context_encoder(vehicle_context)  # (batch, hidden)

        # Pointer mechanism for customer selection
        query = self.pointer_query(context_encoded).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.pointer_key(customer_encoded)  # (batch, num_customers, hidden)

        # Calculate attention scores
        logits = (
            torch.matmul(query, keys.transpose(1, 2)) / self.pointing_scale
        )  # (batch, 1, num_customers)
        logits = logits.squeeze(1)  # (batch, num_customers)

        # Apply mask
        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))

        # Drone usage decision
        # Aggregate customer info with context
        customer_agg = customer_encoded.mean(dim=1)  # (batch, hidden)
        drone_input = torch.cat([customer_agg, context_encoded], dim=1)
        drone_logits = self.drone_decision(drone_input)  # (batch, 2)

        # Value estimation
        value = self.value_head(context_encoded)

        return logits, drone_logits, value
