import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Optional, Tuple

from transformers.activations import ACT2FN


class FiLMLayer(nn.Module):
	def __init__(self, query_size: int, hidden_size: int):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(query_size, hidden_size * 2),
			nn.SiLU(),
			nn.Linear(hidden_size * 2, hidden_size * 2),
		)

	def forward(self, vis_emb: Tensor, query_emb: Tensor):
		"""
		vis_emb:   [B*num_patches, num_tokens, hidden]
		query_emb: [B, hidden]
		"""
		out        = self.mlp(query_emb)          # [B, 2*hidden]
		gamma, beta = out.chunk(2, dim=-1)        # [B, hidden]

		B, hidden  = gamma.shape
		total      = vis_emb.shape[0]            # B * num_patches
		repeat     = total // B                  # num_patches per sample

		# [B, hidden] → [B, 1, hidden] → [B*np, 1, hidden]
		gamma = gamma.unsqueeze(1).repeat_interleave(repeat, dim=0)
		beta  = beta.unsqueeze(1).repeat_interleave(repeat, dim=0)

		return gamma * vis_emb + beta


class ExpertFFN(nn.Module):
	def __init__(self, hidden_size: int, intermediate_size: int, activation: str = 'gelu'):
		super().__init__()
		if activation not in ACT2FN:
			raise ValueError(f'Unsupported activation: {activation}')
		self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
		self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)
		self.act = ACT2FN[activation]

	def forward(self, x: Tensor) -> Tensor:
		return self.w_down(self.act(self.w_up(x)))


class QueryConditionedRouter(nn.Module):
	def __init__(self, hidden_size: int, query_size: int, num_experts: int, top_k: int = 2):
		super().__init__()
		if num_experts <= 0:
			raise ValueError('num_experts must be > 0')
		if top_k <= 0:
			raise ValueError('top_k must be > 0')
		self.top_k = min(top_k, num_experts)
		self.gate = nn.Linear(hidden_size + query_size, num_experts, bias=False)

	def forward(self, vis_emb: Tensor, query_emb: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
		"""
		vis_emb:   [B*num_patches, num_tokens, hidden]
		query_emb: [B, hidden]
		"""
		B      = query_emb.shape[0]
		total  = vis_emb.shape[0]           # B * num_patches
		repeat = total // B                 # num_patches per sample
		_, num_tokens, _ = vis_emb.shape

		# [B, hidden] → [B, 1, hidden] → [B*np, 1, hidden] → [B*np, num_tokens, hidden]
		query_expanded = (
			query_emb
			.unsqueeze(1)                           # [B, 1, hidden]
			.repeat_interleave(repeat, dim=0)       # [B*np, 1, hidden]
			.expand(-1, num_tokens, -1)             # [B*np, num_tokens, hidden]
		)

		gate_input  = torch.cat([vis_emb, query_expanded], dim=-1)  # [B*np, num_tokens, hidden+query_size]
		gate_logits = self.gate(gate_input)                          # [B*np, num_tokens, num_experts]
		gate_scores = F.softmax(gate_logits, dim=-1)
		topk_scores, topk_idx = gate_scores.topk(self.top_k, dim=-1)
		return topk_scores, topk_idx, gate_scores, gate_logits


class QueryConditionedMoE(nn.Module):
	def __init__(
		self,
		hidden_size: int,
		query_size: int,
		num_experts: int = 4,
		num_shared: int = 1,
		top_k: int = 2,
		activation: str = 'gelu',
		dropout: float = 0.0,
	):
		super().__init__()
		if num_shared < 0:
			raise ValueError('num_shared must be >= 0')

		self.film = FiLMLayer(query_size, hidden_size)
		self.shared = nn.ModuleList([
			ExpertFFN(hidden_size, hidden_size * 4, activation)
			for _ in range(num_shared)
		])
		self.router = QueryConditionedRouter(hidden_size, query_size, num_experts, top_k=top_k)
		r_inter = max(hidden_size, hidden_size * 4 * self.router.top_k // num_experts)
		self.experts = nn.ModuleList([
			ExpertFFN(hidden_size, r_inter, activation)
			for _ in range(num_experts)
		])
		self.norm = nn.LayerNorm(hidden_size)
		self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

	@staticmethod
	def _load_balancing_loss(gate_scores: Tensor, topk_idx: Tensor, top_k: int, num_experts: int) -> Tensor:
		# Switch-style load balancing: encourage both routing probabilities and expert usage to be uniform.
		dispatch = F.one_hot(topk_idx, num_classes=num_experts).to(gate_scores.dtype)
		dispatch = dispatch.sum(dim=2) / float(top_k)
		expert_load = dispatch.mean(dim=(0, 1))
		expert_prob = gate_scores.mean(dim=(0, 1))
		return num_experts * torch.sum(expert_load * expert_prob)

	@staticmethod
	def _router_z_loss(gate_logits: Tensor) -> Tensor:
		# Switch-style z-loss regularization keeps router logits numerically stable.
		log_z = torch.logsumexp(gate_logits, dim=-1)
		return (log_z ** 2).mean()

	def forward(self, vis_emb: Tensor, query_emb: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
		if query_emb is None:
			zero = vis_emb.new_zeros(())
			return vis_emb, zero, zero

		residual = vis_emb
		vis_cond = self.film(vis_emb, query_emb)

		shared_out = torch.zeros_like(vis_cond)
		for expert in self.shared:
			shared_out = shared_out + expert(vis_cond)

		topk_scores, topk_idx, gate_scores, gate_logits = self.router(vis_cond, query_emb)
		routed_out = torch.zeros_like(vis_cond)
		for k in range(self.router.top_k):
			idx = topk_idx[..., k]
			g = topk_scores[..., k].unsqueeze(-1)
			for i, expert in enumerate(self.experts):
				mask = (idx == i).to(vis_cond.dtype).unsqueeze(-1)
				routed_out = routed_out + mask * g * expert(vis_cond)

		out = self.dropout(shared_out + routed_out)
		load_balance_loss = self._load_balancing_loss(
			gate_scores=gate_scores.float(),
			topk_idx=topk_idx,
			top_k=self.router.top_k,
			num_experts=len(self.experts),
		)
		z_loss = self._router_z_loss(gate_logits.float())
		return self.norm(out + residual), load_balance_loss, z_loss


def get_query_emb(
	language_model: nn.Module,
	input_ids: Tensor,
	attention_mask: Tensor,
) -> Tensor:
	# with torch.no_grad():
	text_emb = language_model.get_input_embeddings()(input_ids)
	mask = attention_mask.unsqueeze(-1).float()
	query_emb = (text_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
	return query_emb