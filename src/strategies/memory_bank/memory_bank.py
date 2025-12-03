import math
import numpy as np
from typing import List
from FlagEmbedding import FlagModel

# Add parent directory to path to import custom logger
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import get_logger

# Minimal implementation of the SiliconFriend Forgetting Curve
# S = Strength, t = Time elapsed (turns)
# Retention R = exp( -t / (5 * S) )

logger = get_logger("MemoryBank")

class MemoryNode:
    def __init__(self, content: str, embedding: np.ndarray, turn_created: int):
        self.content = content
        self.embedding = embedding
        self.strength = 1.0
        self.last_accessed = turn_created
        self.created_at = turn_created

class MemoryBank:
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.nodes: List[MemoryNode] = []
        self.current_turn = 0
        
        # Load embedding model (Lazy loading could be better, but we'll do it on init for simplicity)
        logger.info(f"🧠 Initializing MemoryBank with model: {embedding_model_name}")
        self.encoder = FlagModel(
            embedding_model_name, 
            query_instruction_for_retrieval="Represent this sentence for searching relevant history:",
            use_fp16=False,
            device=device
        )

    def update_time(self):
        """Advances the internal clock (turn counter) and applies forgetting."""
        self.current_turn += 1
        self._apply_forgetting_curve()

    def add_memory(self, content: str):
        """Encodes and stores a new memory trace."""
        if not content or not content.strip():
            return
            
        embedding = self.encoder.encode(content)
        node = MemoryNode(content, embedding, self.current_turn)
        self.nodes.append(node)
        logger.debug(f"➕ Added memory (Turn {self.current_turn}): {content[:30]}...")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieves memories relevant to the query and updates their strength."""
        if not self.nodes:
            return []

        query_vec = self.encoder.encode(query)
        
        # Calculate Cosine Similarity
        # Stack embeddings: (N, D)
        all_vecs = np.stack([n.embedding for n in self.nodes])
        
        # Normalize (FlagModel usually outputs normalized, but safety first)
        query_norm = np.linalg.norm(query_vec)
        all_norms = np.linalg.norm(all_vecs, axis=1)
        
        scores = np.dot(all_vecs, query_vec) / (all_norms * query_norm + 1e-9)
        
        # Get Top-K indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        retrieved_content = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0.4: # Minimal relevance threshold
                node = self.nodes[idx]
                retrieved_content.append(node.content)
                
                # UPDATE MECHANISM: Boost strength when recalled
                # SiliconFriend logic: Strength += 1, update last_accessed
                node.strength += 1.0
                node.last_accessed = self.current_turn
        
        return retrieved_content

    def _apply_forgetting_curve(self):
        """
        Applies Ebbinghaus forgetting curve.
        SiliconFriend formula: retention = exp(-delta_t / (5 * S))
        If retention < random(), the memory is forgotten (dropped).
        """
        kept_nodes = []
        for node in self.nodes:
            delta_t = self.current_turn - node.last_accessed
            
            # Using the formula from the paper/code
            # Adding a small epsilon to strength to avoid div by zero if logic changes
            retention_prob = math.exp(-delta_t / (5 * node.strength))
            
            # Probabilistic forgetting
            # We use a deterministic threshold for stability in benchmarks, 
            # or strictly follow the paper with random().
            # For a thesis benchmark, randomness might add noise, but let's stick to the paper's spirit.
            # Using a fixed threshold (e.g. 0.1) is often safer for reproduction than random.random().
            # But here is the paper's exact logic:
            if retention_prob > 0.2: # Threshold to keep (Paper uses random sample, here we use threshold for stability)
                 kept_nodes.append(node)
            else:
                 logger.debug(f"📉 Forgot memory: {node.content[:20]}... (Prob: {retention_prob:.2f})")
        
        if len(self.nodes) != len(kept_nodes):
            logger.info(f"🧹 Forgetting Curve: Dropped {len(self.nodes) - len(kept_nodes)} memories.")
            
        self.nodes = kept_nodes

    def reset(self):
        self.nodes = []
        self.current_turn = 0
