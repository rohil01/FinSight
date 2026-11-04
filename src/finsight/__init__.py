"""FinSight — autonomous cyclic research agent for Indian fintech, built on LangGraph and NVIDIA NIMs."""
from .graph import build_graph, run_research

__version__ = "0.2.0"
__all__ = ["__version__", "build_graph", "run_research"]
