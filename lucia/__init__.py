"""Lucía: local-first persistent personal AI system."""

from .model_router import ModelRouter, ModelSelection, ModelProfile, build_default_router, build_ollama_router

__version__ = "0.1.0"

__all__ = [
    "ModelProfile",
    "ModelRouter",
    "ModelSelection",
    "build_default_router",
    "build_ollama_router",
]
