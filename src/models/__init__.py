"""
Model architectures for sign language recognition.

This package contains model definitions for different phases:
- Phase I: BiLSTM baseline with handcrafted features
- Phase II: MobileNetV3 + BiLSTM with end-to-end learning
- Phase III: Optimized models for real-time inference
"""

from .bilstm import OptimizedBiLSTMModel

__all__ = ['OptimizedBiLSTMModel']
