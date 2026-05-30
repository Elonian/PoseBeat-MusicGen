"""Canonical model modules for PoseBeat-MusicGen."""

from .audio_generator import (
    AudioPipelineComponents,
    MotionConditionedUNet,
    create_conditioned_unet,
    create_noise_scheduler,
    freeze_module,
    load_audio_generator_components,
    load_audio_pipeline_components,
    save_audio_pipeline,
)
from .motion_adapter import MotionAdapter, MotionAdapterConfig, MotionConditioning
from .motion_to_music_unet import create_motion_conditioned_unet

__all__ = [
    "AudioPipelineComponents",
    "MotionAdapter",
    "MotionAdapterConfig",
    "MotionConditionedUNet",
    "MotionConditioning",
    "create_conditioned_unet",
    "create_motion_conditioned_unet",
    "create_noise_scheduler",
    "freeze_module",
    "load_audio_generator_components",
    "load_audio_pipeline_components",
    "save_audio_pipeline",
]
