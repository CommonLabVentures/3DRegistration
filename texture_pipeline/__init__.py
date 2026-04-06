"""texture_pipeline — RGBD multi-view texturing for known 3D meshes."""

from .pipeline import TexturePipeline
from .config import PipelineConfig, load_config, default_config

__all__ = ["TexturePipeline", "PipelineConfig", "load_config", "default_config"]
