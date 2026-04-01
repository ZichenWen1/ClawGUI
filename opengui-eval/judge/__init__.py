"""Judge module for GUI grounding evaluation."""
from .base_judge import BaseJudge
from .grounding_judge import ScreenSpotJudge
from .osworld_g_judge import OSWorldGJudge

__all__ = ['BaseJudge', 'ScreenSpotJudge', 'OSWorldGJudge']
