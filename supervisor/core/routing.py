"""Multi-model routing based on task characteristics.

Phase 4 deliverable 4.5: Intelligent model selection for different task types.
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from supervisor.core.state import Database
    from supervisor.metrics.aggregator import MetricsAggregator

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive model selection."""
    enabled: bool = False
    exploration_rate: float = 0.1  # Epsilon for epsilon-greedy
    min_samples: int = 5
    lookback_days: int = 30


class ModelCapability(str, Enum):
    """Capabilities that different models excel at."""

    REASONING = "reasoning"  # Complex logic, edge cases
    SPEED = "speed"  # Fast response for simple tasks
    CONTEXT = "context"  # Large context handling
    CODE_GEN = "code_generation"  # Generating new code
    CODE_REVIEW = "code_review"  # Reviewing existing code
    DOCUMENTATION = "documentation"  # Writing docs
    PLANNING = "planning"  # Architecture, planning


@dataclass
class ModelProfile:
    """Profile of a model's capabilities."""

    name: str
    cli: str
    strengths: list[ModelCapability]
    max_context: int  # Approximate token limit
    relative_speed: float  # 1.0 = baseline
    relative_cost: float  # 1.0 = baseline


# Model capability profiles
MODEL_PROFILES: dict[str, ModelProfile] = {
    "claude": ModelProfile(
        name="Claude",
        cli="claude",
        strengths=[
            ModelCapability.REASONING,
            ModelCapability.PLANNING,
            ModelCapability.CODE_REVIEW,
        ],
        max_context=200000,
        relative_speed=1.0,
        relative_cost=1.0,
    ),
    "codex": ModelProfile(
        name="Codex",
        cli="codex",
        strengths=[
            ModelCapability.SPEED,
            ModelCapability.CODE_GEN,
        ],
        max_context=128000,
        relative_speed=2.0,  # Faster
        relative_cost=0.5,  # Cheaper
    ),
    "gemini": ModelProfile(
        name="Gemini",
        cli="gemini",
        strengths=[
            ModelCapability.CONTEXT,
            ModelCapability.DOCUMENTATION,
            ModelCapability.CODE_REVIEW,
        ],
        max_context=1000000,
        relative_speed=0.8,
        relative_cost=0.7,
    ),
}


# Role to model mapping (default)
ROLE_MODEL_MAP: dict[str, str] = {
    "planner": "claude",
    "implementer": "claude",
    "implementer_fast": "codex",
    "reviewer": "claude",
    "reviewer_gemini": "gemini",
    "reviewer_codex": "codex",
    "investigator": "gemini",
    "doc_generator": "gemini",
}


class ModelRouter:
    """Route tasks to appropriate models.

    ROUTING STRATEGY:
    1. If role specifies CLI, use that
    2. Otherwise, match task type to model strengths
    3. Consider context size (large context -> Gemini)
    4. Consider speed requirements (fast -> Codex)
    """

    def __init__(
        self,
        prefer_speed: bool = False,
        prefer_cost: bool = False,
        aggregator: "MetricsAggregator | None" = None,
        adaptive_config: AdaptiveConfig | None = None,
    ):
        self.prefer_speed = prefer_speed
        self.prefer_cost = prefer_cost
        self.aggregator = aggregator
        self.adaptive_config = adaptive_config or AdaptiveConfig()

    def select_model(
        self,
        role_name: str,
        context_size: int = 0,
        task_type: ModelCapability | None = None,
        role_cli: str | None = None,
    ) -> str:
        """Select the best model for a task.

        Args:
            role_name: Role being executed
            context_size: Estimated context tokens
            task_type: Primary capability needed
            role_cli: CLI specified in role config (takes precedence)

        Returns:
            CLI name to use (claude, codex, gemini)
        """
        # Role config takes precedence
        if role_cli:
            return role_cli

        # Adaptive selection (if enabled and applicable)
        if self.adaptive_config.enabled and self.aggregator:
            adaptive_choice = self._select_adaptive(role_name, context_size)
            if adaptive_choice:
                return adaptive_choice

        # Default mapping
        if role_name in ROLE_MODEL_MAP:
            default = ROLE_MODEL_MAP[role_name]
        else:
            default = "claude"

        # Check context size requirements
        if context_size > 128000:
            # Need large context - Gemini
            return "gemini"

        # Speed preference
        if self.prefer_speed and task_type in [
            ModelCapability.CODE_GEN,
            ModelCapability.SPEED,
        ]:
            return "codex"

        # Cost preference - FIX (PR review): Make generic instead of hardcoding models
        if self.prefer_cost and task_type:
            # Find all models with this capability and sort by cost
            capable_models = [
                (cli, profile)
                for cli, profile in MODEL_PROFILES.items()
                if task_type in profile.strengths
            ]
            if capable_models:
                capable_models.sort(key=lambda x: x[1].relative_cost)
                return capable_models[0][0]

        return default

    def _select_adaptive(self, role_name: str, context_size: int) -> str | None:
        """Select model based on historical performance (epsilon-greedy)."""
        # Context constraint override
        if context_size > 128000:
            return "gemini"

        # Exploration: Randomly select a valid model
        if random.random() < self.adaptive_config.exploration_rate:
            logger.debug(f"Adaptive routing: Exploring random model for {role_name}")
            return random.choice(list(MODEL_PROFILES.keys()))

        # Exploitation: Select best performing model
        # Infer task type from role name (simple heuristic)
        task_type = "other"
        if "plan" in role_name: task_type = "plan"
        elif "implement" in role_name: task_type = "implement"
        elif "review" in role_name: task_type = "review"

        best_cli = self.aggregator.get_best_cli_for_task(
            task_type=task_type,
            days=self.adaptive_config.lookback_days,
            min_samples=self.adaptive_config.min_samples,
        )
        
        if best_cli:
            logger.debug(f"Adaptive routing: Selected {best_cli} for {role_name} (best historical)")
            return best_cli
        
        return None

    def select_model_for_capability(
        self,
        capability: ModelCapability,
        context_size: int = 0,
    ) -> str:
        """Select best model for a specific capability.

        Args:
            capability: Required capability
            context_size: Estimated context tokens

        Returns:
            CLI name to use
        """
        # Context size override
        if context_size > 128000:
            return "gemini"

        # Find models with this capability
        capable_models = [
            (cli, profile)
            for cli, profile in MODEL_PROFILES.items()
            if capability in profile.strengths
        ]

        if not capable_models:
            # Default to Claude if no model explicitly supports this
            return "claude"

        # Apply preferences
        if self.prefer_speed:
            # Sort by speed (higher is faster)
            capable_models.sort(key=lambda x: x[1].relative_speed, reverse=True)
        elif self.prefer_cost:
            # Sort by cost (lower is cheaper)
            capable_models.sort(key=lambda x: x[1].relative_cost)

        return capable_models[0][0]

    def get_profile(self, cli: str) -> ModelProfile | None:
        """Get model profile by CLI name."""
        return MODEL_PROFILES.get(cli)

    def estimate_cost(
        self,
        cli: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate relative cost for a model invocation.

        Returns relative cost units (1.0 = baseline Claude cost).
        Actual pricing should be configured separately.

        Raises:
            ValueError: If CLI is unknown (cannot estimate cost)
        """
        profile = MODEL_PROFILES.get(cli)
        if not profile:
            # FIX (PR review): Raise error instead of returning misleading default
            raise ValueError(f"Unknown model CLI '{cli}' - cannot estimate cost.")

        # Simplified relative cost based on token counts
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * profile.relative_cost

    def get_available_models(self) -> list[str]:
        """Get list of available model CLIs."""
        return list(MODEL_PROFILES.keys())

    def get_models_for_capability(
        self, capability: ModelCapability
    ) -> list[tuple[str, ModelProfile]]:
        """Get all models that support a capability.

        Returns:
            List of (cli, profile) tuples sorted by capability strength
        """
        capable = [
            (cli, profile)
            for cli, profile in MODEL_PROFILES.items()
            if capability in profile.strengths
        ]
        return capable


def create_router(
    prefer_speed: bool = False,
    prefer_cost: bool = False,
    db: "Database | None" = None,
    adaptive_config: AdaptiveConfig | None = None,
) -> ModelRouter:
    """Factory function to create a configured model router.

    Args:
        prefer_speed: Prioritize faster models when possible
        prefer_cost: Prioritize cheaper models when possible
        db: Database instance for metrics aggregation (required for adaptive)
        adaptive_config: Configuration for adaptive routing

    Returns:
        Configured ModelRouter instance
    """
    aggregator = None
    if db:
        from supervisor.metrics.aggregator import MetricsAggregator
        aggregator = MetricsAggregator(db)

    return ModelRouter(
        prefer_speed=prefer_speed,
        prefer_cost=prefer_cost,
        aggregator=aggregator,
        adaptive_config=adaptive_config,
    )


def register_model(cli: str, profile: ModelProfile) -> None:
    """Register a custom model profile.

    Args:
        cli: CLI identifier for the model
        profile: ModelProfile with capabilities and metrics
    """
    MODEL_PROFILES[cli] = profile


def set_role_model(role_name: str, cli: str) -> None:
    """Set the default model for a role.

    Args:
        role_name: Role identifier
        cli: Model CLI to use for this role
    """
    ROLE_MODEL_MAP[role_name] = cli