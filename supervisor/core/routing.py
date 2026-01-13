"""Multi-model routing based on task characteristics.

Phase 4 deliverable 4.5: Intelligent model selection for different task types.

Updated (v28): Research-backed model profiles with granular cli:model routing.
- Each CLI now has multiple model tiers (e.g., claude:opus, claude:sonnet, claude:haiku)
- Capabilities expanded to cover debugging, security, refactoring, architecture, etc.
- Quality-first design: thoroughness valued over speed

Research sources:
- SWE-bench Verified benchmarks (Dec 2025)
- AI Coding Battle 2025 comparisons
- Security researcher vulnerability tests
"""

import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from supervisor.metrics.aggregator import MetricsAggregator

if TYPE_CHECKING:
    from supervisor.core.state import Database

logger = logging.getLogger(__name__)


# Task type inference for metrics categorization
# Maps role names to task types for adaptive routing
_ROLE_TO_TASK_TYPE: dict[str, str] = {
    # Planning & Architecture
    "planner": "architecture",
    "architect": "architecture",
    # Implementation
    "implementer": "code_gen",
    "implementer_frontend": "frontend",
    "implementer_backend": "backend",
    "implementer_algorithm": "algorithms",
    # Debugging
    "debugger": "debugging",
    "debugger_deep": "debugging",
    # Review
    "reviewer": "code_review",
    "reviewer_security": "security",
    # Refactoring
    "refactorer": "refactoring",
    "refactorer_multi_file": "refactoring",
    # Testing
    "tester": "test_gen",
    # Security
    "security_reviewer": "security",
    "security_fix": "security_fix",
    # Investigation
    "investigator": "investigation",
    "investigator_deep": "investigation",
    # Documentation
    "doc_generator": "documentation",
}


def _infer_task_type(role_name: str) -> str:
    """Infer task type from role name for metrics categorization.

    Uses dictionary lookup for exact matching, avoiding false positives
    from substring matching (e.g., 'reimplementer' won't match 'implement').

    FIX (v27 - Codex PR review P2): Handle role variants like 'reviewer_gemini',
    'reviewer_codex' by extracting base role before suffix (after underscore).
    """
    # First try exact match
    if role_name in _ROLE_TO_TASK_TYPE:
        return _ROLE_TO_TASK_TYPE[role_name]

    # FIX (v27 - Codex PR review P2): Try base role (before underscore suffix)
    # e.g., 'reviewer_gemini' -> 'reviewer', 'implementer_fast' -> 'implementer'
    if "_" in role_name:
        base_role = role_name.rsplit("_", 1)[0]
        if base_role in _ROLE_TO_TASK_TYPE:
            return _ROLE_TO_TASK_TYPE[base_role]

    return "other"


# Map string task types to ModelCapability enum
# Used to wire task-based routing from role names
_TASK_TYPE_TO_CAPABILITY: dict[
    str, "ModelCapability"
] = {}  # Populated after ModelCapability defined


def _init_task_type_mapping() -> None:
    """Initialize task type to capability mapping after enum is defined."""
    global _TASK_TYPE_TO_CAPABILITY
    _TASK_TYPE_TO_CAPABILITY = {
        "architecture": ModelCapability.ARCHITECTURE,
        "code_gen": ModelCapability.CODE_GEN,
        "frontend": ModelCapability.FRONTEND,
        "backend": ModelCapability.BACKEND,
        "algorithms": ModelCapability.ALGORITHMS,
        "debugging": ModelCapability.DEBUGGING,
        "code_review": ModelCapability.CODE_REVIEW,
        "refactoring": ModelCapability.REFACTORING,
        "test_gen": ModelCapability.TEST_GEN,
        "security": ModelCapability.SECURITY,
        "security_fix": ModelCapability.SECURITY_FIX,
        "documentation": ModelCapability.DOCUMENTATION,
        "investigation": ModelCapability.LARGE_CONTEXT,  # Large context for codebase exploration
    }


def get_capability_for_role(role_name: str) -> "ModelCapability | None":
    """Get the primary ModelCapability for a role.

    Args:
        role_name: Role identifier (e.g., "implementer", "debugger")

    Returns:
        ModelCapability enum value or None if role not mapped
    """
    task_type = _infer_task_type(role_name)
    return _TASK_TYPE_TO_CAPABILITY.get(task_type)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive model selection."""

    enabled: bool = False
    exploration_rate: float = 0.1  # Epsilon for epsilon-greedy
    min_samples: int = 5
    lookback_days: int = 30


class ModelCapability(str, Enum):
    """Capabilities that different models excel at.

    Updated (v28): Expanded capabilities based on 2025 benchmark research.
    Quality-first design - no "speed" penalties, thoroughness is valued.
    """

    # === Context & Session Management ===
    LARGE_CONTEXT = "large_context"  # 1M+ token handling
    LONG_SESSION = "long_session"  # 30+ hour autonomous operation
    MULTI_FILE = "multi_file"  # Coherent cross-file changes

    # === Code Generation ===
    CODE_GEN = "code_generation"  # General code writing
    FRONTEND = "frontend"  # UI/React/CSS work
    BACKEND = "backend"  # API/database/server logic
    ALGORITHMS = "algorithms"  # Competitive programming, complex algos

    # === Code Quality ===
    REFACTORING = "refactoring"  # Large-scale restructuring
    DEBUGGING = "debugging"  # Bug finding & fixing
    CODE_REVIEW = "code_review"  # Thorough code analysis
    TEST_GEN = "test_generation"  # Unit/integration test generation

    # === Security ===
    SECURITY = "security"  # Vulnerability detection
    SECURITY_FIX = "security_fix"  # Secure code remediation

    # === Design & Planning ===
    ARCHITECTURE = "architecture"  # System design decisions
    REASONING = "reasoning"  # Complex logical analysis

    # === Documentation ===
    DOCUMENTATION = "documentation"  # Docs, comments, READMEs

    # === Cost Efficiency (for budget-sensitive tasks) ===
    COST_EFFICIENCY = "cost_efficiency"  # Low token cost


# Initialize task type to capability mapping now that ModelCapability is defined
_init_task_type_mapping()


@dataclass
class ModelProfile:
    """Profile of a model's capabilities.

    Updated (v28): Added model_id for CLI flag passing and quality_score
    for quality-based routing decisions.
    """

    name: str
    cli: str  # CLI binary name (claude, codex, gemini)
    model_id: str  # Model identifier to pass via --model flag
    strengths: list[ModelCapability]
    max_context: int  # Approximate token limit
    relative_cost: float  # 1.0 = baseline (Claude Opus 4.5)
    quality_score: float = 0.0  # SWE-bench Verified score (0.0-1.0)


# Model capability profiles - Research-backed (v28)
# Keyed by "cli:model" for granular routing
# Sources: SWE-bench Verified Dec 2025, AI Coding Battle 2025
MODEL_PROFILES: dict[str, ModelProfile] = {
    # ==========================================================================
    # CLAUDE FAMILY
    # ==========================================================================
    "claude:opus": ModelProfile(
        name="Claude Opus 4.5",
        cli="claude",
        model_id="claude-opus-4-5-20251101",
        strengths=[
            ModelCapability.ARCHITECTURE,  # Best at system design
            ModelCapability.SECURITY,  # Best vulnerability detection
            ModelCapability.SECURITY_FIX,  # Only model 10/10 on security refactor
            ModelCapability.REFACTORING,  # "Boring in how consistently they handle multi-file refactors"
            ModelCapability.CODE_REVIEW,  # SOTA review quality
            ModelCapability.LONG_SESSION,  # 30+ hour autonomous operation
            ModelCapability.MULTI_FILE,  # Coherent cross-file changes
            ModelCapability.REASONING,  # Deep logical analysis
        ],
        max_context=200000,
        relative_cost=1.0,  # $5/$25 per 1M tokens (baseline)
        quality_score=0.809,  # First to break 80% on SWE-bench Verified
    ),
    "claude:sonnet": ModelProfile(
        name="Claude Sonnet 4.5",
        cli="claude",
        model_id="claude-sonnet-4-5-20250929",
        strengths=[
            ModelCapability.MULTI_FILE,  # "Most reliable for maintaining complex systems"
            ModelCapability.BACKEND,  # Reliable backend development
            ModelCapability.CODE_GEN,  # Strong general code generation
            ModelCapability.SECURITY,  # Doubled Cybench score
            ModelCapability.CODE_REVIEW,  # Good review quality
            ModelCapability.TEST_GEN,  # Comprehensive test suites
        ],
        max_context=200000,
        relative_cost=0.6,  # $3/$15 per 1M tokens
        quality_score=0.76,
    ),
    "claude:haiku": ModelProfile(
        name="Claude Haiku 4.5",
        cli="claude",
        model_id="claude-haiku-4-5",
        strengths=[
            ModelCapability.COST_EFFICIENCY,  # 1/3 cost of Sonnet
            ModelCapability.CODE_GEN,  # 90% capability of Sonnet
        ],
        max_context=200000,
        relative_cost=0.2,  # $1/$5 per 1M tokens
        quality_score=0.70,
    ),
    # ==========================================================================
    # CODEX/OPENAI FAMILY
    # ==========================================================================
    "codex:gpt52": ModelProfile(
        name="GPT-5.2-Codex",
        cli="codex",
        model_id="gpt-5.2-codex",
        strengths=[
            ModelCapability.DEBUGGING,  # "Thorough context gathering catches subtle bugs"
            ModelCapability.FRONTEND,  # 70% win rate on frontend vs o3
            ModelCapability.CODE_GEN,  # "Production-ready code with fewer critical bugs"
            ModelCapability.REASONING,  # 52.9% ARC-AGI-2 (best)
            ModelCapability.MULTI_FILE,  # "Suggests refactors aligning with project architecture"
        ],
        max_context=1000000,
        relative_cost=0.5,  # ~$2.50/$10 per 1M tokens
        quality_score=0.80,  # 80% SWE-bench Verified
    ),
    "codex:gpt51max": ModelProfile(
        name="GPT-5.1-Codex-Max",
        cli="codex",
        model_id="gpt-5.1-codex-max",
        strengths=[
            ModelCapability.DEBUGGING,  # "Most dependable for real-world development"
            ModelCapability.MULTI_FILE,  # Large-scale changes
            ModelCapability.REFACTORING,  # Production-ready refactors
        ],
        max_context=1000000,
        relative_cost=0.6,
        quality_score=0.779,  # 77.9% SWE-bench Verified
    ),
    "codex:o3": ModelProfile(
        name="o3",
        cli="codex",
        model_id="o3",
        strengths=[
            ModelCapability.REASONING,  # Strong reasoning specialist
            ModelCapability.ALGORITHMS,  # Good at algorithmic problems
            ModelCapability.COST_EFFICIENCY,  # Very cheap
        ],
        max_context=200000,
        relative_cost=0.08,  # $0.40/$1.60 per 1M tokens
        quality_score=0.691,
    ),
    "codex:mini": ModelProfile(
        name="GPT-5-Codex-Mini",
        cli="codex",
        model_id="gpt-5-codex-mini",
        strengths=[
            ModelCapability.COST_EFFICIENCY,  # Very cheap
            ModelCapability.CODE_GEN,  # Basic code generation
        ],
        max_context=128000,
        relative_cost=0.08,
        quality_score=0.65,
    ),
    # ==========================================================================
    # GEMINI FAMILY
    # ==========================================================================
    "gemini:flash3": ModelProfile(
        name="Gemini 3 Flash",
        cli="gemini",
        model_id="gemini-3-flash",
        strengths=[
            ModelCapability.LARGE_CONTEXT,  # 1M token context
            ModelCapability.ALGORITHMS,  # Strong on algorithmic challenges
            ModelCapability.COST_EFFICIENCY,  # Best value: 78% at $0.50/$3
            ModelCapability.CODE_GEN,  # Good code generation
            ModelCapability.TEST_GEN,  # Fast test generation
        ],
        max_context=1000000,
        relative_cost=0.12,  # $0.50/$3 per 1M tokens
        quality_score=0.78,  # 78% SWE-bench Verified (beats Pro!)
    ),
    "gemini:pro3": ModelProfile(
        name="Gemini 3 Pro",
        cli="gemini",
        model_id="gemini-3-pro",
        strengths=[
            ModelCapability.LARGE_CONTEXT,  # 1M token context
            ModelCapability.ALGORITHMS,  # 1500+ Elo competitive programming
            ModelCapability.REASONING,  # Complex reasoning tasks
        ],
        max_context=1000000,
        relative_cost=0.25,  # $1.25/$5 per 1M tokens
        quality_score=0.762,
    ),
    "gemini:pro25": ModelProfile(
        name="Gemini 2.5 Pro",
        cli="gemini",
        model_id="gemini-2.5-pro",
        strengths=[
            ModelCapability.LARGE_CONTEXT,  # 1M token context
            ModelCapability.DOCUMENTATION,  # Good for docs
        ],
        max_context=1000000,
        relative_cost=0.25,
        quality_score=0.65,
    ),
}


# Role to model mapping - Quality-first (v28)
# Based on research: thoroughness valued over speed
ROLE_MODEL_MAP: dict[str, str] = {
    # === Planning & Architecture ===
    "planner": "claude:opus",  # Best architecture decisions, 30+ hr sessions
    "architect": "claude:opus",  # System design specialist
    # === Implementation ===
    "implementer": "claude:sonnet",  # Reliable for complex systems
    "implementer_frontend": "codex:gpt52",  # 70% win rate on frontend
    "implementer_backend": "claude:sonnet",  # Best for backend reliability
    "implementer_algorithm": "gemini:pro3",  # 1500+ Elo competitive programming
    # === Debugging (Quality Focus) ===
    "debugger": "codex:gpt52",  # "Thorough context gathering catches subtle bugs"
    "debugger_deep": "claude:opus",  # 2-day production bug fixes, concurrency
    # === Code Review ===
    "reviewer": "claude:opus",  # SOTA review quality
    "reviewer_security": "claude:opus",  # 100% detection + best fixes
    # === Refactoring ===
    "refactorer": "claude:opus",  # 10/10 on security refactor tests
    "refactorer_multi_file": "claude:sonnet",  # Coherent cross-file changes
    # === Testing ===
    "tester": "claude:sonnet",  # Comprehensive test suites
    # === Security ===
    "security_reviewer": "claude:opus",  # Best vulnerability detection
    "security_fix": "claude:opus",  # Only model with complete fixes
    # === Investigation ===
    "investigator": "gemini:flash3",  # 1M context for large codebases
    "investigator_deep": "claude:opus",  # Long session, thorough analysis
    # === Documentation ===
    "doc_generator": "claude:sonnet",  # Quality documentation
}


def parse_model_key(model_key: str) -> tuple[str, str]:
    """Parse a model key into CLI and model_id components.

    Args:
        model_key: Model key in "cli:model" format (e.g., "claude:opus")

    Returns:
        Tuple of (cli_name, model_id)

    Raises:
        ValueError: If model_key is not found in MODEL_PROFILES

    Examples:
        "claude:opus" -> ("claude", "claude-opus-4-5-20251101")
        "codex:gpt52" -> ("codex", "gpt-5.2-codex")
    """
    profile = MODEL_PROFILES.get(model_key)
    if profile:
        return profile.cli, profile.model_id

    # Unknown model key - raise error with helpful message
    available = ", ".join(sorted(MODEL_PROFILES.keys()))
    raise ValueError(f"Unknown model key '{model_key}'. " f"Available models: {available}")


class ModelRouter:
    """Route tasks to appropriate models.

    Updated (v28): Quality-first routing with granular cli:model selection.

    ROUTING PRECEDENCE (in order):
    1. role_cli: Explicit model in role config (must be valid cli:model key)
    2. ROLE_MODEL_MAP: Default model for known roles
       - Context override: If mapped model can't handle context, use large-context model
    3. Large context: If context > 200K, select best 1M+ context model
    4. Adaptive: Historical performance for unknown roles (if enabled)
    5. Task-type: Capability-based selection for unknown roles
    6. Default: claude:sonnet

    All paths respect prefer_cost for cost-first vs quality-first selection.
    """

    def __init__(
        self,
        prefer_cost: bool = False,
        aggregator: "MetricsAggregator | None" = None,
        adaptive_config: AdaptiveConfig | None = None,
    ):
        self.prefer_cost = prefer_cost
        self.aggregator = aggregator
        self.adaptive_config = adaptive_config or AdaptiveConfig()

    def _filter_by_context(
        self,
        models: list[tuple[str, ModelProfile]],
        context_size: int,
    ) -> list[tuple[str, ModelProfile]]:
        """Filter models that can handle the required context size.

        Args:
            models: List of (key, profile) tuples
            context_size: Required context tokens

        Returns:
            Filtered list of models with max_context >= context_size
        """
        if context_size <= 0:
            return models
        return [(k, p) for k, p in models if p.max_context >= context_size]

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
            role_cli: Model specified in role config (takes precedence).
                      Must be a valid "cli:model" key (e.g., "claude:opus")

        Returns:
            Model key in "cli:model" format (e.g., "claude:opus", "codex:gpt52")

        Raises:
            ValueError: If role_cli is provided but not a valid model key,
                       or if context_size is negative
        """
        # Validate context_size
        if context_size < 0:
            raise ValueError(f"context_size must be non-negative, got {context_size}")

        # Role config takes precedence (must be valid cli:model key)
        if role_cli:
            if role_cli not in MODEL_PROFILES:
                available = ", ".join(sorted(MODEL_PROFILES.keys()))
                raise ValueError(
                    f"Unknown model key '{role_cli}' specified for role '{role_name}'. "
                    f"Available models: {available}"
                )
            # Validate role_cli can handle context size
            if context_size > 0:
                profile = MODEL_PROFILES[role_cli]
                if profile.max_context < context_size:
                    # Role specifies a model that can't handle context - find best that can
                    return self._select_best_for_context(context_size, task_type)
            return role_cli

        # ROLE_MODEL_MAP takes precedence for known roles
        # This ensures explicit role mappings are honored
        if role_name in ROLE_MODEL_MAP:
            mapped_model = ROLE_MODEL_MAP[role_name]
            # Validate the mapped model exists
            if mapped_model not in MODEL_PROFILES:
                raise ValueError(
                    f"ROLE_MODEL_MAP has invalid model '{mapped_model}' for role '{role_name}'. "
                    f"Available models: {', '.join(sorted(MODEL_PROFILES.keys()))}"
                )
            # Check if mapped model can handle context size
            if context_size > 0:
                profile = MODEL_PROFILES[mapped_model]
                if profile.max_context < context_size:
                    # Need model that can handle context - find best that can
                    return self._select_best_for_context(context_size, task_type)
            return mapped_model

        # Check context size requirements - need large context model
        if context_size > 200000:
            return self._select_for_large_context(task_type, context_size)

        # Adaptive selection for unknown roles (if enabled)
        if self.adaptive_config.enabled and self.aggregator:
            adaptive_choice = self._select_adaptive(role_name, context_size)
            if adaptive_choice:
                return adaptive_choice

        # Task-type based selection for unknown roles (quality-first or cost-first)
        if task_type:
            capable_models = [
                (key, profile)
                for key, profile in MODEL_PROFILES.items()
                if task_type in profile.strengths
            ]
            # Filter by context size
            capable_models = self._filter_by_context(capable_models, context_size)
            if capable_models:
                if self.prefer_cost:
                    # Cost-first with quality tiebreaker
                    capable_models.sort(key=lambda x: (x[1].relative_cost, -x[1].quality_score))
                else:
                    # Quality-first - find highest quality model with required capability
                    capable_models.sort(key=lambda x: -x[1].quality_score)
                return capable_models[0][0]

        # Default fallback - respect prefer_cost and context size
        all_models = list(MODEL_PROFILES.items())
        all_models = self._filter_by_context(all_models, context_size)
        if not all_models:
            raise ValueError(
                f"No model available with sufficient context capacity for {context_size} tokens."
            )
        if self.prefer_cost:
            # Find cheapest model that can handle context
            all_models.sort(key=lambda x: (x[1].relative_cost, -x[1].quality_score))
        else:
            # Quality-first: highest quality model
            all_models.sort(key=lambda x: -x[1].quality_score)
        return all_models[0][0]

    def _select_for_large_context(
        self,
        task_type: ModelCapability | None = None,
        context_size: int = 0,
    ) -> str:
        """Select best model for large context (>200K tokens).

        Selects among models with 1M+ context using quality-first or
        cost-first ordering based on prefer_cost setting.

        Args:
            task_type: Optional capability preference
            context_size: Required context size (filters models that can't handle it)
        """
        # Find models with large context capability (1M+)
        large_context_models = [
            (key, profile)
            for key, profile in MODEL_PROFILES.items()
            if profile.max_context >= 1000000
        ]

        # Filter by actual context size if specified
        if context_size > 0:
            large_context_models = self._filter_by_context(large_context_models, context_size)

        if not large_context_models:
            # No 1M+ model can handle this - fall back to best model for context
            # This handles future mid-range models (300k-500k) gracefully
            return self._select_best_for_context(context_size, task_type)

        # If task_type specified, prefer models with that capability
        if task_type:
            capable = [(k, p) for k, p in large_context_models if task_type in p.strengths]
            if capable:
                large_context_models = capable

        # Cost-first or quality-first selection
        if self.prefer_cost:
            # Cost-first: cheapest model (with quality as tiebreaker)
            large_context_models.sort(key=lambda x: (x[1].relative_cost, -x[1].quality_score))
        else:
            # Quality-first: highest quality model
            large_context_models.sort(key=lambda x: -x[1].quality_score)

        return large_context_models[0][0]

    def _select_best_for_context(
        self,
        context_size: int,
        task_type: ModelCapability | None = None,
    ) -> str:
        """Select best model that can handle the given context size.

        Unlike _select_for_large_context, this considers ALL models that can
        handle the context, not just 1M+ models. Use this when a specific
        model can't handle the required context.

        Args:
            context_size: Required context size
            task_type: Optional capability preference

        Raises:
            ValueError: If no model can handle the context size, or context_size < 0
        """
        # Defensive validation
        if context_size < 0:
            raise ValueError(f"context_size must be non-negative, got {context_size}")

        # Find all models that can handle the context
        candidates = self._filter_by_context(list(MODEL_PROFILES.items()), context_size)

        if not candidates:
            raise ValueError(
                f"No model available with sufficient context capacity for {context_size} tokens."
            )

        # If task_type specified, prefer models with that capability
        if task_type:
            capable = [(k, p) for k, p in candidates if task_type in p.strengths]
            if capable:
                candidates = capable

        # Cost-first or quality-first selection
        if self.prefer_cost:
            candidates.sort(key=lambda x: (x[1].relative_cost, -x[1].quality_score))
        else:
            candidates.sort(key=lambda x: -x[1].quality_score)

        return candidates[0][0]

    def _select_adaptive(self, role_name: str, context_size: int) -> str | None:
        """Select model based on historical performance (epsilon-greedy).

        Respects prefer_cost setting when selecting among performers.
        Validates returned model exists in MODEL_PROFILES.
        """
        # Context constraint override - need large context
        if context_size > 200000:
            return self._select_for_large_context(context_size=context_size)

        # Exploration: Randomly select a valid model that can handle context
        if random.random() < self.adaptive_config.exploration_rate:
            logger.debug(f"Adaptive routing: Exploring random model for {role_name}")
            # Filter models that can handle the context size
            candidates = self._filter_by_context(list(MODEL_PROFILES.items()), context_size)
            if not candidates:
                return None  # Fall back to other selection methods
            if self.prefer_cost:
                # Biased exploration: sample from cheapest 3 models
                candidates.sort(key=lambda x: x[1].relative_cost)
                top_cheap = candidates[:3]
                return random.choice(top_cheap)[0]
            return random.choice(candidates)[0]

        # Exploitation: Select best performing model
        task_type = _infer_task_type(role_name)

        best_model = self.aggregator.get_best_cli_for_task(
            task_type=task_type,
            days=self.adaptive_config.lookback_days,
            min_samples=self.adaptive_config.min_samples,
        )

        if best_model:
            # Validate model still exists (metrics may contain stale keys)
            if best_model not in MODEL_PROFILES:
                logger.warning(
                    f"Adaptive routing: Ignoring stale model '{best_model}' from metrics"
                )
                return None
            # Validate model can handle context size
            profile = MODEL_PROFILES[best_model]
            if profile.max_context < context_size:
                logger.debug(
                    f"Adaptive routing: Model '{best_model}' can't handle context {context_size}"
                )
                return None
            # Respect prefer_cost: skip expensive historical models in cost-first mode
            if self.prefer_cost:
                # Find cheapest model that can handle context
                eligible = self._filter_by_context(list(MODEL_PROFILES.items()), context_size)
                if eligible:
                    cheapest = min(eligible, key=lambda x: x[1].relative_cost)
                    if profile.relative_cost > cheapest[1].relative_cost:
                        logger.debug(
                            f"Adaptive routing: Skipping '{best_model}' (cost {profile.relative_cost}) "
                            f"in favor of cheaper options in cost-first mode"
                        )
                        return None
            logger.debug(
                f"Adaptive routing: Selected {best_model} for {role_name} (best historical)"
            )
            return best_model

        return None

    def select_model_for_capability(
        self,
        capability: ModelCapability,
        context_size: int = 0,
    ) -> str:
        """Select best model for a specific capability.

        Updated (v28): Quality-first selection, returns cli:model keys.

        Args:
            capability: Required capability
            context_size: Estimated context tokens

        Returns:
            Model key in "cli:model" format

        Raises:
            ValueError: If context_size is negative
        """
        # Validate context_size
        if context_size < 0:
            raise ValueError(f"context_size must be non-negative, got {context_size}")

        # Context size override - need large context
        if context_size > 200000:
            return self._select_for_large_context(capability, context_size)

        # Find models with this capability
        capable_models = [
            (key, profile)
            for key, profile in MODEL_PROFILES.items()
            if capability in profile.strengths
        ]

        # Filter by context size
        capable_models = self._filter_by_context(capable_models, context_size)

        if not capable_models:
            # No model can handle context or no model supports capability
            if context_size > 0:
                return self._select_best_for_context(context_size, capability)
            # Fallback: select from all models respecting prefer_cost
            all_models = list(MODEL_PROFILES.items())
            if self.prefer_cost:
                all_models.sort(key=lambda x: (x[1].relative_cost, -x[1].quality_score))
            else:
                all_models.sort(key=lambda x: -x[1].quality_score)
            return all_models[0][0]

        # Cost-first or quality-first selection
        if self.prefer_cost:
            # Cost-first: cheapest model (with quality as tiebreaker)
            capable_models.sort(key=lambda x: (x[1].relative_cost, -x[1].quality_score))
        else:
            # Quality-first: highest quality model
            capable_models.sort(key=lambda x: -x[1].quality_score)

        return capable_models[0][0]

    def get_profile(self, model_key: str) -> ModelProfile | None:
        """Get model profile by model key.

        Args:
            model_key: Model key in "cli:model" format (e.g., "claude:opus")

        Returns:
            ModelProfile or None if not found
        """
        return MODEL_PROFILES.get(model_key)

    def estimate_cost(
        self,
        model_key: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate relative cost for a model invocation.

        Returns relative cost units (1.0 = baseline Claude Opus cost).
        Actual pricing should be configured separately.

        Args:
            model_key: Model key in "cli:model" format (e.g., "claude:opus")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Raises:
            ValueError: If model key is unknown (cannot estimate cost)
        """
        profile = self.get_profile(model_key)
        if not profile:
            raise ValueError(f"Unknown model '{model_key}' - cannot estimate cost.")

        # Relative cost based on token counts (per 1M tokens pricing)
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1_000_000) * profile.relative_cost

    def get_available_models(self) -> list[str]:
        """Get list of available model keys in cli:model format."""
        return list(MODEL_PROFILES.keys())

    def get_models_for_capability(
        self, capability: ModelCapability
    ) -> list[tuple[str, ModelProfile]]:
        """Get all models that support a capability.

        Returns:
            List of (model_key, profile) tuples sorted by quality_score (descending)
        """
        capable = [
            (key, profile)
            for key, profile in MODEL_PROFILES.items()
            if capability in profile.strengths
        ]
        # Sort by quality_score descending
        capable.sort(key=lambda x: -x[1].quality_score)
        return capable

    def get_best_model_for_quality(
        self,
        capability: ModelCapability | None = None,
    ) -> str:
        """Get the highest quality model, optionally filtered by capability.

        Args:
            capability: Optional capability filter

        Returns:
            Model key with highest quality_score
        """
        if capability:
            capable = [
                (key, profile)
                for key, profile in MODEL_PROFILES.items()
                if capability in profile.strengths
            ]
        else:
            capable = list(MODEL_PROFILES.items())

        if not capable:
            return "claude:opus"  # Default to highest quality

        capable.sort(key=lambda x: -x[1].quality_score)
        return capable[0][0]


def create_router(
    prefer_cost: bool = False,
    db: "Database | None" = None,
    adaptive_config: AdaptiveConfig | None = None,
) -> ModelRouter:
    """Factory function to create a configured model router.

    Updated (v28): Removed prefer_speed (quality-first design).

    Args:
        prefer_cost: Prioritize cheaper models when possible
        db: Database instance for metrics aggregation (required for adaptive)
        adaptive_config: Configuration for adaptive routing

    Returns:
        Configured ModelRouter instance
    """
    aggregator = None
    if db:
        aggregator = MetricsAggregator(db)

    return ModelRouter(
        prefer_cost=prefer_cost,
        aggregator=aggregator,
        adaptive_config=adaptive_config,
    )


_MODEL_KEY_PATTERN = re.compile(r"^[a-z]+:[a-z0-9_-]+$")


def register_model(model_key: str, profile: ModelProfile) -> None:
    """Register a custom model profile.

    Args:
        model_key: Model key in "cli:model" format (e.g., "custom:v1")
        profile: ModelProfile with capabilities and metrics

    Raises:
        ValueError: If model_key doesn't match cli:model format
    """
    if not _MODEL_KEY_PATTERN.match(model_key):
        raise ValueError(
            f"Invalid model key format '{model_key}'. "
            f"Must be 'cli:model' format (e.g., 'custom:v1')."
        )
    MODEL_PROFILES[model_key] = profile


def set_role_model(role_name: str, model_key: str) -> None:
    """Set the default model for a role.

    Args:
        role_name: Role identifier
        model_key: Model key in "cli:model" format

    Raises:
        ValueError: If model_key is not a registered model
    """
    if model_key not in MODEL_PROFILES:
        available = ", ".join(sorted(MODEL_PROFILES.keys()))
        raise ValueError(f"Unknown model key '{model_key}'. " f"Available models: {available}")
    ROLE_MODEL_MAP[role_name] = model_key


def get_cli_and_model_id(model_key: str) -> tuple[str, str]:
    """Get CLI binary name and model ID for a model key.

    This is the main interface for executors to get CLI command info.

    Args:
        model_key: Model key in "cli:model" format (e.g., "claude:opus")

    Returns:
        Tuple of (cli_binary_name, model_id)

    Raises:
        ValueError: If model_key is not found in MODEL_PROFILES

    Examples:
        >>> get_cli_and_model_id("claude:opus")
        ("claude", "claude-opus-4-5-20251101")
        >>> get_cli_and_model_id("codex:gpt52")
        ("codex", "gpt-5.2-codex")
    """
    return parse_model_key(model_key)
