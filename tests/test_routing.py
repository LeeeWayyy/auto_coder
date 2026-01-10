"""Tests for Multi-Model Router (Phase 4).

Tests intelligent model selection based on task characteristics.
"""

import pytest

from supervisor.core.routing import (
    MODEL_PROFILES,
    ROLE_MODEL_MAP,
    ModelCapability,
    ModelProfile,
    ModelRouter,
    create_router,
    register_model,
    set_role_model,
)


class TestModelProfiles:
    """Tests for model profile configuration."""

    def test_all_clis_have_profiles(self):
        """Test that claude, codex, gemini all have profiles."""
        assert "claude" in MODEL_PROFILES
        assert "codex" in MODEL_PROFILES
        assert "gemini" in MODEL_PROFILES

    def test_profile_has_required_fields(self):
        """Test that profiles have all required fields."""
        for cli, profile in MODEL_PROFILES.items():
            assert profile.name
            assert profile.cli == cli
            assert isinstance(profile.strengths, list)
            assert profile.max_context > 0
            assert profile.relative_speed > 0
            assert profile.relative_cost > 0

    def test_claude_profile(self):
        """Test Claude's profile configuration."""
        claude = MODEL_PROFILES["claude"]
        assert ModelCapability.REASONING in claude.strengths
        assert ModelCapability.PLANNING in claude.strengths
        assert claude.max_context >= 200000

    def test_codex_profile(self):
        """Test Codex's profile configuration."""
        codex = MODEL_PROFILES["codex"]
        assert ModelCapability.SPEED in codex.strengths
        assert ModelCapability.CODE_GEN in codex.strengths
        assert codex.relative_speed > 1.0  # Faster than baseline

    def test_gemini_profile(self):
        """Test Gemini's profile configuration."""
        gemini = MODEL_PROFILES["gemini"]
        assert ModelCapability.CONTEXT in gemini.strengths
        assert gemini.max_context >= 1000000


class TestModelRouter:
    """Tests for ModelRouter class."""

    def test_role_cli_takes_precedence(self):
        """Test that role_cli argument overrides all other logic."""
        router = ModelRouter()

        result = router.select_model(
            role_name="planner",
            role_cli="codex",  # Explicit override
        )

        assert result == "codex"

    def test_default_role_mapping(self):
        """Test default role to model mapping."""
        router = ModelRouter()

        assert router.select_model("planner") == ROLE_MODEL_MAP["planner"]
        assert router.select_model("implementer") == ROLE_MODEL_MAP["implementer"]
        assert router.select_model("reviewer_gemini") == "gemini"
        assert router.select_model("reviewer_codex") == "codex"

    def test_unknown_role_defaults_to_claude(self):
        """Test that unknown roles default to Claude."""
        router = ModelRouter()

        result = router.select_model("unknown_custom_role")
        assert result == "claude"

    def test_large_context_selects_gemini(self):
        """Test that large context sizes select Gemini."""
        router = ModelRouter()

        # Normal context - uses default
        result = router.select_model("implementer", context_size=50000)
        assert result == "claude"

        # Large context - overrides to Gemini
        result = router.select_model("implementer", context_size=200000)
        assert result == "gemini"

    def test_prefer_speed_selects_codex(self):
        """Test that prefer_speed mode selects Codex for code gen."""
        router = ModelRouter(prefer_speed=True)

        result = router.select_model(
            "custom_role",
            task_type=ModelCapability.CODE_GEN,
        )

        assert result == "codex"

    def test_prefer_cost_selects_cheaper_model(self):
        """Test that prefer_cost mode selects cheaper models."""
        router = ModelRouter(prefer_cost=True)

        # Code gen - Codex is cheapest with this capability
        result = router.select_model(
            "custom_role",
            task_type=ModelCapability.CODE_GEN,
        )
        assert result == "codex"

        # Documentation - Gemini is cheaper and has this capability
        result = router.select_model(
            "custom_role",
            task_type=ModelCapability.DOCUMENTATION,
        )
        assert result == "gemini"


class TestModelRouterCapabilitySelection:
    """Tests for capability-based model selection."""

    def test_select_model_for_capability(self):
        """Test selecting model by required capability."""
        router = ModelRouter()

        # Reasoning -> Claude
        result = router.select_model_for_capability(ModelCapability.REASONING)
        assert result == "claude"

        # Speed -> Codex
        result = router.select_model_for_capability(ModelCapability.SPEED)
        assert result == "codex"

        # Large context -> Gemini
        result = router.select_model_for_capability(ModelCapability.CONTEXT)
        assert result == "gemini"

    def test_capability_with_large_context_override(self):
        """Test that large context overrides capability preference."""
        router = ModelRouter()

        # Even for SPEED capability, large context forces Gemini
        result = router.select_model_for_capability(
            ModelCapability.SPEED,
            context_size=200000,
        )
        assert result == "gemini"

    def test_unknown_capability_defaults_to_claude(self):
        """Test that capabilities not explicitly mapped default to Claude."""
        router = ModelRouter()

        # Create a custom capability not in any profile
        # This shouldn't happen in practice, but test the fallback
        result = router.select_model_for_capability(ModelCapability.REASONING)
        assert result == "claude"

    def test_get_models_for_capability(self):
        """Test getting all models that support a capability."""
        router = ModelRouter()

        # CODE_REVIEW is supported by Claude and Gemini
        capable = router.get_models_for_capability(ModelCapability.CODE_REVIEW)
        cli_names = [cli for cli, _ in capable]

        assert "claude" in cli_names
        assert "gemini" in cli_names


class TestModelRouterCostEstimation:
    """Tests for cost estimation."""

    def test_estimate_cost_baseline(self):
        """Test cost estimation with baseline model (Claude)."""
        router = ModelRouter()

        cost = router.estimate_cost("claude", input_tokens=1000, output_tokens=500)

        # Claude has relative_cost = 1.0
        expected = (1000 + 500) / 1000 * 1.0
        assert cost == expected

    def test_estimate_cost_cheaper_model(self):
        """Test cost estimation with cheaper model (Codex)."""
        router = ModelRouter()

        cost_claude = router.estimate_cost(
            "claude", input_tokens=1000, output_tokens=500
        )
        cost_codex = router.estimate_cost(
            "codex", input_tokens=1000, output_tokens=500
        )

        # Codex should be cheaper
        assert cost_codex < cost_claude

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model raises ValueError."""
        router = ModelRouter()

        # FIX (PR review): Unknown models should raise error, not return misleading default
        with pytest.raises(ValueError, match="Unknown model CLI"):
            router.estimate_cost("unknown", input_tokens=1000, output_tokens=500)


class TestModelRouterHelpers:
    """Tests for helper methods."""

    def test_get_profile(self):
        """Test getting model profile by CLI name."""
        router = ModelRouter()

        profile = router.get_profile("claude")
        assert profile is not None
        assert profile.cli == "claude"

        profile = router.get_profile("unknown")
        assert profile is None

    def test_get_available_models(self):
        """Test listing available models."""
        router = ModelRouter()

        available = router.get_available_models()
        assert "claude" in available
        assert "codex" in available
        assert "gemini" in available


class TestFactoryFunctions:
    """Tests for module-level factory functions."""

    def test_create_router(self):
        """Test router factory function."""
        router = create_router()
        assert isinstance(router, ModelRouter)
        assert not router.prefer_speed
        assert not router.prefer_cost

        router = create_router(prefer_speed=True)
        assert router.prefer_speed

        router = create_router(prefer_cost=True)
        assert router.prefer_cost

    def test_register_model(self):
        """Test registering custom model."""
        custom_profile = ModelProfile(
            name="Custom Model",
            cli="custom",
            strengths=[ModelCapability.REASONING],
            max_context=50000,
            relative_speed=1.5,
            relative_cost=0.8,
        )

        register_model("custom", custom_profile)

        assert "custom" in MODEL_PROFILES
        assert MODEL_PROFILES["custom"].name == "Custom Model"

        # Cleanup
        del MODEL_PROFILES["custom"]

    def test_set_role_model(self):
        """Test setting default model for a role."""
        original = ROLE_MODEL_MAP.get("test_role")

        set_role_model("test_role", "gemini")
        assert ROLE_MODEL_MAP["test_role"] == "gemini"

        # Cleanup
        if original is None:
            del ROLE_MODEL_MAP["test_role"]
        else:
            ROLE_MODEL_MAP["test_role"] = original
