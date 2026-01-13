"""Tests for Multi-Model Router (Phase 4, Updated v28).

Tests intelligent model selection based on task characteristics.
Updated for granular cli:model routing with quality-first design.
No backward compatibility - cli:model format only.
"""

import pytest

from supervisor.core.routing import (
    MODEL_PROFILES,
    ROLE_MODEL_MAP,
    ModelCapability,
    ModelProfile,
    ModelRouter,
    create_router,
    get_cli_and_model_id,
    get_capability_for_role,
    parse_model_key,
    register_model,
    set_role_model,
)


class TestModelProfiles:
    """Tests for model profile configuration."""

    def test_all_model_tiers_have_profiles(self):
        """Test that all expected cli:model keys have profiles."""
        # Claude family
        assert "claude:opus" in MODEL_PROFILES
        assert "claude:sonnet" in MODEL_PROFILES
        assert "claude:haiku" in MODEL_PROFILES
        # Codex family
        assert "codex:gpt52" in MODEL_PROFILES
        assert "codex:o3" in MODEL_PROFILES
        # Gemini family
        assert "gemini:flash3" in MODEL_PROFILES
        assert "gemini:pro3" in MODEL_PROFILES

    def test_profile_has_required_fields(self):
        """Test that profiles have all required fields."""
        for key, profile in MODEL_PROFILES.items():
            assert profile.name, f"{key} missing name"
            assert profile.cli in ["claude", "codex", "gemini"], f"{key} invalid cli"
            assert profile.model_id, f"{key} missing model_id"
            assert isinstance(profile.strengths, list), f"{key} strengths not a list"
            assert profile.max_context > 0, f"{key} invalid max_context"
            assert profile.relative_cost > 0, f"{key} invalid relative_cost"

    def test_claude_opus_profile(self):
        """Test Claude Opus 4.5 profile configuration."""
        opus = MODEL_PROFILES["claude:opus"]
        assert opus.cli == "claude"
        assert "opus" in opus.model_id.lower()
        assert ModelCapability.ARCHITECTURE in opus.strengths
        assert ModelCapability.SECURITY_FIX in opus.strengths
        assert ModelCapability.REFACTORING in opus.strengths
        assert opus.quality_score > 0.8  # SOTA at 80.9%

    def test_codex_gpt52_profile(self):
        """Test GPT-5.2-Codex profile configuration."""
        gpt52 = MODEL_PROFILES["codex:gpt52"]
        assert gpt52.cli == "codex"
        assert ModelCapability.DEBUGGING in gpt52.strengths
        assert ModelCapability.FRONTEND in gpt52.strengths
        assert gpt52.quality_score >= 0.80

    def test_gemini_flash3_profile(self):
        """Test Gemini 3 Flash profile configuration."""
        flash = MODEL_PROFILES["gemini:flash3"]
        assert flash.cli == "gemini"
        assert ModelCapability.LARGE_CONTEXT in flash.strengths
        assert flash.max_context >= 1000000
        assert flash.relative_cost < 0.2  # Very cost efficient


class TestModelRouter:
    """Tests for ModelRouter class."""

    def test_role_cli_takes_precedence(self):
        """Test that valid role_cli argument overrides all other logic."""
        router = ModelRouter()

        result = router.select_model(
            role_name="planner",
            role_cli="codex:gpt52",  # Explicit override
        )

        assert result == "codex:gpt52"

    def test_invalid_role_cli_raises_error(self):
        """Test that invalid role_cli raises ValueError."""
        router = ModelRouter()

        # Unknown cli:model should raise ValueError
        with pytest.raises(ValueError, match="Unknown model key"):
            router.select_model(
                role_name="planner",
                role_cli="unknown:model",
            )

    def test_default_role_mapping(self):
        """Test default role to model mapping."""
        router = ModelRouter()

        assert router.select_model("planner") == ROLE_MODEL_MAP["planner"]
        assert router.select_model("implementer") == ROLE_MODEL_MAP["implementer"]
        assert router.select_model("debugger") == ROLE_MODEL_MAP["debugger"]

    def test_unknown_role_selects_highest_quality(self):
        """Test that unknown roles select highest quality model (quality-first)."""
        router = ModelRouter()

        result = router.select_model("unknown_custom_role")
        # Quality-first: should select highest quality model
        assert result == "claude:opus"

    def test_large_context_selects_1m_context_model(self):
        """Test that large context sizes select a model with 1M+ context."""
        router = ModelRouter()

        # Normal context - uses default
        result = router.select_model("implementer", context_size=50000)
        assert result == ROLE_MODEL_MAP["implementer"]

        # Large context (>200K) - selects best quality 1M context model
        result = router.select_model("implementer", context_size=300000)
        profile = MODEL_PROFILES[result]
        assert profile.max_context >= 1000000  # Must have 1M+ context

    def test_prefer_cost_selects_cheaper_model(self):
        """Test that prefer_cost mode selects cheaper models."""
        router = ModelRouter(prefer_cost=True)

        # Code gen - find cheapest with this capability
        result = router.select_model(
            "custom_role",
            task_type=ModelCapability.CODE_GEN,
        )

        # Should select one of the cost-efficient models
        profile = MODEL_PROFILES.get(result)
        assert profile is not None
        assert ModelCapability.CODE_GEN in profile.strengths

    def test_task_type_influences_selection(self):
        """Test that task_type is used in model selection."""
        router = ModelRouter(prefer_cost=True)

        # With DEBUGGING capability, should prefer models good at debugging
        result = router.select_model(
            "custom_role",
            task_type=ModelCapability.DEBUGGING,
        )

        profile = MODEL_PROFILES.get(result)
        assert profile is not None
        assert ModelCapability.DEBUGGING in profile.strengths


class TestModelRouterCapabilitySelection:
    """Tests for capability-based model selection."""

    def test_select_model_for_debugging(self):
        """Test selecting model for debugging capability."""
        router = ModelRouter()

        result = router.select_model_for_capability(ModelCapability.DEBUGGING)
        profile = MODEL_PROFILES[result]

        assert ModelCapability.DEBUGGING in profile.strengths

    def test_select_model_for_architecture(self):
        """Test selecting model for architecture capability."""
        router = ModelRouter()

        result = router.select_model_for_capability(ModelCapability.ARCHITECTURE)

        # Should select highest quality model with this capability
        assert result == "claude:opus"

    def test_capability_with_large_context_selects_1m_model(self):
        """Test that large context selects 1M context model with capability if possible."""
        router = ModelRouter()

        # Large context selects best quality 1M context model
        result = router.select_model_for_capability(
            ModelCapability.ARCHITECTURE,
            context_size=300000,
        )
        profile = MODEL_PROFILES[result]
        assert profile.max_context >= 1000000  # Must have 1M+ context

    def test_unknown_capability_defaults_to_sonnet(self):
        """Test that capabilities not found default to Sonnet."""
        router = ModelRouter()

        # DOCUMENTATION should return a valid model
        result = router.select_model_for_capability(ModelCapability.DOCUMENTATION)
        assert result in MODEL_PROFILES

    def test_get_models_for_capability(self):
        """Test getting all models that support a capability."""
        router = ModelRouter()

        # CODE_REVIEW is supported by multiple models
        capable = router.get_models_for_capability(ModelCapability.CODE_REVIEW)

        assert len(capable) > 0
        for key, profile in capable:
            assert ModelCapability.CODE_REVIEW in profile.strengths

    def test_get_models_sorted_by_quality(self):
        """Test that models are sorted by quality_score descending."""
        router = ModelRouter()

        capable = router.get_models_for_capability(ModelCapability.CODE_GEN)

        # Should be sorted by quality_score descending
        scores = [profile.quality_score for _, profile in capable]
        assert scores == sorted(scores, reverse=True)


class TestModelRouterCostEstimation:
    """Tests for cost estimation."""

    def test_estimate_cost_baseline(self):
        """Test cost estimation with baseline model (Claude Opus)."""
        router = ModelRouter()

        cost = router.estimate_cost("claude:opus", input_tokens=1000, output_tokens=500)

        # Claude Opus has relative_cost = 1.0 (per 1M tokens pricing)
        expected = (1000 + 500) / 1_000_000 * 1.0
        assert cost == expected

    def test_estimate_cost_cheaper_model(self):
        """Test cost estimation with cheaper model."""
        router = ModelRouter()

        cost_opus = router.estimate_cost("claude:opus", input_tokens=1000, output_tokens=500)
        cost_flash = router.estimate_cost("gemini:flash3", input_tokens=1000, output_tokens=500)

        # Flash should be much cheaper
        assert cost_flash < cost_opus

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model raises ValueError."""
        router = ModelRouter()

        with pytest.raises(ValueError, match="Unknown model"):
            router.estimate_cost("unknown:model", input_tokens=1000, output_tokens=500)


class TestModelRouterHelpers:
    """Tests for helper methods."""

    def test_get_profile(self):
        """Test getting model profile by model key."""
        router = ModelRouter()

        profile = router.get_profile("claude:opus")
        assert profile is not None
        assert profile.cli == "claude"

        profile = router.get_profile("unknown:model")
        assert profile is None

    def test_get_available_models(self):
        """Test listing available models."""
        router = ModelRouter()

        available = router.get_available_models()
        assert "claude:opus" in available
        assert "claude:sonnet" in available
        assert "codex:gpt52" in available
        assert "gemini:flash3" in available

    def test_get_best_model_for_quality(self):
        """Test getting highest quality model."""
        router = ModelRouter()

        # Overall best
        best = router.get_best_model_for_quality()
        assert best == "claude:opus"  # 80.9% SWE-bench

        # Best for debugging
        best = router.get_best_model_for_quality(ModelCapability.DEBUGGING)
        profile = MODEL_PROFILES[best]
        assert ModelCapability.DEBUGGING in profile.strengths


class TestParseModelKey:
    """Tests for model key parsing."""

    def test_parse_cli_model_format(self):
        """Test parsing cli:model format."""
        cli, model_id = parse_model_key("claude:opus")
        assert cli == "claude"
        assert model_id is not None
        assert "opus" in model_id.lower()

    def test_parse_unknown_model_raises_error(self):
        """Test parsing unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model key"):
            parse_model_key("unknown:model")

    def test_get_cli_and_model_id(self):
        """Test the public interface function."""
        cli, model_id = get_cli_and_model_id("codex:gpt52")
        assert cli == "codex"
        assert "gpt-5.2" in model_id.lower() or "gpt52" in model_id.lower()

    def test_get_cli_and_model_id_unknown_raises(self):
        """Test that unknown model key raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model key"):
            get_cli_and_model_id("invalid:key")


class TestGetCapabilityForRole:
    """Tests for get_capability_for_role function."""

    def test_known_roles_return_capability(self):
        """Test that known roles return appropriate capability."""
        assert get_capability_for_role("planner") == ModelCapability.ARCHITECTURE
        assert get_capability_for_role("implementer") == ModelCapability.CODE_GEN
        assert get_capability_for_role("debugger") == ModelCapability.DEBUGGING
        assert get_capability_for_role("reviewer") == ModelCapability.CODE_REVIEW

    def test_unknown_role_returns_none(self):
        """Test that unknown roles return None."""
        assert get_capability_for_role("unknown_role") is None

    def test_role_variants_map_to_base(self):
        """Test that role variants map to base role capability."""
        # implementer_frontend should map to frontend
        cap = get_capability_for_role("implementer_frontend")
        assert cap == ModelCapability.FRONTEND

        # reviewer_security should map to security
        cap = get_capability_for_role("reviewer_security")
        assert cap == ModelCapability.SECURITY


class TestFactoryFunctions:
    """Tests for module-level factory functions."""

    def test_create_router(self):
        """Test router factory function."""
        router = create_router()
        assert isinstance(router, ModelRouter)
        assert not router.prefer_cost

        router = create_router(prefer_cost=True)
        assert router.prefer_cost

    def test_register_model(self):
        """Test registering custom model."""
        custom_profile = ModelProfile(
            name="Custom Model",
            cli="custom",
            model_id="custom-v1",
            strengths=[ModelCapability.REASONING],
            max_context=50000,
            relative_cost=0.8,
            quality_score=0.75,
        )

        register_model("custom:v1", custom_profile)

        assert "custom:v1" in MODEL_PROFILES
        assert MODEL_PROFILES["custom:v1"].name == "Custom Model"

        # Cleanup
        del MODEL_PROFILES["custom:v1"]

    def test_set_role_model(self):
        """Test setting default model for a role."""
        original = ROLE_MODEL_MAP.get("test_role")

        set_role_model("test_role", "gemini:flash3")
        assert ROLE_MODEL_MAP["test_role"] == "gemini:flash3"

        # Cleanup
        if original is None:
            del ROLE_MODEL_MAP["test_role"]
        else:
            ROLE_MODEL_MAP["test_role"] = original
