"""Pareto Inference model provider for Hermes Agent (https://paretoinference.com).

GLM 5.3 Flash, DeepSeek V4.1 Flash and MiMo V2.6 Flash behind an OpenAI-compatible Chat
Completions API. Every request must stream: a plain call is answered
HTTP 400 ``streaming_only``. Reasoning arrives in ``reasoning_content``.

Registers one provider profile at import and nothing else: Hermes' own client sends the requests,
with ``PARETO_API_KEY`` read by Hermes. Imports only the public ``providers`` / ``providers.base`` API.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

PARETO_MODELS = ("z-ai/glm-5.3-flash", "deepseek/deepseek-v4.1-flash", "xiaomi/mimo-v2.6-flash")
_BARE_IDS = {model_id.split("/", 1)[1]: model_id for model_id in PARETO_MODELS}

# docs.paretoinference.com/api-reference: top-level ``reasoning_effort`` levels. Hermes' extra top
# level ``ultra`` is sent as ``max``.
PARETO_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# models.dev listed only GLM 5.3 Flash under ``pareto`` (2026-11-06); declaring all three keeps context-window
# lookup off Hermes' 200K unknown-model default. Vision stays undeclared (unknown), not False.
_CAPABILITIES = {"context_window": 1_000_000, "supports_tools": True, "supports_reasoning": True}


class ParetoProfile(ProviderProfile):
    """Top-level ``reasoning_effort``; an unset effort keeps Pareto's per-model default."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(reasoning_config, dict):
            return {}, {}
        if reasoning_config.get("enabled") is False:
            return {}, {"reasoning_effort": "none"}
        effort = str(reasoning_config.get("effort") or "").strip().lower()
        effort = "max" if effort == "ultra" else effort
        return {}, ({"reasoning_effort": effort} if effort in PARETO_EFFORTS else {})

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        return PARETO_EFFORTS

    def normalize_model_id(self, model: str, *, api_key: str = "") -> str | None:
        """``glm-5.3-flash`` -> ``z-ai/glm-5.3-flash``: the namespaced id the public catalog lists."""
        return _BARE_IDS.get((model or "").strip().lower())


register_provider(ParetoProfile(
    name="pareto", aliases=("pareto-inference", "paretoinference"), display_name="Pareto Inference",
    description="Pareto Inference — GLM 5.3 Flash, DeepSeek V4.1 Flash, MiMo V2.6 Flash (pay per token)",
    signup_url="https://paretoinference.com/dashboard", env_vars=("PARETO_API_KEY",),
    base_url="https://api.paretoinference.com/v1", auth_type="api_key", api_mode="chat_completions",
    models_dev_id="pareto", hostnames=("api.paretoinference.com",),
    supports_prompt_cache_key=True,
    default_aux_model="z-ai/glm-5.3-flash",
    fallback_models=PARETO_MODELS,
    model_capabilities={model_id: dict(_CAPABILITIES) for model_id in PARETO_MODELS},
))
