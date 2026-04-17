"""ModelFactory — resolves model specs to (ProviderConfig, model_name)."""
from __future__ import annotations

from aura.config.schema import AuraConfig, AuraConfigError, ProviderConfig


class UnknownModelSpecError(AuraConfigError):
    """Raised when a model spec can't be resolved to a known provider."""


class ModelFactory:
    """Namespace class for model resolution utilities."""

    @staticmethod
    def resolve(spec: str, *, cfg: AuraConfig) -> tuple[ProviderConfig, str]:
        """Resolve *spec* to (ProviderConfig, model_name).

        Resolution rules (spec §4.6):
        1. If *spec* is a router alias, substitute once.
        2. Split on the first ':'. Left = provider name, right = model name.
        3. Unknown alias with no colon → UnknownModelSpecError.
        4. Unknown provider name → UnknownModelSpecError.
        """
        resolved = cfg.router.get(spec, spec)

        provider_name, sep, model_name = resolved.partition(":")

        if not sep:
            # spec was not a router alias AND has no colon
            raise UnknownModelSpecError(
                "model spec",
                f"{spec!r} is not a router alias and not in 'provider:model' form",
            )

        # Linear scan — providers list is short (1-5 in practice)
        for provider in cfg.providers:
            if provider.name == provider_name:
                return (provider, model_name)

        raise UnknownModelSpecError(
            "model spec",
            f"unknown provider {provider_name!r} in spec {spec!r}; "
            f"known providers: {[p.name for p in cfg.providers]}",
        )
