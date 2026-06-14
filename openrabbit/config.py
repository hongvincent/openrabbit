"""Loader for ``.openrabbit.yaml`` config-as-code (SPEC section 8.2).

``load_config`` accepts either a filesystem path (``str``/``Path``) or an
already-parsed mapping and returns a typed, defaulted, validated :class:`Config`.
Unknown keys are tolerated additively where the spec says the format is
"additive/versioned", but enum-constrained values are validated strictly so a
typo fails fast rather than silently degrading review quality.

``pyyaml`` is imported lazily inside :func:`load_config` so the module imports
with no third-party deps when callers pass a dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

# --------------------------------------------------------------------------- #
# vocabularies & defaults                                                     #
# --------------------------------------------------------------------------- #
PROFILES = ("chill", "balanced", "assertive")
LENSES = ("correctness", "security", "performance", "tests", "maintainability")
TELEMETRY_MODES = ("opt-in", "opt-out")

DEFAULT_CONFIDENCE_GATE = 0.80
DEFAULT_PROFILE = "balanced"
DEFAULT_LENSES = list(LENSES)

PathLike = Union[str, Path]


class ConfigError(ValueError):
    """Raised when a config is missing, unparsable, or invalid."""


# --------------------------------------------------------------------------- #
# typed config model                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PathInstruction:
    """A per-path reviewer instruction (SPEC 8.2 ``path_instructions``)."""

    path: str
    instructions: str


@dataclass(frozen=True)
class ReviewConfig:
    """The ``review:`` block."""

    profile: str = DEFAULT_PROFILE
    confidence_gate: float = DEFAULT_CONFIDENCE_GATE
    incremental: bool = True
    path_filters: list[str] = field(default_factory=list)
    path_instructions: list[PathInstruction] = field(default_factory=list)
    lenses: list[str] = field(default_factory=lambda: list(DEFAULT_LENSES))


@dataclass(frozen=True)
class ModelRole:
    """A single ``model_roles.<role>`` entry: BYO Bedrock model + region + opts.

    ``model`` and ``region`` are first-class; any other keys (``reasoning_effort``,
    ``store``, ``enabled``, ...) are preserved in :attr:`options`.
    """

    model: str
    region: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalTools:
    """The ``external_tools:`` block — deterministic graders fed into context."""

    enabled: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Telemetry:
    """The ``telemetry:`` block."""

    enabled: bool = True
    mode: str = "opt-out"


@dataclass(frozen=True)
class Config:
    """Parsed, validated ``.openrabbit.yaml`` document."""

    version: int = 1
    review: ReviewConfig = field(default_factory=ReviewConfig)
    model_roles: dict[str, ModelRole] = field(default_factory=dict)
    external_tools: ExternalTools = field(default_factory=ExternalTools)
    telemetry: Telemetry = field(default_factory=Telemetry)


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
def load_config(source: Union[PathLike, Mapping[str, Any]]) -> Config:
    """Load and validate a config from a path or an already-parsed mapping.

    Raises :class:`ConfigError` for missing files, parse errors, or invalid
    values.
    """
    if isinstance(source, Mapping):
        data: Any = dict(source)
    elif isinstance(source, (str, Path)):
        data = _read_yaml(source)
    else:
        raise ConfigError(
            "config source must be a path (str/Path) or a mapping, got "
            f"{type(source).__name__}"
        )

    if not isinstance(data, Mapping):
        raise ConfigError(
            f"config root must be a mapping, got {type(data).__name__}"
        )

    return _parse(dict(data))


# --------------------------------------------------------------------------- #
# internals                                                                   #
# --------------------------------------------------------------------------- #
def _read_yaml(path: PathLike) -> Any:
    import yaml  # local import: pure-python parser, no network

    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"failed to parse YAML at {p}: {exc}") from exc


def _parse(data: dict[str, Any]) -> Config:
    version = data.get("version", 1)
    if not isinstance(version, int):
        raise ConfigError("version must be an integer")

    return Config(
        version=version,
        review=_parse_review(data.get("review", {}) or {}),
        model_roles=_parse_model_roles(data.get("model_roles", {}) or {}),
        external_tools=_parse_external_tools(data.get("external_tools", {}) or {}),
        telemetry=_parse_telemetry(data.get("telemetry", {}) or {}),
    )


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{what} must be a mapping")
    return dict(value)


def _parse_review(raw: Any) -> ReviewConfig:
    block = _require_mapping(raw, "review")

    profile = block.get("profile", DEFAULT_PROFILE)
    if profile not in PROFILES:
        raise ConfigError(
            f"review.profile must be one of {PROFILES}, got {profile!r}"
        )

    gate = block.get("confidence_gate", DEFAULT_CONFIDENCE_GATE)
    if not isinstance(gate, (int, float)) or isinstance(gate, bool):
        raise ConfigError("review.confidence_gate must be a number")
    if not 0.0 <= float(gate) <= 1.0:
        raise ConfigError("review.confidence_gate must be within [0, 1]")

    incremental = block.get("incremental", True)
    if not isinstance(incremental, bool):
        raise ConfigError("review.incremental must be a boolean")

    path_filters = block.get("path_filters", []) or []
    if not _is_str_list(path_filters):
        raise ConfigError("review.path_filters must be a list of strings")

    lenses = block.get("lenses")
    if lenses is None:
        lenses = list(DEFAULT_LENSES)
    if not _is_str_list(lenses):
        raise ConfigError("review.lenses must be a list of strings")
    for lens in lenses:
        if lens not in LENSES:
            raise ConfigError(
                f"review.lenses contains unknown lens {lens!r}; allowed: {LENSES}"
            )

    instructions = _parse_path_instructions(block.get("path_instructions", []) or [])

    return ReviewConfig(
        profile=profile,
        confidence_gate=float(gate),
        incremental=incremental,
        path_filters=list(path_filters),
        path_instructions=instructions,
        lenses=list(lenses),
    )


def _parse_path_instructions(raw: Any) -> list[PathInstruction]:
    if not isinstance(raw, list):
        raise ConfigError("review.path_instructions must be a list")
    out: list[PathInstruction] = []
    for i, item in enumerate(raw):
        entry = _require_mapping(item, f"review.path_instructions[{i}]")
        if "path" not in entry or "instructions" not in entry:
            raise ConfigError(
                f"review.path_instructions[{i}] requires 'path' and 'instructions'"
            )
        out.append(
            PathInstruction(path=str(entry["path"]), instructions=str(entry["instructions"]))
        )
    return out


def _parse_model_roles(raw: Any) -> dict[str, ModelRole]:
    block = _require_mapping(raw, "model_roles")
    roles: dict[str, ModelRole] = {}
    for role, spec in block.items():
        entry = _require_mapping(spec, f"model_roles.{role}")
        if "model" not in entry or not entry["model"]:
            raise ConfigError(f"model_roles.{role} requires a non-empty 'model'")
        options = {k: v for k, v in entry.items() if k not in ("model", "region")}
        roles[str(role)] = ModelRole(
            model=str(entry["model"]),
            region=(str(entry["region"]) if entry.get("region") is not None else None),
            options=options,
        )
    return roles


def _parse_external_tools(raw: Any) -> ExternalTools:
    block = _require_mapping(raw, "external_tools")
    enabled = block.get("enabled", []) or []
    if not _is_str_list(enabled):
        raise ConfigError("external_tools.enabled must be a list of strings")
    return ExternalTools(enabled=list(enabled))


def _parse_telemetry(raw: Any) -> Telemetry:
    block = _require_mapping(raw, "telemetry")
    enabled = block.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("telemetry.enabled must be a boolean")
    mode = block.get("mode", "opt-out")
    if mode not in TELEMETRY_MODES:
        raise ConfigError(
            f"telemetry.mode must be one of {TELEMETRY_MODES}, got {mode!r}"
        )
    return Telemetry(enabled=enabled, mode=mode)


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)
