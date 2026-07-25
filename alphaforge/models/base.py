"""Unified, typed model contract for every alpha model (SF-S2-MR4).

Every model — sklearn wrapper, torch sequence model, ensemble, or naive
baseline — subclasses :class:`AlphaModel` and therefore shares one contract:

* ``fit(X, y)`` / ``predict(X)`` (abstract), plus optional ``predict_proba``,
  ``predict_uncertainty``, and ``feature_importance``;
* **validated fitted-state semantics** — a model must be fit before it predicts,
  enforced uniformly;
* **feature-schema checks** — the training feature schema is recorded at fit and
  a model's required features are checked at predict;
* **versioned metadata** via :meth:`AlphaModel.metadata`; and
* **deterministic serialization** via :meth:`AlphaModel.save` / :meth:`load`
  (a stable, sorted-key JSON container for models with JSON-safe state — every
  naive baseline — and a joblib fallback for arbitrary fitted estimators).

The shared fitted-state and schema behaviour is installed once, in
:meth:`AlphaModel.__init_subclass__`, which wraps each subclass's own ``fit`` and
``predict``. A re-entrancy guard (``_fitting``) means a model that calls its own
``predict`` *during* ``fit`` is never tripped by the not-fitted check.

Errors are typed (:class:`ModelError` and subclasses) so callers can react to a
specific failure rather than parsing a message.
"""

from __future__ import annotations

import functools
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Semantic version of the model contract; bumped when the interface changes.
CONTRACT_VERSION = "1.0.0"

_JSON_FORMAT = "alphaforge-model/json"
_JSON_FORMAT_VERSION = 1


class ModelError(Exception):
    """Base class for all model-contract errors."""


class NotFittedError(ModelError):
    """Raised when predict/serialization is attempted before ``fit``."""


class FeatureSchemaError(ModelError):
    """Raised when an input frame violates the model's feature schema."""


class InvalidLabelError(ModelError):
    """Raised when the target labels are unsupported for the model's task."""


class ProbabilityNotSupportedError(ModelError):
    """Raised when ``predict_proba`` is called on a non-probabilistic model."""


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable, versioned description of a model and its fitted state."""

    name: str
    task: str  # "regression" | "classification"
    contract_version: str
    fitted: bool
    n_features: int | None
    feature_names: tuple[str, ...] | None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (deterministic under ``sort_keys``)."""
        return {
            "name": self.name,
            "task": self.task,
            "contract_version": self.contract_version,
            "fitted": self.fitted,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names) if self.feature_names is not None else None,
            "params": dict(self.params),
        }


def _wrap_fit(fit_impl: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fit_impl)
    def wrapper(self: AlphaModel, X: pd.DataFrame, y: pd.Series, *args: Any, **kwargs: Any) -> Any:
        self._validate_fit_inputs(X, y)
        self._fitting = True
        try:
            result = fit_impl(self, X, y, *args, **kwargs)
        finally:
            self._fitting = False
        self._record_fit_schema(X)
        self._is_fitted = True
        return result

    wrapper.__af_contract_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_predict(predict_impl: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(predict_impl)
    def wrapper(self: AlphaModel, X: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        if not self._fitting:
            self._ensure_fitted()
            self._validate_predict_features(X)
        return predict_impl(self, X, *args, **kwargs)

    wrapper.__af_contract_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


class AlphaModel(ABC):
    """A model mapping a feature matrix to expected forward returns.

    X is a DataFrame of numeric features; rows may carry a (date, symbol)
    MultiIndex, which sequence models use to build causal windows. Tabular
    models simply ignore the index.
    """

    name: str = "alpha_model"
    #: "regression" (predict expected returns) or "classification" (predict_proba).
    task: str = "regression"
    #: Sequence models set this True so the driver attaches a (date, symbol) index.
    needs_sequence_index: bool = False
    #: True for models whose prediction ignores the feature columns (constants).
    feature_agnostic: bool = False

    # Fitted-state, managed by the fit/predict wrappers. Class-level defaults act
    # as the "not fitted" state until the first fit sets instance attributes.
    _is_fitted: bool = False
    _fitting: bool = False
    _feature_names: tuple[str, ...] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for attr, wrap in (("fit", _wrap_fit), ("predict", _wrap_predict)):
            impl = cls.__dict__.get(attr)
            if impl is not None and not getattr(impl, "__af_contract_wrapped__", False):
                setattr(cls, attr, wrap(impl))

    # ----- required interface ------------------------------------------------
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> AlphaModel: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    # ----- optional interface (typed defaults) -------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of an up-move for classification models.

        Regression models do not support probabilities and raise
        :class:`ProbabilityNotSupportedError`.
        """
        raise ProbabilityNotSupportedError(
            f"{self.name} is a {self.task} model and does not implement predict_proba"
        )

    def predict_uncertainty(self, X: pd.DataFrame) -> np.ndarray | None:
        """Optional per-row predictive uncertainty (standard deviation)."""
        return None

    def feature_importance(self) -> pd.Series | None:
        """Optional per-feature importance (higher = more important)."""
        return None

    def get_params(self) -> dict[str, Any]:
        """JSON-safe constructor parameters used to rebuild the model."""
        return {}

    def required_features(self) -> tuple[str, ...] | None:
        """Features that must be present at predict time.

        ``None`` means "the full training schema" (recorded at fit). Models that
        consume only specific columns override this to return just those.
        """
        return None

    # ----- fitted-state and schema helpers -----------------------------------
    def _validate_fit_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not isinstance(X, pd.DataFrame):
            raise FeatureSchemaError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise InvalidLabelError("y must be a pandas Series")
        if len(X) != len(y):
            raise FeatureSchemaError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        if len(X) == 0:
            raise FeatureSchemaError("cannot fit on an empty training set")
        try:
            x_values = X.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise FeatureSchemaError("all feature columns must be numeric") from exc
        if np.isinf(x_values).any():
            raise FeatureSchemaError("features contain non-finite (inf) values")
        try:
            y_values = y.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidLabelError("labels must be numeric") from exc
        if not np.isfinite(y_values).all():
            raise InvalidLabelError("labels contain non-finite (NaN or inf) values")

    def _record_fit_schema(self, X: pd.DataFrame) -> None:
        if isinstance(X, pd.DataFrame):
            self._feature_names = tuple(str(c) for c in X.columns)

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(f"{self.name} must be fit before this operation")

    def _validate_predict_features(self, X: pd.DataFrame) -> None:
        if self.feature_agnostic or not isinstance(X, pd.DataFrame):
            return
        required = self.required_features()
        if required is None:
            required = self._feature_names
        if required is None:
            return
        missing = [column for column in required if column not in X.columns]
        if missing:
            raise FeatureSchemaError(f"{self.name}: missing required features {missing}")

    # ----- metadata and serialization ----------------------------------------
    def metadata(self) -> ModelMetadata:
        """Return this model's versioned metadata."""
        return ModelMetadata(
            name=self.name,
            task=self.task,
            contract_version=CONTRACT_VERSION,
            fitted=self._is_fitted,
            n_features=len(self._feature_names) if self._feature_names is not None else None,
            feature_names=self._feature_names,
            params=self.get_params(),
        )

    def _fitted_state(self) -> dict[str, Any] | None:
        """JSON-safe fitted state, or ``None`` to use the joblib fallback.

        Naive baselines override this to return a small, deterministic dict.
        """
        return None

    def _load_fitted_state(self, state: dict[str, Any]) -> None:  # noqa: B027
        """Restore fitted state produced by :meth:`_fitted_state`.

        An intentional no-op default: models with no JSON state (or that use the
        joblib fallback) need not override it.
        """

    def save(self, path: str | Path) -> None:
        """Serialize the model to ``path``.

        Models with JSON-safe state (every naive baseline) are written as a
        deterministic, sorted-key JSON container so two saves of an
        equally-fitted model are byte-identical. Other models fall back to
        joblib, which round-trips but is not byte-deterministic.
        """
        path = Path(path)
        state = self._fitted_state()
        if state is not None:
            container = {
                "format": _JSON_FORMAT,
                "format_version": _JSON_FORMAT_VERSION,
                "metadata": self.metadata().to_dict(),
                "state": state,
            }
            path.write_text(
                json.dumps(container, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            import joblib

            joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> AlphaModel:
        """Load a model saved by :meth:`save` (JSON or joblib)."""
        path = Path(path)
        with open(path, "rb") as handle:
            first_byte = handle.read(1)
        if first_byte == b"{":
            container = json.loads(path.read_text(encoding="utf-8"))
            from alphaforge.models.registry import create_model  # lazy: avoid import cycle

            meta = container["metadata"]
            model = create_model(meta["name"], **meta["params"])
            model._load_fitted_state(container["state"])
            names = meta["feature_names"]
            model._feature_names = tuple(names) if names is not None else None
            model._is_fitted = bool(meta["fitted"])
            return model
        import joblib

        loaded = joblib.load(path)
        if not isinstance(loaded, AlphaModel):
            raise ModelError(f"{path} did not contain an AlphaModel")
        return loaded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={self._is_fitted})"
