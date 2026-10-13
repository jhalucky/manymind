
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseMind(ABC):
    """Common interface for specialized ManyMind models."""

    @abstractmethod
    def fit(self, X: Any, y: Any = None) -> "BaseMind":
        """Train the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Generate predictions."""
        raise NotImplementedError

    @abstractmethod
    def score(self, X: Any) -> Any:
        """Return model-specific scores."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the model."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> "BaseMind":
        """Load a persisted model."""
        raise NotImplementedError
