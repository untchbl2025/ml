from typing import Callable, Iterable, Dict, List, Optional, Tuple


class PatternRegistry:
    """Registry for synthetic pattern generators."""

    def __init__(self) -> None:
        self._patterns: Dict[str, Callable[..., object]] = {}
        self._next_wave: Dict[str, List[str]] = {}

    def register(
        self,
        name: str,
        generator: Callable[..., object],
        next_wave: Optional[Iterable[str]] = None,
    ) -> Callable[..., object]:
        """Register a pattern generator with optional follow-up waves."""
        self._patterns[name] = generator
        if next_wave is not None:
            if isinstance(next_wave, str):
                self._next_wave[name] = [next_wave]
            else:
                self._next_wave[name] = list(next_wave)
        return generator

    def generators(self) -> List[Tuple[Callable[..., object], str]]:
        """Return list of registered generators as ``(func, name)`` tuples."""
        return [(func, name) for name, func in self._patterns.items()]

    def get_next_wave(self, name: str) -> List[str]:
        """Return configured follow-up waves for ``name``."""
        return self._next_wave.get(name, [])


pattern_registry = PatternRegistry()


def register_pattern(name: str, next_wave: Optional[Iterable[str]] = None):
    """Decorator to register a pattern generator."""

    def decorator(func: Callable[..., object]):
        pattern_registry.register(name, func, next_wave)
        return func

    return decorator
