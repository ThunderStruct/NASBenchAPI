"""Base class for all NAS benchmarks."""

import random
from typing import List, Dict, Any, Optional, Iterator


class NASBenchBase:
    """Abstract base for unified NAS benchmarks.

    Subclasses should implement the architecture I/O, sampling, iteration,
    and mutation interfaces.
    """
    def load(self, data_path: Optional[str] = None) -> "NASBenchBase":
        """Load the dataset from a path or environment variable.

        Args:
            data_path: Optional explicit path to the dataset.

        Returns:
            Self reference to the loaded API instance.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources (optional)."""
        pass

    # Reproducibility
    def set_seed(self, seed: int) -> None:
        """Set global RNG seed for reproducibility.

        Args:
            seed: Seed value to initialize random number generators.
        """
        random.seed(seed)

    # Introspection / metadata (common)
    def bench_name(self) -> str:
        """Return the benchmark short name.

        Returns:
            Short name such as 'nb101', 'nb201', or 'nb301'.
        """
        raise NotImplementedError

    def datasets(self) -> List[str]:
        """Return list of available datasets for this benchmark.

        Returns:
            List of dataset names (e.g., ['cifar10']).
        """
        return ['cifar10']

    def splits(self, dataset: str) -> List[str]:
        """Return supported splits for a dataset.

        Args:
            dataset: Dataset name.

        Returns:
            List of split names (e.g., ['train', 'val', 'test']).
        """
        return ['train', 'val', 'test']

    # Common architecture I/O surface
    def decode(self, encoding: Any) -> Any:
        """Decode native encoding into an architecture object.

        Args:
            encoding: Benchmark-native representation of an architecture.

        Returns:
            Decoded architecture object.
        """
        raise NotImplementedError

    def encode(self, arch: Any) -> Any:
        """Encode architecture object into native encoding.

        Args:
            arch: Architecture object.

        Returns:
            Benchmark-native encoding for the given architecture.
        """
        raise NotImplementedError

    def id(self, arch: Any) -> str:
        """Return a stable identifier for the architecture.

        Args:
            arch: Architecture object.

        Returns:
            Stable identifier string.
        """
        raise NotImplementedError

    # Sampling / enumeration
    def random_sample(self, n: int = 1, seed: Optional[int] = None) -> List[Any]:
        """Return random architectures from the search space or dataset.

        Args:
            n: Number of architectures to sample.
            seed: Optional random seed.

        Returns:
            List of sampled architectures.
        """
        raise NotImplementedError

    def iter_all(self) -> Iterator[Any]:
        """Iterate all available architectures, if supported.

        Returns:
            Iterator over architectures.
        """
        raise NotImplementedError

    # Mutation
    def mutate(self, arch: Any, rng, kind: Optional[str] = None) -> Any:
        """Return a one-edit mutation of the given architecture.

        Args:
            arch: Architecture to mutate.
            rng: Instance of random.Random to use.
            kind: Optional mutation kind (benchmark-defined).

        Returns:
            Mutated architecture.
        """
        raise NotImplementedError