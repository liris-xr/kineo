# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")


class Annotations(ABC, Generic[T], Iterable[T]):
    _annotations: list[T]

    def __init__(self, annotations: list[T]):
        self._annotations = annotations

    @property
    def annotations(self) -> list[T]:
        return self._annotations

    def __iter__(self) -> Iterator[T]:
        return iter(self._annotations)

    def __getitem__(self, index: int) -> T:
        return self._annotations[index]

    def __len__(self) -> int:
        return len(self._annotations)

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_dict(dict_data: dict) -> "Annotations[T]":
        raise NotImplementedError

    def first_or_default(self, default: T | None = None) -> T:
        return self._annotations[0] if len(self._annotations) > 0 else default
