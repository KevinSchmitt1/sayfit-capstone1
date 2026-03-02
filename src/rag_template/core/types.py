from dataclasses import dataclass, field


@dataclass(slots=True)
class Document:
    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)
