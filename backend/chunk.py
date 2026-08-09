from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    filename: str
    category: str
    heading: str
    text: str
    source_path: str | None = None
    page_number: int | None = None
