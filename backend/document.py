from dataclasses import dataclass


@dataclass
class Document:
    filename: str
    category: str
    content: str
    source_path: str | None = None
    page_number: int | None = None
