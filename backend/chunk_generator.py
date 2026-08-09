from __future__ import annotations

import re
import uuid

from backend.chunk import Chunk
from backend.config import CHUNK_OVERLAP, MAX_CHUNK_SIZE
from backend.document import Document


class ChunkGenerator:
    def __init__(self, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def create_chunks(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        lines = document.content.splitlines()
        current_heading = "General"
        current_text: list[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if re.match(r"^#{1,6}\s", line):
                chunks.extend(self._split_section(document, current_heading, "\n".join(current_text)))
                current_heading = line.lstrip("#").strip()
                current_text = []
            else:
                current_text.append(line)

        chunks.extend(self._split_section(document, current_heading, "\n".join(current_text)))
        return chunks

    def _split_section(self, document: Document, heading: str, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        if not text.strip():
            return chunks

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.max_chunk_size
            chunk_text = text[start:end].strip()

            if not chunk_text:
                break

            source_key = f"{document.source_path or document.filename}|{heading}|{chunk_index}|{start}"
            chunk_id = uuid.uuid5(uuid.NAMESPACE_URL, source_key).hex

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    filename=document.filename,
                    category=document.category,
                    heading=heading,
                    text=chunk_text,
                    source_path=document.source_path,
                    page_number=document.page_number,
                )
            )

            start += self.max_chunk_size - self.overlap
            chunk_index += 1

        return chunks
