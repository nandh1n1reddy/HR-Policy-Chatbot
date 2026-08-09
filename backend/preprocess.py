import re


class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalize whitespace before chunking.
        """

        if not text:
            return ""

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = text.replace("\t", " ")

        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def remove_markdown_symbols(text: str) -> str:
        """
        Remove lightweight markdown markers while preserving headings and bullets.
        """

        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)", r"\1", text)
        text = re.sub(r"`(.*?)`", r"\1", text)
        return text

    @staticmethod
    def preprocess(text: str) -> str:
        text = TextPreprocessor.clean_text(text)
        text = TextPreprocessor.remove_markdown_symbols(text)
        return text


if __name__ == "__main__":
    sample = """
# Leave Policy


Employees      receive     12 casual leaves.


**Unused** leave may be carried forward.


"""

    processed = TextPreprocessor.preprocess(sample)
    print(processed)
