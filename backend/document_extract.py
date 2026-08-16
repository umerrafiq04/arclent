"""Extracts plain text from an uploaded JD file (.txt / .pdf) for the AI recruiter chat.

Never touches the filesystem — everything happens on the in-memory upload bytes — and never
surfaces a filesystem path back to the client. Callers should catch DocumentExtractError and
turn it into a clean 4xx response; this module never lets a raw parser exception escape.
"""

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_EXTRACTED_CHARS = 8000  # keep prompts bounded even for long documents
SUPPORTED_EXTENSIONS = (".txt", ".pdf")


class DocumentExtractError(Exception):
    """Raised for any recoverable upload problem — message is safe to show the recruiter."""


def _extract_txt(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return content.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise DocumentExtractError(
                "I couldn't read that text file — it doesn't look like plain text."
            ) from exc


def _extract_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
    except PdfReadError as exc:
        raise DocumentExtractError(
            "I couldn't open that PDF — it may be corrupted or password-protected."
        ) from exc
    except Exception as exc:
        raise DocumentExtractError("I couldn't open that PDF.") from exc

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            continue  # skip an unreadable page rather than fail the whole document
    return "\n".join(pages_text)


def extract_text(filename: str, content: bytes) -> str:
    if not content:
        raise DocumentExtractError("That file appears to be empty.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise DocumentExtractError("That file is too large — please upload something under 5 MB.")

    lower_name = (filename or "").lower()
    if lower_name.endswith(".txt"):
        text = _extract_txt(content)
    elif lower_name.endswith(".pdf"):
        text = _extract_pdf(content)
    else:
        raise DocumentExtractError("I can only read .txt or .pdf files for now.")

    text = text.strip()
    if not text:
        raise DocumentExtractError(
            "I couldn't find any readable text in that file — it may be a scanned image. "
            "Could you paste the text directly, or try a different file?"
        )

    truncated = len(text) > MAX_EXTRACTED_CHARS
    if truncated:
        text = text[:MAX_EXTRACTED_CHARS] + "\n[document truncated for length]"
    return text
