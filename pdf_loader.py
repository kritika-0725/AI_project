from io import BytesIO

from pypdf import PdfReader


def extract_pdf_text(
    pdf_bytes: bytes
) -> str:

    reader = PdfReader(
        BytesIO(pdf_bytes)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        if text.strip():

            pages.append(
                f"[Page {page_number}]\n{text}"
            )

    return "\n\n".join(pages)
