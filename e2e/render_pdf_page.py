#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pymupdf>=1.24.0",
# ]
# ///
"""
Render PDF pages to PNG images for OCR testing.

Usage:
    uv run render_pdf_page.py <pdf_path> <page_number> <output_path>

Example:
    uv run render_pdf_page.py ../examples/epstein/epstein-docs/court-2024-giuffre-v-maxwell.pdf 1 /tmp/epstein_page.png
"""

import sys
import fitz  # pymupdf


def render_page(pdf_path: str, page_num: int, output_path: str, dpi: int = 150) -> None:
    """Render a PDF page to a PNG image."""
    doc = fitz.open(pdf_path)

    if page_num < 1 or page_num > len(doc):
        print(f"Error: Page {page_num} out of range (1-{len(doc)})", file=sys.stderr)
        sys.exit(1)

    # Pages are 0-indexed in pymupdf
    page = doc[page_num - 1]

    # Render at specified DPI
    zoom = dpi / 72  # 72 is the default PDF DPI
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Save as PNG
    pix.save(output_path)

    print(f"Rendered page {page_num} ({pix.width}x{pix.height}) to {output_path}")
    doc.close()


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2])
    output_path = sys.argv[3]
    dpi = int(sys.argv[4]) if len(sys.argv) > 4 else 150

    render_page(pdf_path, page_num, output_path, dpi)


if __name__ == "__main__":
    main()
