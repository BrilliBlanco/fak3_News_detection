"""
Extra input sources for the fake news detector: pull article text from a
URL, or from a screenshot/image via OCR. Both just produce plain text,
which then goes through the exact same clean_text() + TF-IDF + model
pipeline as manually pasted text - no change to the classifier itself.
"""

import requests
from bs4 import BeautifulSoup


def extract_text_from_url(url: str, timeout: int = 10) -> str:
    """Fetch a URL and pull out visible paragraph text (best-effort)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)

    if len(text.strip()) < 50:
        # Fallback: some sites don't use <p> tags much - grab all body text
        body = soup.find("body")
        text = body.get_text(" ", strip=True) if body else soup.get_text(" ", strip=True)

    return text.strip()


def extract_text_from_image(image_file) -> str:
    """
    OCR a screenshot/image into text.
    Requires the Tesseract OCR engine installed on the system (not just
    the pytesseract Python package) - see README for install instructions.
    """
    import pytesseract
    from PIL import Image

    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text.strip()
