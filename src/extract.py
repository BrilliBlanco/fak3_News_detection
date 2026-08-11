"""
Extra input sources for the fake news detector: pull article text from a
URL, or from a screenshot/image via OCR. Both just produce plain text,
which then goes through the exact same clean_text() + TF-IDF + model
pipeline as manually pasted text - no change to the classifier itself.

Both third-party stacks (requests+bs4 for URLs, pytesseract+PIL for OCR) are
imported *inside* the functions on purpose. They are optional extras, and a
missing OCR engine on one teammate's laptop must not stop the core text
classifier from importing.
"""

import re

_WHITESPACE_RE = re.compile(r"\s+")

# Tags that never contain article prose
_JUNK_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form",
              "noscript", "iframe", "figcaption"]


def extract_text_from_url(url: str, timeout: int = 10, max_chars: int = 50000) -> str:
    """
    Fetch a URL and pull out visible article text (best-effort).

    Raises RuntimeError with an actionable message if the optional scraping
    dependencies are missing, or if the URL is unusable.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "URL input needs `requests` and `beautifulsoup4`. Install them with:\n"
            "    pip install -r requirements.txt"
        ) from exc

    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    if "html" not in resp.headers.get("Content-Type", "text/html").lower():
        raise RuntimeError(f"That URL returned {resp.headers.get('Content-Type')}, not HTML.")

    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(_JUNK_TAGS):
        tag.decompose()

    # Prefer a real <article>/<main> container when the page has one
    container = soup.find("article") or soup.find("main") or soup
    paragraphs = container.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)

    if len(text.strip()) < 50:
        # Fallback: some sites barely use <p> - take all body text
        body = soup.find("body")
        text = body.get_text(" ", strip=True) if body else soup.get_text(" ", strip=True)

    # Prepend the headline: it carries a lot of signal and the models were
    # trained on title + body concatenated.
    title = soup.find("title")
    if title:
        headline = title.get_text(" ", strip=True)
        if headline and headline.lower() not in text[:200].lower():
            text = f"{headline}. {text}"

    return _WHITESPACE_RE.sub(" ", text).strip()[:max_chars]


def extract_text_from_image(image_file) -> str:
    """
    OCR a screenshot/image into text.

    Requires the Tesseract OCR engine installed on the system (not just the
    pytesseract Python package) - see README for install instructions.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Screenshot input needs `pytesseract` and `pillow`. Install them with:\n"
            "    pip install -r requirements.txt\n"
            "You also need the Tesseract OCR engine itself - see the README."
        ) from exc

    image = Image.open(image_file)
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")

    try:
        text = pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or not on your PATH.\n"
            "  Linux/Kali: sudo apt install tesseract-ocr\n"
            "  Windows:    https://github.com/UB-Mannheim/tesseract/wiki"
        ) from exc

    return _WHITESPACE_RE.sub(" ", text).strip()
