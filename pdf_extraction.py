import tempfile
from dataclasses import dataclass

import flyte
import flyte.report
from flyte.io import File

# Define the task environment with required packages
env = flyte.TaskEnvironment(
    name="pdf_wordcloud",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "httpx",
        "pymupdf>=1.24.0",  # For PDF text extraction
        "wordcloud>=1.9.0",  # For wordcloud generation
        "matplotlib>=3.7.0",  # For plotting
    ),
    resources=flyte.Resources(cpu="4", memory="4Gi"),
    cache="auto",
)


@dataclass
class PipelineOutput:
    """Output of the PDF wordcloud pipeline."""
    summary: str
    extracted_text: File
    wordcloud_image: File


@env.task
async def download_pdf(url: str) -> File:
    """
    Download a PDF file from a URL.

    Args:
        url: The URL of the PDF file to download.

    Returns:
        The raw bytes of the PDF file.
    """
    import httpx

    print(f"Downloading PDF from: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
        return await File.from_local(local_path=tmp_path)


@env.task
async def extract_text(pdf_file: File) -> dict:
    """
    Extract all text from a PDF file.

    Args:
        pdf_bytes: The raw bytes of the PDF file.

    Returns:
        A dictionary containing extracted text and metadata.
    """
    import pymupdf

    result = {
        "pages": [],
        "metadata": {},
        "full_text": "",
    }

    # Open PDF from bytes
    # Read bytes from the async file handle and open with PyMuPDF (pymupdf)
    path = await pdf_file.download()
    doc = pymupdf.open(path, filetype="pdf")

    # Extract document metadata
    result["metadata"] = {
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "subject": doc.metadata.get("subject", ""),
        "keywords": doc.metadata.get("keywords", ""),
        "page_count": len(doc),
    }

    print(f"Processing PDF with {len(doc)} pages")

    all_text_parts = []
    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        result["pages"].append({
            "page_number": page_num + 1,
            "text": page_text,
            "char_count": len(page_text),
        })
        all_text_parts.append(page_text)

    full_text = "\n\n".join(all_text_parts)
    full_text = full_text.replace("Property of AmericanRhetoric.com", "")
    full_text = full_text.replace("AmericanRhetoric.com", "")
    result["full_text"] = full_text
    for page in result["pages"]:
        page["text"] = page["text"].replace("AmericanRhetoric.com", "")
    doc.close()

    print(f"Extracted {len(result['full_text'])} characters total")
    return result


@env.task
async def generate_wordcloud(text: str) -> File:
    """
    Generate a wordcloud image from the given text.

    Args:
        text: The text to create a wordcloud from.

    Returns:
        PNG image bytes of the wordcloud.
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import io

    print("Generating wordcloud...")

    # Create the wordcloud
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=200,
        min_font_size=10,
        max_font_size=150,
    ).generate(text)

    # Save to bytes
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    tmp_filename = "wordcloud.png"
    fig.savefig(tmp_filename, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # Output a Flyte File object
    return await File.from_local(local_path=tmp_filename)


@env.task(report=True)
async def generate_report(
    extracted_data: dict,
    wordcloud_image: File,
    source_url: str,
) -> str:
    """
    Generate a Flyte report displaying the extracted text and wordcloud.

    Args:
        extracted_data: Dictionary containing extracted text and metadata.
        wordcloud_image: PNG bytes of the wordcloud image.
        source_url: The original URL of the PDF.

    Returns:
        A summary string of the extraction.
    """
    import html
    import base64

    metadata = extracted_data.get("metadata", {})
    pages = extracted_data.get("pages", [])
    full_text = extracted_data.get("full_text", "")

    # Encode wordcloud image to base64 for embedding
    async with wordcloud_image.open("rb") as f:
        image_bytes = await f.read()
    wordcloud_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Calculate statistics
    total_chars = len(full_text)
    word_count = len(full_text.split())

    # Build the HTML report
    report_html = f"""
    <style>
        .container {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }}
        .header {{
            background: linear-gradient(90deg, #0f3460, #533483);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        .header h1 {{
            margin: 0 0 15px 0;
            font-size: 2.2em;
            background: linear-gradient(90deg, #e94560, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .summary-stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(233, 69, 96, 0.1);
            border: 1px solid rgba(233, 69, 96, 0.3);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            min-width: 150px;
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: 700;
            color: #e94560;
        }}
        .stat-label {{
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .wordcloud-section {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .wordcloud-section h2 {{
            color: #e94560;
            margin-top: 0;
        }}
        .wordcloud-img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .metadata {{
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #e94560;
        }}
        .metadata h2 {{
            color: #e94560;
            margin-top: 0;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metadata-item {{
            background: rgba(0,0,0,0.2);
            padding: 10px 15px;
            border-radius: 6px;
        }}
        .metadata-label {{
            color: #888;
            font-size: 0.85em;
            text-transform: uppercase;
        }}
        .metadata-value {{
            color: #fff;
            font-weight: 500;
            margin-top: 5px;
        }}
        .page-section {{
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            margin-bottom: 25px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .page-header {{
            background: linear-gradient(90deg, #533483, #0f3460);
            padding: 15px 20px;
            font-weight: 600;
            font-size: 1.1em;
        }}
        .page-content {{
            padding: 20px;
        }}
        .text-box {{
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            font-family: 'Fira Code', 'Monaco', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .source-link {{
            color: #4ecdc4;
            word-break: break-all;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            background: #0f3460;
            margin-left: 10px;
        }}
    </style>

    <div class="container">
        <div class="header">
            <h1>📄 PDF Text Extraction & Word Cloud Report</h1>
            <p>Source: <a href="{html.escape(source_url)}" class="source-link" target="_blank">{html.escape(source_url[:80])}...</a></p>
        </div>

        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{metadata.get("page_count", 0)}</div>
                <div class="stat-label">Pages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_chars:,}</div>
                <div class="stat-label">Characters</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{word_count:,}</div>
                <div class="stat-label">Words</div>
            </div>
        </div>

        <div class="wordcloud-section">
            <h2>☁️ Word Cloud</h2>
            <img src="data:image/png;base64,{wordcloud_b64}" class="wordcloud-img" alt="Word Cloud">
        </div>

        <div class="metadata">
            <h2>📋 Document Metadata</h2>
            <div class="metadata-grid">
    """

    for key, value in metadata.items():
        if value:
            report_html += f"""
                <div class="metadata-item">
                    <div class="metadata-label">{html.escape(str(key))}</div>
                    <div class="metadata-value">{html.escape(str(value))}</div>
                </div>
            """

    report_html += """
            </div>
        </div>
    """

    # Add page sections
    for page in pages:
        page_num = page.get("page_number", 0)
        page_text = page.get("text", "")
        char_count = page.get("char_count", 0)

        report_html += f"""
        <div class="page-section">
            <div class="page-header">
                📖 Page {page_num}
                <span class="badge">{char_count:,} chars</span>
            </div>
            <div class="page-content">
        """

        if page_text.strip():
            report_html += f"""
                <div class="text-box">{html.escape(page_text)}</div>
            """
        else:
            report_html += """
                <p style="color: #888; font-style: italic;">No text found on this page.</p>
            """

        report_html += """
            </div>
        </div>
        """

    report_html += "</div>"

    await flyte.report.log.aio(report_html, do_flush=True)

    summary = f"Extracted {word_count:,} words from {metadata.get('page_count', 0)} pages."
    return summary


@env.task
async def pdf_wordcloud_pipeline(pdf_url: str) -> PipelineOutput:
    """
    Main pipeline that orchestrates PDF text extraction and wordcloud generation.

    Args:
        pdf_url: URL of the PDF to process.

    Returns:
        PipelineOutput containing summary, extracted text file, and wordcloud image file.
    """
    import os

    print("Starting PDF text extraction and wordcloud pipeline...")

    # Step 1: Download the PDF
    pdf_file = await download_pdf(pdf_url)

    # Step 2: Extract text from the PDF
    extracted_data = await extract_text(pdf_file)

    # Step 3: Generate wordcloud from the extracted text
    wordcloud_file = await generate_wordcloud(extracted_data["full_text"])

    # Step 4: Generate the report
    summary = await generate_report(extracted_data, wordcloud_file, pdf_url)

    # Step 5: Save extracted text to a file
    text_file_path = os.path.join(tempfile.gettempdir(), "extracted_text.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(extracted_data["full_text"])
    extracted_text_file = await File.from_local(local_path=text_file_path)


    print(f"Pipeline complete: {summary}")
    return PipelineOutput(
        summary=summary,
        extracted_text=extracted_text_file,
        wordcloud_image=wordcloud_file,
    )


if __name__ == "__main__":
    flyte.init_from_config()
    deployments = flyte.deploy(env)
    print(f"Deployments: {deployments}")
