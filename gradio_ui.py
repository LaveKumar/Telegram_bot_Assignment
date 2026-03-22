"""
gradio_ui.py — Local Gradio debug interface for the Vision pipeline.

Run independently (no Telegram token needed):
  python gradio_ui.py

Opens http://localhost:7860 where you can:
  • Upload any image
  • See LLaVA caption + tags
  • Browse the cache (recent captions)
  • Inspect semantic similarity between two captions
"""

import gradio as gr
import numpy as np

from cache  import ImageCache
from vision import describe_image

cache = ImageCache()


# ── Tab 1: Image Description ───────────────────────────────────────────────────

def run_vision(image):
    """Called by Gradio with a PIL Image."""
    if image is None:
        return "No image provided.", "", ""

    import io
    from PIL import Image as PILImage

    buf = io.BytesIO()
    if not isinstance(image, PILImage.Image):
        image = PILImage.fromarray(image)
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    sha = ImageCache.sha256(img_bytes)

    # Layer-1 exact
    cached = cache.get_by_hash(sha)
    if cached:
        return cached["caption"], ", ".join(cached["tags"]), "⚡ Exact cache hit"

    # Ollama
    result = describe_image(img_bytes)

    # Layer-2 semantic
    sem = cache.get_by_embedding(result["caption"])
    if sem:
        return sem["caption"], ", ".join(sem["tags"]), "⚡ Semantic cache hit"

    cache.store(sha, result)
    return result["caption"], ", ".join(result["tags"]), "✅ New result from LLaVA"


# ── Tab 2: Cache Browser ───────────────────────────────────────────────────────

def browse_cache():
    rows = cache.recent_captions(10)
    if not rows:
        return "Cache is empty."
    lines = []
    for i, r in enumerate(rows, 1):
        lines.append(f"**{i}.** {r['caption']}\n   🏷 {', '.join(r['tags'])}")
    return "\n\n".join(lines)


# ── Tab 3: Semantic Similarity Tester ─────────────────────────────────────────

def check_similarity(caption_a: str, caption_b: str) -> str:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    va = model.encode(caption_a, normalize_embeddings=True)
    vb = model.encode(caption_b, normalize_embeddings=True)
    score = float(np.dot(va, vb))
    verdict = "✅ Would be a semantic cache HIT" if score >= 0.92 else "❌ Would MISS (too different)"
    return f"Cosine similarity: **{score:.4f}**\n\n{verdict}"


# ── Build UI ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="Vision Bot Debug UI") as demo:
    gr.Markdown("# 🖼 Vision Captioning Bot — Debug UI\nTest the LLaVA pipeline locally without Telegram.")

    with gr.Tab("📷 Describe Image"):
        img_in     = gr.Image(label="Upload Image", type="pil")
        btn        = gr.Button("Describe", variant="primary")
        caption_out = gr.Textbox(label="Caption")
        tags_out    = gr.Textbox(label="Tags")
        status_out  = gr.Textbox(label="Cache status")
        btn.click(run_vision, inputs=img_in, outputs=[caption_out, tags_out, status_out])

    with gr.Tab("🗃 Cache Browser"):
        refresh_btn   = gr.Button("Refresh")
        cache_display = gr.Markdown()
        refresh_btn.click(browse_cache, outputs=cache_display)
        demo.load(browse_cache, outputs=cache_display)

    with gr.Tab("🔬 Similarity Tester"):
        gr.Markdown("Check if two captions would hit the semantic cache (threshold = 0.92).")
        cap_a   = gr.Textbox(label="Caption A")
        cap_b   = gr.Textbox(label="Caption B")
        sim_btn = gr.Button("Compare")
        sim_out = gr.Markdown()
        sim_btn.click(check_similarity, inputs=[cap_a, cap_b], outputs=sim_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
