"""Gradio chat UI.

Left: chat history + input. Right: image gallery for the most recent
turn's ranked results, each with a provenance caption and a "why
matched" tooltip drawn from the underlying record.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import gradio as gr

from imagecb.config import SETTINGS
from imagecb.extractors.dispatch import SUPPORTED_EXTS
from imagecb.ingest import ingest_paths
from imagecb.paths import resolve_image_file
from imagecb.formatting.assistant_reply import build_assistant_reply
from imagecb.retrieval.rerank import RankedResult
from imagecb.retrieval.session import ChatSession
from imagecb.storage import metadata_db, vector_store
from imagecb.uploads import save_uploads

logger = logging.getLogger(__name__)

_RERANK_SUPPORTED_REGIONS = (
    "us-east-1",
    "us-west-2",
    "ca-central-1",
    "eu-central-1",
    "ap-northeast-1",
)


def _format_query_error(exc: BaseException) -> str:
    msg = str(exc)
    name = type(exc).__name__
    lower = msg.lower()
    needs_rerank_hint = (
        "rerank" in lower
        or "model identifier is invalid" in lower
        or name in ("ValidationException", "NoCredentialsError", "AccessDeniedException")
    )
    if not needs_rerank_hint:
        return f"Error: {exc}"
    lines = [f"Error: {exc}", ""]
    lines.append(
        f"Reranker config: `AWS_REGION={SETTINGS.aws_region}`, "
        f"`RERANKER_MODEL={SETTINGS.reranker_model}`."
    )
    if SETTINGS.aws_region not in _RERANK_SUPPORTED_REGIONS:
        regions = ", ".join(_RERANK_SUPPORTED_REGIONS)
        lines.append(
            f"Cohere Rerank 3.5 is not available in `{SETTINGS.aws_region}`. "
            f"Set `AWS_REGION` in `.env` to one of: {regions}."
        )
    lines.append(
        "Enable `cohere.rerank-v3-5:0` in the Bedrock console for that region, "
        "or run `python -m imagecb.cli validate-reranker` to test access."
    )
    return "\n".join(lines)


_UPLOAD_FILE_TYPES = sorted(SUPPORTED_EXTS)


def _format_ingest_summary(stats: dict, *, staged: int, stage_errors: list[str]) -> str:
    lines: list[str] = []
    if stage_errors:
        lines.append("**Staging warnings**")
        for err in stage_errors:
            lines.append(f"- {err}")
        lines.append("")
    if staged == 0 and not stats.get("images_seen"):
        if not lines:
            return "No supported files to ingest."
        return "\n".join(lines)
    elapsed = stats.get("elapsed_sec", 0)
    processed = stats.get("images_added", 0) + stats.get("images_updated", 0)
    rate = ""
    if elapsed > 0 and processed > 0:
        rate = f" ({processed / elapsed:.2f} images/s)"
    lines.append(
        f"**Ingest complete** in {elapsed}s{rate}: "
        f"files={stats.get('files', 0)}, "
        f"images_seen={stats.get('images_seen', 0)}, "
        f"added={stats.get('images_added', 0)}, "
        f"updated={stats.get('images_updated', 0)}, "
        f"duplicates={stats.get('skipped_duplicates', 0)}, "
        f"errors={stats.get('errors', 0)}."
    )
    from imagecb.repair import format_post_repair_summary

    repair_line = format_post_repair_summary(stats.get("post_repair") or {})
    if repair_line:
        lines.append(f"**{repair_line}**")
    lines.append(
        "_PPTX/PDF: only embedded raster images are indexed. "
        "Re-upload with **Force re-ingest** to refresh captions._"
    )
    return "\n".join(lines)


def _gallery_items(results: List[RankedResult]) -> List[Tuple[str, str]]:
    reply = build_assistant_reply(results)
    by_id = {r.image_id: r for r in results}
    items: List[Tuple[str, str]] = []
    for card in reply.results:
        ranked = by_id.get(card.image_id)
        if ranked is None:
            continue
        path = resolve_image_file(ranked.record)
        if path is None:
            continue
        chips = " · ".join(card.provenance.chips())
        label = chips
        if card.caption:
            label += f"\n{card.caption}"
        if card.match_hint:
            label += f"\n{card.match_hint}"
        items.append((str(path), label))
    return items


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Imagecb - Multimodal Image Search") as demo:
        gr.Markdown(
            "# Imagecb\n"
            "Conversational search over your image corpus (standalone images plus "
            "images inside PPTX and PDF). Ask in natural language and refine across turns."
        )

        session_state = gr.State(ChatSession())

        with gr.Accordion("Add to corpus", open=False):
            gr.Markdown(
                "Upload images (`.png`, `.jpg`, …), `.pdf`, or `.pptx` files. "
                "Ingest runs here (~2 Bedrock calls per extracted image) and may take several minutes."
            )
            upload_files = gr.File(
                label="Files",
                file_count="multiple",
                file_types=list(_UPLOAD_FILE_TYPES),
            )
            with gr.Row():
                skip_caption_cb = gr.Checkbox(label="Skip captions (faster, weaker search)", value=False)
                skip_ocr_cb = gr.Checkbox(label="Skip OCR", value=False)
                force_cb = gr.Checkbox(label="Force re-ingest duplicates", value=False)
            ingest_btn = gr.Button("Ingest uploads", variant="primary")
            ingest_status_md = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                    buttons=["copy"],
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder='e.g. "Screenshots of dashboards from Q3_Review.pptx" or "only the ones modified this month"',
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=30, value=10, step=1, label="Max results"
                    )
                    clear_btn = gr.Button("Clear session", variant="secondary")
                status_md = gr.Markdown(_status_line())

            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="Results",
                    columns=2,
                    rows=3,
                    height=480,
                    object_fit="contain",
                    show_label=True,
                )
                details_md = gr.Markdown("")

        def on_send(user_text, top_k, session, chat_history):
            user_text = (user_text or "").strip()
            if not user_text:
                return chat_history, [], "", session, ""
            chat_history = list(chat_history or [])
            chat_history.append({"role": "user", "content": user_text})
            try:
                ask_result = session.ask(user_text, top_k=int(top_k))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Query failed")
                chat_history.append(
                    {"role": "assistant", "content": _format_query_error(exc)}
                )
                return chat_history, [], "", session, ""

            spec, results = ask_result.spec, ask_result.results
            reply = build_assistant_reply(results, spec)
            chat_history.append({"role": "assistant", "content": reply.message})
            details = _format_spec(spec)
            return chat_history, _gallery_items(results), details, session, ""

        def on_clear(session):
            try:
                session.reset()
            except Exception:  # noqa: BLE001
                session = ChatSession()
            return [], [], "", session, ""

        def on_ingest(files, skip_caption, skip_ocr, force, progress=gr.Progress()):
            if not files:
                return "Select one or more files to ingest.", _status_line()
            progress(0, desc="Saving uploads…")
            saved, stage_errors = save_uploads(files)
            if not saved:
                msg = "No supported files could be staged."
                if stage_errors:
                    msg += "\n\n" + "\n".join(f"- {e}" for e in stage_errors)
                return msg, _status_line()
            progress(0.1, desc="Ingesting (Bedrock calls per image)…")
            try:
                stats = ingest_paths(
                    saved,
                    skip_caption=bool(skip_caption),
                    skip_ocr=bool(skip_ocr),
                    force=bool(force),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Ingest failed")
                return f"**Ingest failed:** {exc}", _status_line()
            progress(1.0, desc="Done")
            summary = _format_ingest_summary(stats, staged=len(saved), stage_errors=stage_errors)
            return summary, _status_line()

        ingest_btn.click(
            on_ingest,
            inputs=[upload_files, skip_caption_cb, skip_ocr_cb, force_cb],
            outputs=[ingest_status_md, status_md],
        )

        send_btn.click(
            on_send,
            inputs=[msg, top_k_slider, session_state, chatbot],
            outputs=[chatbot, gallery, details_md, session_state, msg],
        )
        msg.submit(
            on_send,
            inputs=[msg, top_k_slider, session_state, chatbot],
            outputs=[chatbot, gallery, details_md, session_state, msg],
        )
        clear_btn.click(
            on_clear,
            inputs=[session_state],
            outputs=[chatbot, gallery, details_md, session_state, msg],
        )

    return demo


def _status_line() -> str:
    try:
        n = vector_store.count()
    except Exception:  # noqa: BLE001
        n = 0
    return (
        f"Indexed images: **{n}**. "
        "Use **Add to corpus** above or `python -m imagecb.cli ingest <path>` to index files."
    )


def _format_spec(spec) -> str:
    if spec is None:
        return ""
    lines = ["**Parsed query**"]
    lines.append(f"- semantic: `{spec.semantic_query}`")
    if spec.must_have_keywords:
        lines.append(f"- must have: {', '.join(spec.must_have_keywords)}")
    if spec.must_avoid_keywords:
        lines.append(f"- must avoid: {', '.join(spec.must_avoid_keywords)}")
    sf = spec.source_filters
    if sf.file_types or sf.filename_contains or sf.authors:
        parts = []
        if sf.file_types:
            parts.append("types=" + ",".join(sf.file_types))
        if sf.filename_contains:
            parts.append("filename~" + "|".join(sf.filename_contains))
        if sf.authors:
            parts.append("authors=" + ",".join(sf.authors))
        lines.append("- source filters: " + "; ".join(parts))
    tf = spec.time_filter
    if tf.before or tf.after:
        bits = []
        if tf.after:
            bits.append(f"after {tf.after.date().isoformat()}")
        if tf.before:
            bits.append(f"before {tf.before.date().isoformat()}")
        lines.append("- time: " + ", ".join(bits))
    if spec.is_refinement:
        lines.append("- mode: refining previous results")
    return "\n".join(lines)


def _gradio_allowed_paths() -> List[str]:
    paths = {
        str(SETTINGS.data_dir.resolve()),
        str(SETTINGS.image_cache_dir.resolve()),
    }
    for record in metadata_db.get_all_records():
        resolved = resolve_image_file(record)
        if resolved is not None:
            paths.add(str(resolved.parent.resolve()))
    return sorted(paths)


def launch(*, host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    demo = build_ui()
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(),
        allowed_paths=_gradio_allowed_paths(),
    )
