import os
import uuid
import html
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from config import UPLOAD_DIR, EXTRACTION_REPORT_DIR
from database import (
    init_db,
    create_user,
    authenticate,
    save_message,
    get_chat_history,
    list_documents,
)
from ingest import ingest_pdf, index_exists
from rag_engine import get_rag_answer, get_last_rag_metrics
from kb_manager import delete_document_from_faiss, rename_kb_document
from utils import safe_filename


st.set_page_config(
    page_title="SourceLens",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()



# SESSION STATE


if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "executor" not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=1)
if "upload_future" not in st.session_state:
    st.session_state.upload_future = None
if "pending_delete_doc_id" not in st.session_state:
    st.session_state.pending_delete_doc_id = None
if "last_citation_docs" not in st.session_state:
    st.session_state.last_citation_docs = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {}
if "active_panel" not in st.session_state:
    st.session_state.active_panel = "sources"



# CSS


st.markdown(
    """
<style>
    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="stHeader"] {
        background: transparent;
        height: 0rem;
    }

    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {
        display: none;
        visibility: hidden;
    }

    .block-container {
        max-width: 1420px;
        padding-top: 1rem;
        padding-bottom: 0.8rem;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    body {
        background: #ffffff;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 2px 18px 2px;
        border-bottom: 1px solid #eeeeee;
        margin-bottom: 18px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-mark {
        width: 38px;
        height: 38px;
        border-radius: 14px;
        background: #f4f1ec;
        border: 1px solid #e6ded3;
        color: #6a5135;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 18px;
    }

    .brand-title {
        color: #191919;
        font-size: 23px;
        font-weight: 850;
        letter-spacing: -0.045em;
        line-height: 1;
    }

    .brand-subtitle {
        margin-top: 4px;
        color: #777777;
        font-size: 12.5px;
    }

    .user-mini {
        color: #6c6c6c;
        font-size: 13px;
        text-align: right;
    }

    .content-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 350px;
        gap: 24px;
        align-items: start;
    }

    .welcome {
        max-width: 760px;
        margin: 12vh auto 0 auto;
        text-align: center;
    }

    .welcome-mark {
        width: 54px;
        height: 54px;
        border-radius: 18px;
        background: #f4f1ec;
        border: 1px solid #e6ded3;
        color: #6a5135;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 24px;
        margin: 0 auto 18px auto;
    }

    .welcome-title {
        font-size: 34px;
        font-weight: 850;
        letter-spacing: -0.06em;
        color: #191919;
        margin-bottom: 10px;
    }

    .welcome-copy {
        color: #6f6f6f;
        font-size: 15px;
        line-height: 1.65;
        max-width: 680px;
        margin: auto;
    }

    .quick-prompts {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 24px;
    }

    .quick-chip {
        padding: 9px 12px;
        border-radius: 999px;
        border: 1px solid #e7e7e7;
        background: #ffffff;
        color: #5f5f5f;
        font-size: 12.5px;
    }

    .msg-user-row {
        display: flex;
        justify-content: flex-end;
        margin: 16px 0;
    }

    .msg-assistant-row {
        display: flex;
        justify-content: flex-start;
        margin: 16px 0;
    }

    .msg-user {
        max-width: 76%;
        background: #191919;
        color: #ffffff;
        border-radius: 22px 22px 6px 22px;
        padding: 13px 16px;
        line-height: 1.58;
        font-size: 14px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.08);
    }

    .msg-assistant {
        max-width: 84%;
        background: #f7f7f7;
        color: #222222;
        border: 1px solid #ededed;
        border-radius: 22px 22px 22px 6px;
        padding: 14px 16px;
        line-height: 1.62;
        font-size: 14px;
    }

    .panel {
        border: 1px solid #eeeeee;
        border-radius: 20px;
        background: #ffffff;
        padding: 15px;
        margin-bottom: 14px;
    }

    .panel-title {
        color: #191919;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .panel-note {
        color: #777777;
        font-size: 12.4px;
        line-height: 1.5;
        margin-bottom: 12px;
    }

    .source-card {
        background: #ffffff;
        border: 1px solid #eeeeee;
        border-radius: 16px;
        padding: 13px;
        margin-bottom: 10px;
    }

    .source-title {
        color: #191919;
        font-size: 13px;
        font-weight: 760;
        overflow-wrap: anywhere;
    }

    .source-meta {
        color: #777777;
        font-size: 11.5px;
        line-height: 1.45;
        margin-top: 5px;
    }
    div[data-testid="stImage"] img {
    max-height: 260px;
    object-fit: contain;
    }

    .citation-image-box {
    border: 1px solid #eeeeee;
    border-radius: 16px;
    padding: 8px;
    margin-bottom: 12px;
    background: #fafafa;
    }

    .citation-image-note {
        color: #777777;
        font-size: 11.5px;
        margin: 6px 0 10px 0;
    }

    .notice {
        border-radius: 14px;
        padding: 11px 12px;
        font-size: 12.4px;
        line-height: 1.45;
        margin-bottom: 12px;
        font-weight: 650;
    }

    .notice-ok {
        background: #f0f8f2;
        color: #24623a;
        border: 1px solid #cfebd5;
    }

    .notice-info {
        background: #f4f7fb;
        color: #315f90;
        border: 1px solid #dae7f4;
    }

    .notice-warn {
        background: #fff8ee;
        color: #8a4b12;
        border: 1px solid #f2dfc6;
    }

    .notice-error {
        background: #fff2f3;
        color: #a8323b;
        border: 1px solid #f2cbd0;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
    }

    .metric-box {
        background: #ffffff;
        border: 1px solid #eeeeee;
        border-radius: 14px;
        padding: 11px;
    }

    .metric-label {
        color: #777777;
        font-size: 10.5px;
        margin-bottom: 4px;
    }

    .metric-value {
        color: #191919;
        font-size: 18px;
        font-weight: 800;
    }

    .login-wrap {
        max-width: 430px;
        margin: 12vh auto 0 auto;
        text-align: center;
    }

    .login-mark {
        width: 52px;
        height: 52px;
        border-radius: 18px;
        background: #f4f1ec;
        border: 1px solid #e6ded3;
        color: #6a5135;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 24px;
        margin: 0 auto 18px auto;
    }

    .login-title {
        color: #191919;
        font-size: 34px;
        font-weight: 850;
        letter-spacing: -0.06em;
        margin-bottom: 9px;
    }

    .login-note {
        color: #6f6f6f;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 26px;
    }

    .stButton > button {
        border-radius: 13px;
        border: 1px solid #dddddd;
        background: #ffffff;
        color: #242424;
        min-height: 40px;
        font-weight: 650;
    }

    .stButton > button:hover {
        border-color: #cfcfcf;
        background: #f7f7f7;
        color: #191919;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #d8d8d8;
        border-radius: 15px;
        padding: 8px;
    }

    input, textarea {
        border-radius: 13px !important;
    }

    @media (max-width: 1050px) {
        .content-grid {
            grid-template-columns: 1fr;
        }
    }
    
</style>
""",
    unsafe_allow_html=True,
)



# HELPERS


def esc(value) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def html_text(value) -> str:
    return esc(value).replace("\n", "<br>")


def render_message(role: str, content: str):
    if role == "user":
        st.markdown(
            f"""
<div class="msg-user-row">
    <div class="msg-user">{html_text(content)}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
<div class="msg-assistant-row">
    <div class="msg-assistant">{html_text(content)}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_source_card(doc: dict):
    display_name = doc.get("display_name") or doc.get("file_name")
    file_name = doc.get("file_name", "")
    source_type = doc.get("source_type", "")
    status = doc.get("status", "")
    pages = doc.get("pages", 0)
    chunks = doc.get("chunks", 0)
    tables = doc.get("structured_tables", 0)
    future = doc.get("ocr_future_items", 0)

    st.markdown(
        f"""
<div class="source-card">
    <div class="source-title">{esc(display_name)}</div>
    <div class="source-meta">
        {esc(file_name)}<br>
        {esc(source_type)} · {esc(status)} · {pages} pages · {chunks} chunks<br>
        {tables} structured table(s) indexed · {future} future OCR item(s)
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_metric_summary(metrics: dict):
    if not metrics:
        st.markdown(
            '<div class="notice notice-info">Ask a question to see retrieval details.</div>',
            unsafe_allow_html=True,
        )
        return

    keyword_coverage = metrics.get("keyword_coverage", "-")
    citation_count = metrics.get("citation_count", "-")
    table_hits = metrics.get("table_hits", "-")
    mode = metrics.get("mode", "-")

    st.markdown(
        f"""
<div class="metric-grid">
    <div class="metric-box">
        <div class="metric-label">Question match</div>
        <div class="metric-value">{esc(keyword_coverage)}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Citations</div>
        <div class="metric-value">{esc(citation_count)}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Table hits</div>
        <div class="metric-value">{esc(table_hits)}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Mode</div>
        <div class="metric-value" style="font-size:12px;">{esc(mode)}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
def render_citations(citation_docs):
    if not citation_docs:
        st.markdown(
            '<div class="notice notice-warn">No source page was returned for this answer.</div>',
            unsafe_allow_html=True,
        )
        return

    
    # UI-level dedupe:
    # If same PDF + same page appears multiple times, show it once.
    # Prefer table/table_row citation over plain text citation.
    
    best_by_page = {}

    def citation_priority(doc):
        meta = doc.metadata or {}
        kind = meta.get("source_kind", "")

        if kind in {"table", "table_row"}:
            return 0

        return 1

    for doc in citation_docs:
        meta = doc.metadata or {}

        file_name = meta.get("file_name")
        page = meta.get("page")

        key = (file_name, page)

        if key not in best_by_page:
            best_by_page[key] = doc
        else:
            existing_doc = best_by_page[key]

            if citation_priority(doc) < citation_priority(existing_doc):
                best_by_page[key] = doc

    unique_citations = list(best_by_page.values())[:3]

    if not unique_citations:
        st.markdown(
            '<div class="notice notice-warn">No source page was returned for this answer.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="panel-title">Sources used</div>',
        unsafe_allow_html=True,
    )

    
    # To show all citations in one row.
    # Since images are inside columns, they appear smaller.
    
    cols = st.columns(len(unique_citations))

    for col, doc in zip(cols, unique_citations):
        meta = doc.metadata or {}

        file_name = meta.get("file_name", "Unknown file")
        page = meta.get("page", "-")
        source_kind = meta.get("source_kind", "-")
        table_number = meta.get("table_number")
        image_path = meta.get("image_path")

        table_text = f" · Table {table_number}" if table_number else ""

        with col:
            st.markdown(
                f"""
<div class="source-card">
    <div class="source-title">{esc(file_name)}</div>
    <div class="source-meta">
        Page {esc(page)} · {esc(source_kind)}{esc(table_text)}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            # Page screenshot preview for citation.
            # This is only UI preview. It is NOT OCR and is NOT sent to the LLM.
            if image_path and os.path.exists(image_path):
                st.image(
                    image_path,
                    caption=f"{file_name} — page {page}",
                    use_container_width=True,
                )
            else:
                st.markdown(
                    """
<div class="notice notice-warn">
    Page preview not available. Rebuild the knowledge base after adding image_path metadata.
</div>
""",
                    unsafe_allow_html=True,
                )

            with st.expander("Evidence"):
                st.write((doc.page_content or "")[:2200])

# LOGIN


def render_login():
    st.markdown(
        """
<div class="login-wrap">
    <div class="login-mark">◌</div>
    <div class="login-title">SourceLens</div>
    <div class="login-note">
        Text and table grounded PDF assistant with citations.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 1.1, 1])

    with center:
        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            username = st.text_input("Username", placeholder="Enter username", key="login_username")
            password = st.text_input("Password", type="password", placeholder="Enter password", key="login_password")

            if st.button("Continue", use_container_width=True, key="login_button"):
                role = authenticate(username, password)

                if role:
                    st.session_state.user = username.strip()
                    st.session_state.role = role
                    st.session_state.session_id = str(uuid.uuid4())
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_register:
            username = st.text_input("Create username", placeholder="Choose username", key="register_username")
            password = st.text_input("Create password", type="password", placeholder="Choose password", key="register_password")

            if st.button("Create account", use_container_width=True, key="register_button"):
                if create_user(username, password, role="user"):
                    st.success("Account created. Please sign in.")
                else:
                    st.error("Username already exists or input is invalid.")



# MAIN UI


def render_missing_kb():
    st.markdown(
        """
<div class="welcome">
    <div class="welcome-mark">◌</div>
    <div class="welcome-title">Knowledge library is not ready</div>
    <div class="welcome-copy">
        Run <b>python prepare_kb.py</b> once. After that, this app will load the saved FAISS index directly.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        f"""
<div class="topbar">
    <div class="brand">
        <div class="brand-mark">◌</div>
        <div>
            <div class="brand-title">SourceLens</div>
            <div class="brand-subtitle">Ask from extractable PDF text and structured tables</div>
        </div>
    </div>
    <div class="user-mini">{esc(st.session_state.user)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([0.12, 0.12, 0.12, 0.64])

    with c1:
        if st.button("New", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.last_citation_docs = []
            st.session_state.last_metrics = {}
            st.rerun()

    with c2:
        if st.button("Sources", use_container_width=True):
            st.session_state.active_panel = "sources"
            st.rerun()

    with c3:
        if st.button("Details", use_container_width=True):
            st.session_state.active_panel = "details"
            st.rerun()

    with c4:
        if st.button("Sign out", use_container_width=True):
            st.session_state.user = None
            st.session_state.role = None
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()


def render_chat_column():
    history = get_chat_history(
        st.session_state.user,
        st.session_state.session_id,
        limit=30,
    )

    if not history:
        st.markdown(
            """
<div class="welcome">
    <div class="welcome-mark">◌</div>
    <div class="welcome-title">What would you like to know?</div>
    <div class="welcome-copy">
        Ask from indexed PDF text and structured tables. Visual-only figures and OCR-heavy items are kept aside for future processing.
    </div>
    <div class="quick-prompts">
        <div class="quick-chip">What are the indicators of unaffordability of healthy diets?</div>
        <div class="quick-chip">Summarize Table 5 entry points for managing risk</div>
        <div class="quick-chip">What does Table 3 say about incentive-based interventions?</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        for msg in history:
            render_message(msg["role"], msg["content"])


def handle_upload():
    st.markdown('<div class="panel-title">Add PDF</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-note">Only extractable text and structured tables will be indexed. OCR/visual items are skipped for now.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("Add source", use_container_width=True):
            safe_name = safe_filename(uploaded_file.name)
            save_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.upload_future = st.session_state.executor.submit(
                ingest_pdf,
                str(save_path),
                "uploaded",
            )

            st.success("Upload started.")
            st.rerun()

    if st.session_state.upload_future is not None:
        if st.session_state.upload_future.done():
            try:
                chunks = st.session_state.upload_future.result()

                if chunks > 0:
                    st.markdown(
                        '<div class="notice notice-ok">Source added successfully.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="notice notice-info">No new content was added.</div>',
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.markdown(
                    f'<div class="notice notice-error">Upload failed: {esc(e)}</div>',
                    unsafe_allow_html=True,
                )

            st.session_state.upload_future = None

        else:
            st.markdown(
                '<div class="notice notice-info">Adding source in the background...</div>',
                unsafe_allow_html=True,
            )

            if st.button("Refresh", use_container_width=True):
                st.rerun()


def handle_manage(docs):
    st.markdown('<div class="panel-title">Manage</div>', unsafe_allow_html=True)

    if not docs:
        st.info("No sources available.")
        return

    doc_options = {}

    for d in docs:
        label = f"{d.get('display_name') or d.get('file_name')} | {d.get('doc_id')}"
        doc_options[label] = d.get("doc_id")

    selected_label = st.selectbox("Select source", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_label]
    current_name = selected_label.split(" | ")[0]

    new_name = st.text_input("Display name", value=current_name)

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Rename", use_container_width=True):
            ok = rename_kb_document(selected_doc_id, new_name)

            if ok:
                st.success("Renamed.")
                st.rerun()
            else:
                st.error("Could not rename.")

    with c2:
        if st.button("Remove", use_container_width=True):
            st.session_state.pending_delete_doc_id = selected_doc_id

    if st.session_state.pending_delete_doc_id == selected_doc_id:
        st.warning("Remove this source from answers?")

        y, n = st.columns(2)

        with y:
            if st.button("Confirm", use_container_width=True):
                ok = delete_document_from_faiss(selected_doc_id)

                if ok:
                    st.session_state.pending_delete_doc_id = None
                    st.success("Removed.")
                    st.rerun()
                else:
                    st.error("Remove failed.")

        with n:
            if st.button("Cancel", use_container_width=True):
                st.session_state.pending_delete_doc_id = None
                st.rerun()


def render_reports_note():
    st.markdown('<div class="panel-title">Future OCR items</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="panel-note">
    OCR/visual items are not indexed. Manifests are saved in:<br>
    <b>{esc(EXTRACTION_REPORT_DIR)}</b>
</div>
""",
        unsafe_allow_html=True,
    )


def render_side_column():
    docs = list_documents()
    ready_docs = [d for d in docs if d.get("status") == "ready"]

    st.markdown(
        f"""
<div class="panel">
    <div class="notice notice-ok">{len(ready_docs)} source(s) ready</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.session_state.active_panel == "details":
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Answer details</div>', unsafe_allow_html=True)
        render_metric_summary(st.session_state.last_metrics)

        with st.expander("Raw retrieval details"):
            st.json(st.session_state.last_metrics or {})

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        render_citations(st.session_state.last_citation_docs)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Library</div>', unsafe_allow_html=True)

        if not docs:
            st.markdown(
                '<div class="notice notice-error">No sources found. Prepare the library first.</div>',
                unsafe_allow_html=True,
            )
        else:
            for doc in docs[:8]:
                render_source_card(doc)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        handle_upload()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        handle_manage(docs)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        render_reports_note()
        st.markdown('</div>', unsafe_allow_html=True)



# APP FLOW


if not st.session_state.user:
    render_login()
    st.stop()

if not index_exists():
    render_missing_kb()
    st.stop()

render_header()

st.markdown('<div class="content-grid">', unsafe_allow_html=True)

left_col, right_col = st.columns([0.74, 0.26], gap="large")

with left_col:
    render_chat_column()

with right_col:
    render_side_column()

st.markdown("</div>", unsafe_allow_html=True)



# CHAT INPUT


question = st.chat_input("Ask about your sources...")

if question:
    save_message(
        st.session_state.user,
        st.session_state.session_id,
        "user",
        question,
    )

    render_message("user", question)

    with st.spinner("Checking sources..."):
        updated_history = get_chat_history(
            st.session_state.user,
            st.session_state.session_id,
            limit=30,
        )

        answer, citation_docs = get_rag_answer(question, updated_history)
        metrics = get_last_rag_metrics()

    save_message(
        st.session_state.user,
        st.session_state.session_id,
        "assistant",
        answer,
    )

    st.session_state.last_citation_docs = citation_docs or []
    st.session_state.last_metrics = metrics or {}

    render_message("assistant", answer)
    render_citations(st.session_state.last_citation_docs)

    with st.expander("Answer details", expanded=False):
        render_metric_summary(st.session_state.last_metrics)
        st.json(st.session_state.last_metrics or {})