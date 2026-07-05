import sqlite3
import bcrypt
from datetime import datetime

from config import DB_PATH, DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password BLOB NOT NULL,
            role TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            display_name TEXT,
            file_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            status TEXT NOT NULL,
            pages INTEGER DEFAULT 0,
            chunks INTEGER DEFAULT 0,
            structured_tables INTEGER DEFAULT 0,
            ocr_future_items INTEGER DEFAULT 0,
            created_at DATETIME NOT NULL,
            updated_at DATETIME
        )
    """)

    c.execute("PRAGMA table_info(documents)")
    cols = [row[1] for row in c.fetchall()]

    if "display_name" not in cols:
        c.execute("ALTER TABLE documents ADD COLUMN display_name TEXT")
    if "structured_tables" not in cols:
        c.execute("ALTER TABLE documents ADD COLUMN structured_tables INTEGER DEFAULT 0")
    if "ocr_future_items" not in cols:
        c.execute("ALTER TABLE documents ADD COLUMN ocr_future_items INTEGER DEFAULT 0")
    if "updated_at" not in cols:
        c.execute("ALTER TABLE documents ADD COLUMN updated_at DATETIME")

    c.execute("SELECT username FROM users WHERE username=?", (DEFAULT_ADMIN_USERNAME,))
    if not c.fetchone():
        hashed = bcrypt.hashpw(DEFAULT_ADMIN_PASSWORD.encode("utf-8"), bcrypt.gensalt())
        c.execute(
            "INSERT INTO users(username, password, role) VALUES (?, ?, ?)",
            (DEFAULT_ADMIN_USERNAME, hashed, "admin")
        )

    conn.commit()
    conn.close()


def create_user(username: str, password: str, role: str = "user") -> bool:
    username = (username or "").strip()
    if not username or not password:
        return False

    conn = get_conn()
    c = conn.cursor()

    try:
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        c.execute(
            "INSERT INTO users(username, password, role) VALUES (?, ?, ?)",
            (username, hashed, role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate(username: str, password: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT password, role FROM users WHERE username=?", ((username or "").strip(),))
    row = c.fetchone()
    conn.close()

    if row and bcrypt.checkpw(password.encode("utf-8"), row[0]):
        return row[1]

    return None


def save_message(username: str, session_id: str, role: str, content: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        INSERT INTO chat_history(username, session_id, role, content, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (username, session_id, role, content, datetime.now()))

    conn.commit()
    conn.close()


def get_chat_history(username: str, session_id: str, limit: int = 30):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        SELECT role, content
        FROM chat_history
        WHERE username=? AND session_id=?
        ORDER BY timestamp ASC
    """, (username, session_id))

    rows = c.fetchall()
    conn.close()

    return [{"role": row[0], "content": row[1]} for row in rows][-limit:]


def upsert_document(
    doc_id,
    file_name,
    file_path,
    source_type,
    status,
    pages=0,
    chunks=0,
    structured_tables=0,
    ocr_future_items=0,
):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        INSERT INTO documents(
            doc_id, file_name, display_name, file_path, source_type,
            status, pages, chunks, structured_tables, ocr_future_items,
            created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            file_name=excluded.file_name,
            file_path=excluded.file_path,
            source_type=excluded.source_type,
            status=excluded.status,
            pages=excluded.pages,
            chunks=excluded.chunks,
            structured_tables=excluded.structured_tables,
            ocr_future_items=excluded.ocr_future_items,
            updated_at=excluded.updated_at
    """, (
        doc_id,
        file_name,
        file_name,
        file_path,
        source_type,
        status,
        pages,
        chunks,
        structured_tables,
        ocr_future_items,
        datetime.now(),
        datetime.now()
    ))

    conn.commit()
    conn.close()


def get_document(doc_id: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        SELECT doc_id, file_name, display_name, file_path, source_type,
               status, pages, chunks, structured_tables, ocr_future_items,
               created_at, updated_at
        FROM documents
        WHERE doc_id=?
    """, (doc_id,))

    row = c.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "doc_id": row[0],
        "file_name": row[1],
        "display_name": row[2] or row[1],
        "file_path": row[3],
        "source_type": row[4],
        "status": row[5],
        "pages": row[6],
        "chunks": row[7],
        "structured_tables": row[8],
        "ocr_future_items": row[9],
        "created_at": row[10],
        "updated_at": row[11],
    }


def document_is_ready(doc_id: str) -> bool:
    doc = get_document(doc_id)
    return bool(doc and doc["status"] == "ready" and doc["chunks"] > 0)


def list_documents(include_deleted: bool = False):
    conn = get_conn()
    c = conn.cursor()

    if include_deleted:
        c.execute("""
            SELECT doc_id, file_name, display_name, source_type, status,
                   pages, chunks, structured_tables, ocr_future_items,
                   created_at, updated_at
            FROM documents
            ORDER BY created_at DESC
        """)
    else:
        c.execute("""
            SELECT doc_id, file_name, display_name, source_type, status,
                   pages, chunks, structured_tables, ocr_future_items,
                   created_at, updated_at
            FROM documents
            WHERE status != 'deleted'
            ORDER BY created_at DESC
        """)

    rows = c.fetchall()
    conn.close()

    return [
        {
            "doc_id": row[0],
            "file_name": row[1],
            "display_name": row[2] or row[1],
            "source_type": row[3],
            "status": row[4],
            "pages": row[5],
            "chunks": row[6],
            "structured_tables": row[7],
            "ocr_future_items": row[8],
            "created_at": row[9],
            "updated_at": row[10],
        }
        for row in rows
    ]


def rename_document(doc_id: str, display_name: str) -> bool:
    display_name = (display_name or "").strip()
    if not display_name:
        return False

    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        UPDATE documents
        SET display_name=?, updated_at=?
        WHERE doc_id=?
    """, (display_name, datetime.now(), doc_id))

    changed = c.rowcount > 0
    conn.commit()
    conn.close()

    return changed


def mark_document_deleted(doc_id: str) -> bool:
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        UPDATE documents
        SET status='deleted', updated_at=?
        WHERE doc_id=?
    """, (datetime.now(), doc_id))

    changed = c.rowcount > 0
    conn.commit()
    conn.close()

    return changed