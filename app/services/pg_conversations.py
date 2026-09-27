"""Postgres persistence for the same conversation lifecycle used by local SQLite."""
import re
import threading
from contextlib import contextmanager

from app.core.config import settings

SCHEMA = '''
CREATE SCHEMA IF NOT EXISTS ai;
CREATE TABLE IF NOT EXISTS ai.conversations (
    id TEXT PRIMARY KEY, brand TEXT NOT NULL, channel TEXT NOT NULL,
    sender TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ai', reason TEXT,
    priority TEXT NOT NULL DEFAULT 'normal', agent TEXT, product TEXT,
    product_at DOUBLE PRECISION, lead_at DOUBLE PRECISION,
    unresolved INTEGER NOT NULL DEFAULT 0, updated DOUBLE PRECISION NOT NULL,
    version INTEGER NOT NULL DEFAULT 0, UNIQUE(brand, channel, sender)
);
CREATE TABLE IF NOT EXISTS ai.messages (
    id BIGSERIAL PRIMARY KEY,
    conversation TEXT NOT NULL REFERENCES ai.conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL, text TEXT NOT NULL, created DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS conversation_message_history ON ai.messages(conversation, created);
CREATE TABLE IF NOT EXISTS ai.preferences (
    conversation TEXT NOT NULL REFERENCES ai.conversations(id) ON DELETE CASCADE,
    key TEXT NOT NULL, value TEXT NOT NULL, created DOUBLE PRECISION NOT NULL,
    PRIMARY KEY(conversation, key)
);
'''


def postgres_sql(sql):
    """Translate the store's static SQL, never parameter values."""
    ignore = 'INSERT OR IGNORE' in sql
    sql = sql.replace('INSERT OR IGNORE', 'INSERT').replace('?', '%s')
    sql = re.sub(r'\b(conversations|messages|preferences)\b', r'ai.\1', sql)
    # SQLite accepts integer CASE predicates; Postgres requires a boolean.
    sql = sql.replace('CASE WHEN %s THEN', 'CASE WHEN %s <> 0 THEN')
    sql = sql.replace('CASE WHEN %s IS NULL THEN', 'CASE WHEN %s::text IS NULL THEN')
    if ignore:
        sql += ' ON CONFLICT DO NOTHING'
    return sql


class ConnectionAdapter:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=()):
        # bool is integer in SQLite, whereas Postgres distinguishes the types.
        params = tuple(int(p) if isinstance(p, bool) else p for p in params)
        return self.conn.execute(postgres_sql(sql), params)


class PgConversationDatabase:
    def __init__(self):
        self._pool = None
        self._lock = threading.Lock()

    def pool(self):
        with self._lock:
            if self._pool is None:
                from psycopg.rows import dict_row
                from psycopg_pool import ConnectionPool
                pool = ConnectionPool(settings.DATABASE_URL, min_size=0, max_size=5,
                                      kwargs={'prepare_threshold': None, 'row_factory': dict_row}, open=True)
                try:
                    with pool.connection() as conn:
                        # Serialize initial DDL across workers.
                        conn.execute('SELECT pg_advisory_xact_lock(72422101)')
                        for statement in SCHEMA.split(';'):
                            if statement.strip():
                                conn.execute(statement)
                except Exception:
                    pool.close()
                    raise
                self._pool = pool
            return self._pool

    @contextmanager
    def transaction(self):
        with self.pool().connection() as conn:
            # Equivalent to SQLite's BEGIN IMMEDIATE for these short state changes.
            # Model/network calls always happen outside this transaction.
            conn.execute('SELECT pg_advisory_xact_lock(72422102)')
            yield ConnectionAdapter(conn)


pg_conversations = PgConversationDatabase()
