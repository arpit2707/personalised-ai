"""SQLite memory and inbox. All lookups are scoped to an authenticated merchant.

The version check prevents an in-flight AI response from overriding human takeover.
No rolling summary is saved: inbox summaries are derived from unexpired records.
"""
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from app.core.config import settings


class ConversationConflict(Exception):
    pass


class ConversationStore:
    def __init__(self, path=None, clock=time.time):
        self.path = path
        self.clock = clock

    @contextmanager
    def db(self):
        if settings.DATABASE_URL and self.path is None:
            from app.services.pg_conversations import pg_conversations
            with pg_conversations.transaction() as conn:
                self._purge(conn)
                yield conn
            return
        path = self.path or settings.CONVERSATION_DB_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA secure_delete=ON")
        try:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, brand TEXT NOT NULL, channel TEXT NOT NULL,
                    sender TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ai',
                    reason TEXT, priority TEXT NOT NULL DEFAULT 'normal', agent TEXT,
                    product TEXT, product_at REAL, lead_at REAL,
                    unresolved INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(brand, channel, sender)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY, conversation TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL, text TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS message_history ON messages(conversation, created);
                CREATE TABLE IF NOT EXISTS preferences (
                    conversation TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    key TEXT NOT NULL, value TEXT NOT NULL, created REAL NOT NULL,
                    PRIMARY KEY(conversation, key)
                );
            ''')
            conn.execute("BEGIN IMMEDIATE")
            self._purge(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _purge(self, db):
        cutoff = self.clock() - settings.MEMORY_RETENTION_DAYS * 86400
        db.execute("DELETE FROM messages WHERE created <= ?", (cutoff,))
        db.execute("DELETE FROM preferences WHERE created <= ?", (cutoff,))
        db.execute("UPDATE conversations SET product=NULL, product_at=NULL WHERE product_at <= ?", (cutoff,))
        db.execute("UPDATE conversations SET lead_at=NULL WHERE lead_at <= ?", (cutoff,))
        db.execute("DELETE FROM conversations WHERE updated <= ?", (cutoff,))

    def purge(self):
        with self.db():
            pass

    @staticmethod
    def _row(db, brand, cid):
        row = db.execute("SELECT * FROM conversations WHERE brand=? AND id=?", (brand, cid)).fetchone()
        if not row:
            raise KeyError(cid)
        return dict(row)

    def begin(self, req):
        now = self.clock()
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO conversations(id,brand,channel,sender,updated) VALUES(?,?,?,?,?)",
                       (str(uuid4()), req.brand_id, req.channel_type, req.sender_id, now))
            row = db.execute("SELECT * FROM conversations WHERE brand=? AND channel=? AND sender=?",
                             (req.brand_id, req.channel_type, req.sender_id)).fetchone()
            cid = row['id']
            if getattr(req, 'resume_if_pending', False) and row['status'] == 'pending' and row['agent'] is None:
                db.execute("UPDATE conversations SET status='ai',reason=NULL,unresolved=0 WHERE id=?", (cid,))
            db.execute("INSERT INTO messages(conversation,role,text,created) VALUES(?,?,?,?)",
                       (cid, 'user', req.message_text, now))
            db.execute("UPDATE conversations SET updated=?,version=version+1 WHERE id=?", (now, cid))
            return self._row(db, req.brand_id, cid)

    def context(self, brand, cid):
        with self.db() as db:
            self._row(db, brand, cid)
            rows = db.execute("SELECT role,text FROM messages WHERE conversation=? ORDER BY id DESC LIMIT ?",
                              (cid, settings.HISTORY_MAX_MESSAGES)).fetchall()
            history, remaining = [], settings.HISTORY_MAX_CHARS
            for row in rows:
                if len(row['text']) > remaining:
                    break
                history.append(dict(row))
                remaining -= len(row['text'])
            prefs = db.execute("SELECT key,value FROM preferences WHERE conversation=?", (cid,)).fetchall()
            return {'history': list(reversed(history)), 'preferences': {p['key']: p['value'] for p in prefs}}

    def finish(self, brand, snapshot, *, reply=None, reason=None, product=None,
               interested=False, unresolved=False, preferences=(), source_text=''):
        now = self.clock()
        cid = snapshot['id']
        with self.db() as db:
            row = self._row(db, brand, cid)
            if row['version'] != snapshot['version'] or row['status'] != 'ai':
                raise ConversationConflict(cid)
            count = row['unresolved'] + 1 if unresolved else 0
            if count >= 2 and reason is None:
                reason = 'unresolved_query'
            status = 'pending' if reason else 'ai'
            priority = 'high' if reason == 'purchase_assistance' else 'normal'
            db.execute('''UPDATE conversations SET status=?, reason=?, priority=?, unresolved=?,
                       product=COALESCE(?,product), product_at=CASE WHEN ? IS NULL THEN product_at ELSE ? END,
                       lead_at=CASE WHEN ? THEN ? ELSE lead_at END, updated=?, version=version+1 WHERE id=?''',
                       (status, reason, priority, count, product, product, now, interested, now, now, cid))
            for pref in preferences:
                # Only explicitly stated, allowlisted preferences; no arbitrary memory instructions.
                if pref['evidence'].casefold() not in source_text.casefold():
                    continue
                if pref['value'].casefold() not in pref['evidence'].casefold():
                    continue
                db.execute('''INSERT INTO preferences VALUES(?,?,?,?) ON CONFLICT(conversation,key)
                           DO UPDATE SET value=excluded.value,created=excluded.created''',
                           (cid, pref['key'], pref['value'], now))
            if reason:
                reply = self.handoff_text(reason)
            if reply:
                db.execute("INSERT INTO messages(conversation,role,text,created) VALUES(?,?,?,?)",
                           (cid, 'assistant', reply, now))
            return self._row(db, brand, cid), reply

    @staticmethod
    def handoff_text(reason):
        prefix = "I apologize for the trouble. " if reason in ('complaint', 'order_support') else ''
        return prefix + "Your conversation is in the team's queue for review. An agent has not joined yet."

    def detail(self, brand, cid):
        with self.db() as db:
            row = self._row(db, brand, cid)
            messages = [dict(m) for m in db.execute(
                "SELECT role,text,created FROM messages WHERE conversation=? ORDER BY id DESC LIMIT 100", (cid,))]
            row['messages'] = list(reversed(messages))
            row['preferences'] = {p['key']: p['value'] for p in db.execute(
                "SELECT key,value FROM preferences WHERE conversation=?", (cid,))}
            user_texts = [m['text'][:240] for m in row['messages'] if m['role'] == 'user'][-3:]
            row['summary'] = {'reason': row['reason'], 'product_sku': row['product'], 'recent_customer_messages': user_texts}
            return row

    def inbox(self, brand, limit=50, offset=0):
        with self.db() as db:
            return [dict(r) for r in db.execute('''SELECT * FROM conversations WHERE brand=? AND status IN ('pending','active')
                       ORDER BY CASE priority WHEN 'high' THEN 0 ELSE 1 END, updated ASC LIMIT ? OFFSET ?''',
                       (brand, limit, offset))]

    def agent_action(self, brand, cid, agent, action, text=None):
        with self.db() as db:
            row = self._row(db, brand, cid)
            if action == 'claim':
                if row['status'] != 'pending' and not (row['status'] == 'active' and row['agent'] == agent):
                    raise ConversationConflict('Conversation is not available to claim')
                db.execute("UPDATE conversations SET status='active',agent=?,updated=?,version=version+1 WHERE id=?",
                           (agent, self.clock(), cid))
            else:
                if row['status'] != 'active' or row['agent'] != agent:
                    raise ConversationConflict('Only the assigned agent may message or release')
                if action == 'release':
                    db.execute("UPDATE conversations SET status='ai',reason=NULL,agent=NULL,unresolved=0,updated=?,version=version+1 WHERE id=?",
                               (self.clock(), cid))
                elif action == 'message':
                    db.execute("INSERT INTO messages(conversation,role,text,created) VALUES(?,?,?,?)",
                               (cid, 'agent', text, self.clock()))
                    db.execute("UPDATE conversations SET updated=?,version=version+1 WHERE id=?", (self.clock(), cid))
            return self._row(db, brand, cid)


conversation_store = ConversationStore()
