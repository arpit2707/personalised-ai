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
from app.services.handoff_texts import PAUSING_REASONS, handoff_text, is_fixed_line, soft_again, soft_texts


# At most one "message mil gaya" per this long in a DM (the backend's
# AI_DOWN_NOTICE_MS): a long outage must not leave the chat silent for good.
AI_DOWN_NOTICE_SECONDS = 30 * 60


class ConversationConflict(Exception):
    pass


class ConversationStore:
    def __init__(self, path=None, clock=time.time):
        self.path = path
        self.clock = clock

    @contextmanager
    def db(self):
        # Expired rows are deleted by purge() (startup and every 60 s), not in
        # every transaction: reads skip anything past retention on their own.
        if settings.DATABASE_URL and self.path is None:
            from app.services.pg_conversations import pg_conversations
            with pg_conversations.transaction() as conn:
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
                    role TEXT NOT NULL, text TEXT NOT NULL, created REAL NOT NULL, message_id TEXT
                );
                CREATE INDEX IF NOT EXISTS message_history ON messages(conversation, created);
                CREATE TABLE IF NOT EXISTS preferences (
                    conversation TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    key TEXT NOT NULL, value TEXT NOT NULL, created REAL NOT NULL,
                    PRIMARY KEY(conversation, key)
                );
            ''')
            # Files created before message ids were stored.
            if 'message_id' not in {c['name'] for c in conn.execute("PRAGMA table_info(messages)")}:
                conn.execute("ALTER TABLE messages ADD COLUMN message_id TEXT")
            conn.executescript('''
                CREATE INDEX IF NOT EXISTS message_expiry ON messages(created);
                CREATE INDEX IF NOT EXISTS preference_expiry ON preferences(created);
                CREATE INDEX IF NOT EXISTS conversation_expiry ON conversations(updated);
            ''')
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _cutoff(self):
        return self.clock() - settings.MEMORY_RETENTION_DAYS * 86400

    def _purge(self, db):
        cutoff = self._cutoff()
        db.execute("DELETE FROM messages WHERE created <= ?", (cutoff,))
        db.execute("DELETE FROM preferences WHERE created <= ?", (cutoff,))
        db.execute("UPDATE conversations SET product=NULL, product_at=NULL WHERE product_at <= ?", (cutoff,))
        db.execute("UPDATE conversations SET lead_at=NULL WHERE lead_at <= ?", (cutoff,))
        db.execute("DELETE FROM conversations WHERE updated <= ?", (cutoff,))

    def purge(self):
        with self.db() as db:
            self._purge(db)

    def _fresh(self, row):
        """The row with expired product/lead observations hidden until the purge clears them."""
        row = dict(row)
        cutoff = self._cutoff()
        if row.get('product_at') is not None and row['product_at'] <= cutoff:
            row['product'] = row['product_at'] = None
        if row.get('lead_at') is not None and row['lead_at'] <= cutoff:
            row['lead_at'] = None
        return row

    def _row(self, db, brand, cid):
        row = db.execute("SELECT * FROM conversations WHERE brand=? AND id=?", (brand, cid)).fetchone()
        if not row:
            raise KeyError(cid)
        return self._fresh(row)

    def begin(self, req):
        """Stores the customer's message and returns the snapshot finish() must still match.

        The snapshot's message_row is the stored message this turn answers. A
        retry of the newest message (same message_id) is neither stored again
        nor a new turn, so an answer already in flight for it stays valid.
        """
        now = self.clock()
        with self.db() as db:
            # A chat idle past retention starts over, even before the next purge.
            db.execute("DELETE FROM conversations WHERE brand=? AND channel=? AND sender=? AND updated <= ?",
                       (req.brand_id, req.channel_type, req.sender_id, self._cutoff()))
            db.execute("INSERT OR IGNORE INTO conversations(id,brand,channel,sender,updated) VALUES(?,?,?,?,?)",
                       (str(uuid4()), req.brand_id, req.channel_type, req.sender_id, now))
            row = db.execute("SELECT * FROM conversations WHERE brand=? AND channel=? AND sender=?",
                             (req.brand_id, req.channel_type, req.sender_id)).fetchone()
            cid = row['id']
            if getattr(req, 'resume_if_pending', False) and row['status'] == 'pending' and row['agent'] is None:
                db.execute("UPDATE conversations SET status='ai',reason=NULL,unresolved=0 WHERE id=?", (cid,))
            message_id = getattr(req, 'message_id', None)
            last = self._last_user_message(db, cid)
            if not (message_id and last and last['message_id'] == message_id):
                db.execute("INSERT INTO messages(conversation,role,text,created,message_id) VALUES(?,?,?,?,?)",
                           (cid, 'user', req.message_text, now, message_id))
                db.execute("UPDATE conversations SET updated=?,version=version+1 WHERE id=?", (now, cid))
                last = self._last_user_message(db, cid)
            snapshot = self._row(db, req.brand_id, cid)
            snapshot['message_row'] = last['id']
            return snapshot

    @staticmethod
    def _last_user_message(db, cid):
        return db.execute("SELECT id,message_id FROM messages WHERE conversation=? AND role='user' "
                          "ORDER BY id DESC LIMIT 1", (cid,)).fetchone()

    def answered_later(self, brand, snapshot):
        """True when the customer wrote again after the message this snapshot answers."""
        if snapshot.get('message_row') is None:
            return False
        with self.db() as db:
            self._row(db, brand, snapshot['id'])
            return db.execute("SELECT id FROM messages WHERE conversation=? AND role='user' AND id>? LIMIT 1",
                              (snapshot['id'], snapshot['message_row'])).fetchone() is not None

    def context(self, brand, cid):
        with self.db() as db:
            self._row(db, brand, cid)
            cutoff = self._cutoff()
            rows = db.execute("SELECT role,text FROM messages WHERE conversation=? AND created > ? "
                              "ORDER BY id DESC LIMIT ?", (cid, cutoff, settings.HISTORY_MAX_MESSAGES)).fetchall()
            history, remaining = [], settings.HISTORY_MAX_CHARS
            for row in rows:
                # Hand-off and helpline lines are not part of the talk: a model
                # shown them as its own answers starts repeating them.
                if row['role'] == 'assistant' and is_fixed_line(row['text']):
                    continue
                if len(row['text']) > remaining:
                    break
                history.append(dict(row))
                remaining -= len(row['text'])
            prefs = db.execute("SELECT key,value FROM preferences WHERE conversation=? AND created > ?",
                               (cid, cutoff)).fetchall()
            return {'history': list(reversed(history)), 'preferences': {p['key']: p['value'] for p in prefs}}

    def finish(self, brand, snapshot, *, reply=None, reason=None, product=None, interested=False,
               unresolved=False, preferences=(), source_text='', flag=None, event_type='dm', language=None):
        now = self.clock()
        cid = snapshot['id']
        with self.db() as db:
            row = self._row(db, brand, cid)
            if row['version'] != snapshot['version'] or row['status'] != 'ai':
                raise ConversationConflict(cid)
            count = row['unresolved'] + 1 if unresolved else 0
            if count >= 2 and reason is None:
                reason = 'unresolved_query'
            # Soft reasons ("the team will confirm this") keep the AI answering,
            # and a flag (a hot purchase chat) only marks it for the seller.
            status = 'pending' if reason in PAUSING_REASONS else 'ai'
            stored = reason or flag
            priority = 'high' if stored == 'purchase_assistance' else 'normal'
            db.execute('''UPDATE conversations SET status=?, reason=?, priority=?, unresolved=?,
                       product=COALESCE(?,product), product_at=CASE WHEN ? IS NULL THEN product_at ELSE ? END,
                       lead_at=CASE WHEN ? THEN ? ELSE lead_at END, updated=?, version=version+1 WHERE id=?''',
                       (status, stored, priority, count, product, product, now, interested, now, now, cid))
            for pref in preferences:
                # Only explicitly stated, allowlisted preferences; no arbitrary memory instructions.
                if pref['evidence'].casefold() not in source_text.casefold():
                    continue
                if pref['value'].casefold() not in pref['evidence'].casefold():
                    continue
                db.execute('''INSERT INTO preferences VALUES(?,?,?,?) ON CONFLICT(conversation,key)
                           DO UPDATE SET value=excluded.value,created=excluded.created''',
                           (cid, pref['key'], pref['value'], now))
            if reason and not (reply and reason == 'crisis'):
                reply = handoff_text(reason, source_text, language)
                # Only in a DM: every comment is its own public thread and always
                # gets its line, even while the model is down.
                if reason not in PAUSING_REASONS and event_type == 'dm':
                    # A seller's message in between means the line is not a repeat.
                    last = db.execute("SELECT role,text,created FROM messages WHERE conversation=? "
                                      "AND role IN ('assistant','agent') ORDER BY id DESC LIMIT 1", (cid,)).fetchone()
                    # The same "team will confirm" line twice in a row reads like a bot stuck
                    # in a loop, and silence reads like a dead one: say it another way.
                    if last and last['role'] == 'assistant' and last['text'] in soft_texts(reason):
                        if reason != 'generation_unavailable':
                            reply = soft_again(source_text, language)
                        elif now - last['created'] < AI_DOWN_NOTICE_SECONDS:
                            # While the model is down, one "message mil gaya" per 30 minutes.
                            reply = None
            # The backend retries a comment the model could not answer and never
            # posts this line under it, so it is not something the customer read.
            if reply and not (reason == 'generation_unavailable' and event_type == 'comment'):
                db.execute("INSERT INTO messages(conversation,role,text,created) VALUES(?,?,?,?)",
                           (cid, 'assistant', reply, now))
            return self._row(db, brand, cid), reply

    @staticmethod
    def handoff_text(reason, customer_text=''):
        return handoff_text(reason, customer_text)

    def mark_crisis(self, brand, snapshot, reply):
        """Pauses the chat for the seller whatever state it was in, and records the helpline once."""
        now = self.clock()
        cid = snapshot['id']
        with self.db() as db:
            db.execute("UPDATE conversations SET status='pending', reason='crisis', priority='high', "
                       "updated=?, version=version+1 WHERE id=?", (now, cid))
            db.execute("INSERT INTO messages(conversation,role,text,created) VALUES(?,?,?,?)",
                       (cid, 'assistant', reply, now))
            return self._row(db, brand, cid)

    def detail(self, brand, cid):
        with self.db() as db:
            row = self._row(db, brand, cid)
            cutoff = self._cutoff()
            messages = [dict(m) for m in db.execute(
                "SELECT role,text,created FROM messages WHERE conversation=? AND created > ? ORDER BY id DESC LIMIT 100",
                (cid, cutoff))]
            row['messages'] = list(reversed(messages))
            row['preferences'] = {p['key']: p['value'] for p in db.execute(
                "SELECT key,value FROM preferences WHERE conversation=? AND created > ?", (cid, cutoff))}
            user_texts = [m['text'][:240] for m in row['messages'] if m['role'] == 'user'][-3:]
            row['summary'] = {'reason': row['reason'], 'product_sku': row['product'], 'recent_customer_messages': user_texts}
            return row

    def inbox(self, brand, limit=50, offset=0):
        with self.db() as db:
            return [self._fresh(r) for r in db.execute('''SELECT * FROM conversations WHERE brand=?
                       AND status IN ('pending','active') AND updated > ?
                       ORDER BY CASE priority WHEN 'high' THEN 0 ELSE 1 END, updated ASC LIMIT ? OFFSET ?''',
                       (brand, self._cutoff(), limit, offset))]

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
