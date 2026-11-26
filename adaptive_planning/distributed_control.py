"""Authenticated control-plane requests and durable, fail-closed frame reservations.

Each frame is charged before sending. Missing settlements retain the complete
reservation across service restarts. Control traffic is outside the application
byte budget and is measured separately by the experiment wall timer.
"""

import hmac
import json
import sqlite3
import ssl
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from .byte_budget import ByteBudgetExceeded


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Ledger:
    def __init__(self, path, limit):
        if type(limit) is not int or not 0 < limit < 2**63:
            raise ValueError("Invalid budget")
        self.path = str(path)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS config (limit_bytes INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS denials (owner TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS frames
                (id TEXT PRIMARY KEY, owner TEXT NOT NULL, size INTEGER NOT NULL,
                 sent INTEGER, created REAL NOT NULL);
            """)
            row = db.execute("SELECT limit_bytes FROM config").fetchone()
            if row and row[0] != limit:
                raise ValueError("Existing ledger has a different limit")
            if not row:
                db.execute("INSERT INTO config VALUES (?)", (limit,))
        self.limit = limit

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level="IMMEDIATE")
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def apply(self, owner, operation, body):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            admitted = False
            if operation in ("reserve", "cancel"):
                ident, size = body["id"], body["size"]
                if (
                    not isinstance(ident, str)
                    or not 0 < len(ident) < 100
                    or type(size) is not int
                    or not 0 <= size < 2**63
                ):
                    raise ValueError("Invalid reservation")
                old = db.execute(
                    "SELECT owner,size,sent FROM frames WHERE id=?", (ident,)
                ).fetchone()
                if old:
                    if old[:2] != (owner, size):
                        raise ValueError("Conflicting reservation")
                    if operation == "cancel":
                        if old[2] not in (None, 0):
                            raise ValueError("Cannot cancel a sent frame")
                        db.execute("UPDATE frames SET sent=0 WHERE id=?", (ident,))
                    else:
                        admitted = old[2] is None
                elif operation == "cancel":
                    # Tombstone fences a delayed reserve that has not reached SQLite yet.
                    db.execute(
                        "INSERT INTO frames VALUES (?,?,?,?,?)",
                        (ident, owner, size, 0, time.time()),
                    )
                else:
                    charged = db.execute(
                        "SELECT COALESCE(SUM(COALESCE(sent,size)),0) FROM frames"
                    ).fetchone()[0]
                    if size > self.limit - charged:
                        db.execute("INSERT INTO denials VALUES (?,?)", (owner, time.time()))
                        db.commit()
                        raise ByteBudgetExceeded("Application byte quota exhausted")
                    admitted = True
                    db.execute(
                        "INSERT INTO frames VALUES (?,?,?,NULL,?)",
                        (ident, owner, size, time.time()),
                    )
            elif operation == "settle":
                old = db.execute(
                    "SELECT owner,size,sent FROM frames WHERE id=?", (body["id"],)
                ).fetchone()
                sent = body["sent"]
                if not old or old[0] != owner or type(sent) is not int or not 0 <= sent <= old[1]:
                    raise ValueError("Invalid frame settlement")
                if old[2] is not None and old[2] != sent:
                    raise ValueError("Conflicting settlement")
                db.execute("UPDATE frames SET sent=? WHERE id=?", (sent, body["id"]))
            elif operation != "state":
                raise ValueError("Unknown ledger operation")
            spent, reserved, count = db.execute(
                "SELECT COALESCE(SUM(sent),0), COALESCE(SUM(CASE WHEN sent IS NULL THEN size ELSE 0 END),0), COUNT(*) FROM frames"
            ).fetchone()
            return dict(
                admitted=admitted,
                limit=self.limit,
                spent=spent,
                reserved=reserved,
                frames=count,
                denied_frames=db.execute("SELECT COUNT(*) FROM denials").fetchone()[0],
            )


class ControlClient:
    def __init__(self, url, owner, key, ca=None, timeout=5):
        if not (url.startswith("https://") or url.startswith("http://127.0.0.1:")):
            raise ValueError("Control plane requires HTTPS or loopback")
        self.url, self.owner, self.key = url.rstrip("/"), owner, bytes.fromhex(key)
        self.context = ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()
        self.timeout = timeout

    def call(self, operation, body, timeout=None):
        message = canonical(
            dict(owner=self.owner, operation=operation, body=body, nonce=uuid.uuid4().hex)
        )
        req = Request(
            self.url,
            data=message,
            headers={
                "Content-Type": "application/json",
                "X-OCP-MAC": hmac.digest(self.key, message, "sha256").hex(),
            },
        )
        with urlopen(
            req,
            context=self.context,
            timeout=self.timeout if timeout is None else min(self.timeout, timeout),
        ) as reply:
            result = json.load(reply)
        if "error" in result:
            raise ByteBudgetExceeded(result["error"])
        return result


class RemoteBudget:
    def __init__(self, client):
        self.client = client
        self.local = threading.local()
        self.settlement_grace_s = 2 * client.timeout + 1

    def reserve(self, size):
        ident = uuid.uuid4().hex
        # A retry uses this same id. A lost acknowledgement never permits sending.
        deadline = getattr(self.local, "deadline", time.time() + self.client.timeout)
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Budget admission deadline expired")
        try:
            reply = self.client.call("reserve", dict(id=ident, size=size), timeout=remaining)
            if not reply["admitted"]:
                raise ByteBudgetExceeded("Reservation already settled or canceled")
        except (OSError, ValueError):
            # No application byte has been sent. Fencing is safe in either request order.
            try:
                self.client.call("cancel", dict(id=ident, size=size))
            except (OSError, ValueError):
                pass  # Unknown reservations remain conservatively charged.
            raise
        self.local.frame = [ident, size, 0]

    def sent(self, size):
        self.local.frame[2] += size

    def release(self, unsent):
        ident, size, sent = self.local.frame
        if sent + unsent != size:
            raise ValueError("Invalid local settlement")
        for attempt in range(3):
            try:
                self.client.call("settle", dict(id=ident, sent=sent))
                break
            except OSError:
                if attempt == 2:
                    raise
        del self.local.frame

    def state(self):
        return self.client.call("state", {})


class _ControlHTTPServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


class ControlServer:
    def __init__(self, address, ledger, keys, evidence, cert=None, key=None):
        self.ledger, self.keys, self.evidence = ledger, keys, evidence
        if address[0] != "127.0.0.1" and not (cert and key):
            raise ValueError("Non-loopback control service needs TLS")
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 32768:
                        raise ValueError("Request size")
                    raw = self.rfile.read(size)
                    msg = json.loads(raw)
                    secret = outer.keys[msg["owner"]]
                    if not hmac.compare_digest(
                        hmac.digest(secret, raw, "sha256").hex(), self.headers.get("X-OCP-MAC", "")
                    ):
                        raise ValueError("Authentication failed")
                    op = msg["operation"]
                    result = (
                        outer.evidence(msg["body"])
                        if op == "evidence"
                        else outer.ledger.apply(msg["owner"], op, msg["body"])
                    )
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                payload = canonical(result)
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except OSError:
                    pass  # The committed reservation survives a disconnected client.

        self.http = _ControlHTTPServer(address, Handler)
        self.http.daemon_threads = True
        if cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            self.http.socket = context.wrap_socket(self.http.socket, server_side=True)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=5)
