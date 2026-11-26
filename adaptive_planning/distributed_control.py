"""Authenticated control-plane requests and durable, fail-closed frame reservations.

Each frame is charged before sending. Missing settlements retain the complete
reservation across service restarts. Control traffic is outside the application
byte budget and is measured separately by the experiment wall timer.
"""

import hmac
import http.client
import queue
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
import socket
import ssl
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .byte_budget import ByteBudgetExceeded
from .control_telemetry import ControlTelemetry, process_load


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Ledger:
    """One durable transaction per change, constant-time accounting per request.

    Totals are initialized once when opening a legacy database. The frame journal
    and totals always change within the same SQLite transaction. The Python lock
    reuses one connection; SQLite still serializes independent Ledger instances.
    """

    def __init__(self, path, limit):
        if type(limit) is not int or not 0 < limit < 2**63:
            raise ValueError("Invalid budget")
        self.path, self.limit = str(path), limit
        self.lock = threading.RLock()
        self.timings = threading.local()
        self._db = sqlite3.connect(
            self.path, timeout=5, check_same_thread=False, isolation_level=None
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS config (limit_bytes INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS denials (owner TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS frames
            (id TEXT PRIMARY KEY, owner TEXT NOT NULL, size INTEGER NOT NULL,
             sent INTEGER, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS totals
            (id INTEGER PRIMARY KEY CHECK(id=1), spent INTEGER NOT NULL,
             reserved INTEGER NOT NULL, frames INTEGER NOT NULL, denied INTEGER NOT NULL);
        """)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT limit_bytes FROM config").fetchone()
            if row and row[0] != limit:
                raise ValueError("Existing ledger has a different limit")
            if not row:
                db.execute("INSERT INTO config VALUES (?)", (limit,))
            if not db.execute("SELECT id FROM totals WHERE id=1").fetchone():
                spent, reserved, frames = db.execute(
                    "SELECT COALESCE(SUM(sent),0), COALESCE(SUM(CASE WHEN sent IS NULL THEN size ELSE 0 END),0), COUNT(*) FROM frames"
                ).fetchone()
                denied = db.execute("SELECT COUNT(*) FROM denials").fetchone()[0]
                db.execute(
                    "INSERT INTO totals VALUES (1,?,?,?,?)", (spent, reserved, frames, denied)
                )

    @contextmanager
    def connect(self):
        started = time.perf_counter()
        with self.lock:
            acquired = time.perf_counter()
            try:
                with self._db:
                    yield self._db
            finally:
                self.timings.wait_s = acquired - started
                self.timings.sql_s = time.perf_counter() - acquired

    def close(self):
        with self.lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    def __del__(self):
        if getattr(self, "_db", None) is not None:
            self.close()

    def apply(self, owner, operation, body):
        with self.connect() as db:
            if operation != "state":
                db.execute("BEGIN IMMEDIATE")
            spent, reserved, frames, denied = db.execute(
                "SELECT spent,reserved,frames,denied FROM totals WHERE id=1"
            ).fetchone()
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
                        if old[2] is None:
                            reserved -= size
                            db.execute("UPDATE frames SET sent=0 WHERE id=?", (ident,))
                    else:
                        admitted = old[2] is None
                elif operation == "cancel":
                    db.execute(
                        "INSERT INTO frames VALUES (?,?,?,?,?)",
                        (ident, owner, size, 0, time.time()),
                    )
                    frames += 1
                else:
                    if size > self.limit - spent - reserved:
                        db.execute("INSERT INTO denials VALUES (?,?)", (owner, time.time()))
                        db.execute("UPDATE totals SET denied=denied+1 WHERE id=1")
                        db.commit()
                        raise ByteBudgetExceeded("Application byte quota exhausted")
                    db.execute(
                        "INSERT INTO frames VALUES (?,?,?,NULL,?)",
                        (ident, owner, size, time.time()),
                    )
                    reserved += size
                    frames += 1
                    admitted = True
            elif operation == "settle":
                old = db.execute(
                    "SELECT owner,size,sent FROM frames WHERE id=?", (body["id"],)
                ).fetchone()
                sent = body["sent"]
                if not old or old[0] != owner or type(sent) is not int or not 0 <= sent <= old[1]:
                    raise ValueError("Invalid frame settlement")
                if old[2] is not None and old[2] != sent:
                    raise ValueError("Conflicting settlement")
                if old[2] is None:
                    db.execute("UPDATE frames SET sent=? WHERE id=?", (sent, body["id"]))
                    reserved -= old[1]
                    spent += sent
            elif operation != "state":
                raise ValueError("Unknown ledger operation")
            if not 0 <= spent + reserved <= self.limit or min(spent, reserved, frames, denied) < 0:
                raise ValueError("Ledger invariant violated")
            if operation != "state":
                db.execute(
                    "UPDATE totals SET spent=?,reserved=?,frames=?,denied=? WHERE id=1",
                    (spent, reserved, frames, denied),
                )
            return dict(
                admitted=admitted,
                limit=self.limit,
                spent=spent,
                reserved=reserved,
                frames=frames,
                denied_frames=denied,
            )

    def audit(self):
        with self.connect() as db:
            db.execute("BEGIN")
            actual = db.execute(
                "SELECT COALESCE(SUM(sent),0), COALESCE(SUM(CASE WHEN sent IS NULL THEN size ELSE 0 END),0), COUNT(*) FROM frames"
            ).fetchone()
            totals = db.execute("SELECT spent,reserved,frames FROM totals WHERE id=1").fetchone()
            if actual != totals:
                raise ValueError("Journal and totals disagree")
            return dict(spent=actual[0], reserved=actual[1], frames=actual[2])


class ControlServiceError(ValueError):
    """A non-quota rejection, distinct from a lost response or denied budget."""


class ControlClient:
    def __init__(self, url, owner, key, ca=None, timeout=5, pool_size=2, attempts=3, log_path=None):
        parsed = urlsplit(url)
        if not (
            parsed.scheme == "https" or parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
        ):
            raise ValueError("Control plane requires HTTPS or loopback")
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.hostname
        ):
            raise ValueError("Invalid control URL")
        if not 0 < timeout <= 60 or not 1 <= pool_size <= 4 or not 1 <= attempts <= 5:
            raise ValueError("Invalid control client limits")
        self.url, self.owner, self.key = url.rstrip("/"), owner, bytes.fromhex(key)
        self.parsed = parsed
        self.context = ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()
        self.timeout, self.attempts = timeout, attempts
        self.telemetry = ControlTelemetry(log_path)
        self.closed = False
        self.pool = queue.LifoQueue(pool_size)
        for _ in range(pool_size):
            self.pool.put(None)

    def _connection(self, timeout):
        if self.parsed.scheme == "https":
            return http.client.HTTPSConnection(
                self.parsed.hostname, self.parsed.port, timeout=timeout, context=self.context
            )
        return http.client.HTTPConnection(self.parsed.hostname, self.parsed.port, timeout=timeout)

    def call(self, operation, body, timeout=None):
        if self.closed:
            raise ValueError("Control client is closed")
        started = time.perf_counter()
        allowance = self.timeout if timeout is None else min(self.timeout, timeout)
        deadline = started + allowance
        request_id = uuid.uuid4().hex
        fields = dict(
            owner=self.owner, operation=operation, request_id=request_id, frame_id=body.get("id")
        )
        if allowance <= 0:
            raise TimeoutError("Control request deadline expired")
        pool_started = time.perf_counter()
        try:
            connection = self.pool.get(timeout=allowance)
        except queue.Empty as exc:
            self.telemetry.event(
                "client_pool_timeout",
                **fields,
                elapsed_s=time.perf_counter() - started,
                error_type="TimeoutError",
            )
            raise TimeoutError("Control connection pool deadline") from exc
        pool_wait = time.perf_counter() - pool_started
        if connection and time.perf_counter() - getattr(connection, "_ocp_last_used", 0) > 1.5:
            connection.close()
            connection = None
        try:
            for attempt in range(1, self.attempts + 1):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("Control request deadline expired")
                if connection is None:
                    connection = self._connection(remaining)
                connection.timeout = remaining
                if connection.sock:
                    connection.sock.settimeout(remaining)
                message = canonical(
                    dict(owner=self.owner, operation=operation, body=body, nonce=uuid.uuid4().hex)
                )
                try:
                    connection.request(
                        "POST",
                        self.parsed.path or "/",
                        body=message,
                        headers={
                            "Content-Type": "application/json",
                            "X-OCP-Request": request_id,
                            "X-OCP-Attempt": str(attempt),
                            "X-OCP-MAC": hmac.digest(self.key, message, "sha256").hex(),
                        },
                    )
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise TimeoutError("Control request deadline expired")
                    if connection.sock:
                        connection.sock.settimeout(remaining)
                    response = connection.getresponse()
                    declared = response.getheader("Content-Length")
                    if (
                        declared is None
                        or not declared.isdecimal()
                        or not 0 < int(declared) <= 262144
                    ):
                        raise ControlServiceError("Invalid control response length")
                    raw = response.read(int(declared))
                    if len(raw) != int(declared):
                        raise http.client.IncompleteRead(raw, int(declared) - len(raw))
                    if len(raw) > 262144:
                        raise ControlServiceError("Oversized control response")
                    result = json.loads(raw)
                    if response.status in (429, 503):
                        raise OSError("Control service temporarily unavailable")
                    if "error" in result:
                        code = result.get("error_type")
                        if code == "ByteBudgetExceeded":
                            raise ByteBudgetExceeded(result["error"])
                        raise ControlServiceError(result["error"])
                    if response.status != 200:
                        raise ControlServiceError("Unexpected control HTTP status")
                    self.telemetry.event(
                        "client_success",
                        **fields,
                        attempt=attempt,
                        pool_wait_s=pool_wait,
                        elapsed_s=time.perf_counter() - started,
                    )
                    return result
                except (OSError, http.client.HTTPException) as exc:
                    self.telemetry.event(
                        "client_transport_error",
                        **fields,
                        attempt=attempt,
                        pool_wait_s=pool_wait,
                        elapsed_s=time.perf_counter() - started,
                        error_type=type(exc).__name__,
                    )
                    connection.close()
                    connection = None
                    if attempt == self.attempts or deadline - time.perf_counter() <= 0.002:
                        if isinstance(exc, OSError):
                            raise
                        raise OSError("Control transport failed: " + type(exc).__name__) from exc
                    backoff = min(
                        0.005 * (2 ** (attempt - 1)), max(0.0, (deadline - time.perf_counter()) / 2)
                    )
                    self.telemetry.event(
                        "client_retry", **fields, attempt=attempt + 1, backoff_s=backoff
                    )
                    if backoff:
                        time.sleep(backoff)
                    # Same frame id and one total request deadline. No application retry.
                except (ValueError, KeyError) as exc:
                    self.telemetry.event(
                        "client_rejected",
                        **fields,
                        attempt=attempt,
                        elapsed_s=time.perf_counter() - started,
                        error_type=type(exc).__name__,
                    )
                    connection.close()
                    connection = None
                    raise
        except BaseException as exc:
            if isinstance(exc, TimeoutError):
                self.telemetry.event(
                    "client_deadline",
                    **fields,
                    elapsed_s=time.perf_counter() - started,
                    error_type=type(exc).__name__,
                )
            if connection:
                connection.close()
                connection = None
            raise
        finally:
            if connection:
                connection._ocp_last_used = time.perf_counter()
            self.pool.put(connection)

    def close(self):
        self.closed = True
        while True:
            try:
                connection = self.pool.get_nowait()
            except queue.Empty:
                break
            if connection:
                connection.close()
        self.telemetry.close()

    def __del__(self):
        if hasattr(self, "pool"):
            self.close()


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
            reply = self.client.call(
                "reserve",
                dict(
                    id=ident,
                    size=size,
                    admit_until_ns=int(deadline * 1e9),
                    round=getattr(self.local, "round_id", None),
                ),
                timeout=remaining,
            )
            if not reply["admitted"]:
                raise ByteBudgetExceeded("Reservation already settled or canceled")
        except (OSError, ValueError):
            # No application byte has been sent. Fencing is safe in either request order.
            try:
                self.client.call(
                    "cancel", dict(id=ident, size=size, round=getattr(self.local, "round_id", None))
                )
            except (OSError, ValueError) as cancel_error:
                telemetry = getattr(self.client, "telemetry", None)
                if telemetry:
                    telemetry.event(
                        "cancellation_unknown",
                        frame_id=ident,
                        error_type=type(cancel_error).__name__,
                    )
                # Unknown reservations remain conservatively charged.
            raise
        self.local.frame = [ident, size, 0, getattr(self.local, "round_id", None)]

    def sent(self, size):
        self.local.frame[2] += size

    def release(self, unsent):
        ident, size, sent, round_id = self.local.frame
        if sent + unsent != size:
            raise ValueError("Invalid local settlement")
        self.client.call("settle", dict(id=ident, sent=sent, round=round_id))
        del self.local.frame

    def state(self):
        return self.client.call("state", {})


class _ControlHTTPServer(ThreadingHTTPServer):
    request_queue_size = 128

    def __init__(self, address, handler, telemetry, context=None, max_workers=64):
        self.telemetry, self.tls_context = telemetry, context
        self.workers = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="control-http"
        )
        self.slots = threading.BoundedSemaphore(max_workers * 2)
        self.load_lock = threading.Lock()
        self.queued, self.active, self.peak = 0, 0, 0
        super().__init__(address, handler)

    def process_request(self, request, address):
        # Accept raw TCP here; handshake happens in a bounded worker, never the accept loop.
        if not self.slots.acquire(blocking=False):
            self.telemetry.event(
                "server_overloaded", peer=address[0], error_type="CapacityExceeded"
            )
            self.shutdown_request(request)
            return
        accepted = time.perf_counter()
        with self.load_lock:
            self.queued += 1
        self.workers.submit(self._serve_connection, request, address, accepted)

    def _serve_connection(self, request, address, accepted):
        stage = "handshake"
        queue_s = time.perf_counter() - accepted
        tls_started = time.perf_counter()
        with self.load_lock:
            self.queued -= 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            request.settimeout(5)
            request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self.tls_context:
                request = self.tls_context.wrap_socket(request, server_side=True)
            self.telemetry.event(
                "connection_open",
                peer=address[0],
                queue_s=queue_s,
                tls_s=time.perf_counter() - tls_started,
            )
            stage = "request"
            self.finish_request(request, address)
        except (OSError, ValueError) as exc:
            self.telemetry.event(
                "connection_error", peer=address[0], stage=stage, error_type=type(exc).__name__
            )
        except Exception as exc:
            self.telemetry.event(
                "handler_error", peer=address[0], stage=stage, error_type=type(exc).__name__
            )
            self.handle_error(request, address)
        finally:
            self.shutdown_request(request)
            with self.load_lock:
                self.active -= 1
            self.slots.release()

    def load(self):
        with self.load_lock:
            return dict(
                active_connections=self.active,
                queued_connections=self.queued,
                peak_connections=self.peak,
            )

    def server_close(self):
        super().server_close()
        self.workers.shutdown(wait=True)


class ControlServer:
    def __init__(
        self, address, ledger, keys, evidence, cert=None, key=None, log_path=None, max_workers=64
    ):
        self.ledger, self.keys, self.evidence = ledger, keys, evidence
        if address[0] != "127.0.0.1" and not (cert and key):
            raise ValueError("Non-loopback control service needs TLS")
        self.telemetry = ControlTelemetry(log_path)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass  # Structured authenticated-request events are recorded below.

            def log_error(self, message, *args):
                outer.telemetry.event(
                    "http_read_error", peer=self.client_address[0], detail=(message % args)[:200]
                )

            def handle(self):
                self.connection.settimeout(2)  # Idle keep-alive reads are bounded.
                try:
                    super().handle()
                except OSError as exc:
                    outer.telemetry.event(
                        "request_read_error",
                        peer=self.client_address[0],
                        error_type=type(exc).__name__,
                    )

            def do_POST(self):
                started = time.perf_counter()
                fields = dict(
                    peer=self.client_address[0],
                    request_id=self.headers.get("X-OCP-Request", "")[:64],
                    attempt=self.headers.get("X-OCP-Attempt", "")[:8],
                )
                stage = "read_body"
                applied = False
                try:
                    self.connection.settimeout(5)
                    size = int(self.headers.get("Content-Length", "0"))
                    if self.headers.get("Transfer-Encoding") or not 0 < size <= 32768:
                        raise ValueError("Request size")
                    raw = self.rfile.read(size)
                    if len(raw) != size:
                        raise ValueError("Incomplete request")
                    msg = json.loads(raw)
                    secret = outer.keys[msg["owner"]]
                    if not hmac.compare_digest(
                        hmac.digest(secret, raw, "sha256").hex(), self.headers.get("X-OCP-MAC", "")
                    ):
                        raise ValueError("Authentication failed")
                    fields.update(
                        owner=msg["owner"],
                        operation=msg["operation"],
                        frame_id=msg["body"].get("id"),
                        round=msg["body"].get("round"),
                        admit_until_ns=msg["body"].get("admit_until_ns"),
                    )
                    stage = "operation"
                    op_started = time.perf_counter()
                    result = (
                        outer.evidence(msg["body"])
                        if msg["operation"] == "evidence"
                        else outer.ledger.apply(msg["owner"], msg["operation"], msg["body"])
                    )
                    fields["operation_s"] = time.perf_counter() - op_started
                    if msg["operation"] != "evidence":
                        fields["ledger_wait_s"] = getattr(outer.ledger.timings, "wait_s", 0.0)
                        fields["ledger_transaction_s"] = getattr(outer.ledger.timings, "sql_s", 0.0)
                    applied = True
                    status = 200
                except Exception as exc:
                    status = 503 if isinstance(exc, sqlite3.OperationalError) else 400
                    result = dict(error=str(exc)[:250], error_type=type(exc).__name__)
                    self.close_connection = True
                    outer.telemetry.event(
                        "operation_error",
                        **fields,
                        stage=stage,
                        error_type=type(exc).__name__,
                        detail=str(exc)[:250],
                        elapsed_s=time.perf_counter() - started,
                    )
                payload = canonical(result)
                stage = "headers"
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    if self.close_connection:
                        self.send_header("Connection", "close")
                    self.end_headers()
                    stage = "body"
                    self.wfile.write(payload)
                    self.wfile.flush()
                    outer.telemetry.event(
                        "response_written",
                        **fields,
                        status=status,
                        operation_applied=applied,
                        elapsed_s=time.perf_counter() - started,
                        **outer.http.load(),
                        **process_load(),
                    )
                except OSError as exc:
                    self.close_connection = True
                    outer.telemetry.event(
                        "response_lost",
                        **fields,
                        stage=stage,
                        status=status,
                        operation_applied=applied,
                        error_type=type(exc).__name__,
                        elapsed_s=time.perf_counter() - started,
                        **outer.http.load(),
                        **process_load(),
                    )
                finally:
                    self.connection.settimeout(2)

        context = None
        if cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
        self.http = _ControlHTTPServer(address, Handler, self.telemetry, context, max_workers)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=5)
        self.telemetry.event("server_closed", **self.http.load(), **process_load())
        self.telemetry.close()
