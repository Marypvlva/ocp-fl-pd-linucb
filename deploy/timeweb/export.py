"""Serve one completed results ZIP over authenticated HTTPS from the coordinator."""

import argparse
import base64
import hmac
from hashlib import sha256
import json
from pathlib import Path
import secrets
import ssl
from tempfile import NamedTemporaryFile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zipfile import ZipFile, ZIP_DEFLATED


def build(source, target):
    # Only public research outputs. Never include TLS/SSH keys, ledger credentials or cloud state.
    allowed = {".json", ".jsonl", ".txt", ".sha256", ".csv", ".log"}
    hashes = []
    with NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as f:
        temporary = Path(f.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as z:
            for p in sorted(source.rglob("*")):
                if (
                    p.is_file()
                    and not p.is_symlink()
                    and p.suffix in allowed
                    and p.name != "RUNNING"
                ):
                    data = p.read_bytes()
                    relative = p.relative_to(source).as_posix()
                    z.writestr(relative, data)
                    hashes.append(f"{sha256(data).hexdigest()}  {relative}\n")
            z.writestr("SHA256SUMS", "".join(hashes))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cert", type=Path, required=True)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--port", type=int, default=8443)
    a = p.parse_args()
    if (a.out / "RUNNING").exists():
        raise ValueError("Study is still running")
    archive = a.out.with_suffix(".results.zip")
    build(a.out, archive)
    password = secrets.token_urlsafe(24)
    auth = "Basic " + base64.b64encode(("research:" + password).encode()).decode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path != "/results.zip" or not hmac.compare_digest(
                self.headers.get("Authorization", ""), auth
            ):
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="research results"')
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(archive.stat().st_size))
            self.end_headers()
            with archive.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    self.wfile.write(chunk)

    server = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(a.cert, a.key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(
        json.dumps(dict(user="research", password=password, path="/results.zip", port=a.port)),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
