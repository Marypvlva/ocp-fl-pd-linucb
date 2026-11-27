"""Check release checksums, export completeness and secret indicators without printing secrets."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import re
import zipfile

PATTERNS = {
    "private_key_pem": rb"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----",
    "github_token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b",
    "cloud_access_key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "credential_url": rb'https?://[^\s"/@:]+:[^\s"/@]+@',
    "authorization_value": rb'(?i)(?:authorization["\s]*[:=]["\s]*)(?:bearer|basic)\s+[A-Za-z0-9+/=_-]{8,}',
}
SENSITIVE_FIELDS = re.compile(
    r"^(?:private_?key|secret(?:_?key)?|password|access_?token|refresh_?token|credentials|x-ocp-mac)$",
    re.I,
)


def scan_value(value, path=()):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            current = path + (str(key),)
            if SENSITIVE_FIELDS.fullmatch(str(key)) and item not in ("", None, {}, []):
                found.append("/".join(current))
            found.extend(scan_value(item, current))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.extend(scan_value(item, path + (str(i),)))
    return found


def check(directory):
    directory = Path(directory)
    manifest = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        if Path(name).name != name or name in manifest:
            raise ValueError("Invalid release checksum manifest")
        manifest[name] = digest
    reports = []
    for name, expected in manifest.items():
        path = directory / name
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        if digest != expected:
            raise ValueError("Release SHA256 mismatch: " + name)
        findings = []
        counts = Counter()
        metadata_ips = set()
        jsonl_events = Counter()
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate export path")
            for member in names:
                p = Path(member)
                if p.is_absolute() or ".." in p.parts or "\\" in member:
                    raise ValueError("Unsafe export path")
                if p.suffix.lower() in (".key", ".pem", ".sqlite", ".db") or p.name in (
                    "id_rsa",
                    "id_ed25519",
                    "coordinator_key",
                ):
                    findings.append(dict(file=member, kind="sensitive_file_type"))
                data = z.read(member)
                for kind, pattern in PATTERNS.items():
                    if re.search(pattern, data):
                        findings.append(dict(file=member, kind=kind))
                metadata_ips.update(re.findall(rb"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", data))
                if p.suffix == ".json":
                    value = json.loads(data)
                    for field in scan_value(value):
                        findings.append(dict(file=member, kind="sensitive_json_field", field=field))
                elif p.suffix == ".jsonl":
                    for line in data.splitlines():
                        value = json.loads(line)
                        jsonl_events[value.get("event", "other")] += 1
                        for field in scan_value(value):
                            findings.append(
                                dict(file=member, kind="sensitive_jsonl_field", field=field)
                            )
                if p.name.startswith("round-") and p.suffix == ".json":
                    counts["rounds"] += 1
                if p.name == "complete.json":
                    counts["complete_episodes"] += 1
                if p.name == "failure.json":
                    counts["failure_files"] += 1
                if p.name == "RUNNING":
                    counts["running_markers"] += 1
            reports.append(
                dict(
                    archive=name,
                    sha256=digest,
                    release_checksum_ok=True,
                    files=len(names),
                    uncompressed_bytes=sum(i.file_size for i in z.infolist()),
                    counts=dict(counts),
                    secret_indicators=findings,
                    operational_ipv4_values=len(metadata_ips),
                    jsonl_event_counts=dict(jsonl_events),
                )
            )
    return dict(
        archives=reports,
        no_detected_secret_indicators=not any(r["secret_indicators"] for r in reports),
        scope="Automated indicators plus manual schema review required. Does not prove absence of every possible secret. Public keys, signatures, session identifiers, disclosed research VM addresses and path metadata are not private credentials. No values of detected secrets are printed.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = check(a.directory)
    with a.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            dict(
                no_detected_secret_indicators=result["no_detected_secret_indicators"],
                counts=[r["counts"] for r in result["archives"]],
            ),
            indent=2,
        )
    )
    if not result["no_detected_secret_indicators"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
