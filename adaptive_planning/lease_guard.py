"""Each role validates its CollectorLeases grant before application sends."""

import json
import time
from pathlib import Path
from eth_abi import encode
from eth_hash.auto import keccak


def read_evidence(w3, pin, request):
    root = Path(__file__).resolve().parents[1]
    source_abi = json.loads((root / "contracts/artifacts/OcpfScheduler.json").read_text())["abi"]
    lease_abi = json.loads(
        (root / "experiments/collector_v3/kit/CollectorLeases.json").read_text()
    )["abi"]
    source = w3.eth.contract(address=pin["source"], abi=source_abi)
    lease = w3.eth.contract(address=pin["lease"], abi=lease_abi)
    rd, block = request["round"], request["block"]
    # All reads share one concrete state root; later read confirms anchor is canonical.
    at = w3.eth.block_number
    meta = source.functions.roundMeta(rd).call(block_identifier=at)
    result = dict(
        chain_id=w3.eth.chain_id,
        source=pin["source"],
        lease=pin["lease"],
        source_code=keccak(bytes(w3.eth.get_code(pin["source"], at))).hex(),
        lease_code=keccak(bytes(w3.eth.get_code(pin["lease"], at))).hex(),
        cfg=lease.functions.policyConfig().call(block_identifier=at).hex(),
        lease_source=lease.functions.source().call(block_identifier=at),
        finalized=bool(meta[4]),
        mixed=bytes(meta[6]).hex(),
        selected=list(source.functions.selected(rd).call(block_identifier=at)),
        grant=lease.functions.grant(rd).call(block_identifier=at).hex(),
        block_hash=w3.eth.get_block(block).hash.hex(),
        confirmations=at - block + 1,
        read_block=at,
    )
    if pin.get("require_finalized"):
        result["finalized_block"] = w3.eth.get_block("finalized").number
    return result


class LeaseGuard:
    def __init__(self, identity, pin, reader):
        self.identity, self.pin, self.reader = identity, pin, reader
        self.current = None
        self.evidence = None

    def install(self, request):
        if self.current and (
            request["round"] < self.current["round"]
            or request["round"] == self.current["round"]
            and request != self.current
        ):
            raise ValueError("Stale or conflicting grant")
        e = self.reader(request)
        for key in ("chain_id", "source", "lease", "source_code", "lease_code", "cfg"):
            if e[key] != self.pin[key]:
                raise ValueError("Pinned chain/contract/configuration mismatch")
        mask = request["mask"]
        if type(mask) is not int or not 0 < mask < 16 or mask.bit_count() > 2:
            raise ValueError("Invalid collector mask")
        if (
            not e["finalized"]
            or e["lease_source"] != self.pin["source"]
            or e["confirmations"] < self.pin["confirmations"]
        ):
            raise ValueError("Unfinalized or foreign grant")
        if self.pin.get("require_finalized") and e.get("finalized_block", -1) < request["block"]:
            raise ValueError("Reservation has no consensus finalized block")
        if request["window"] not in self.pin.get("windows", [80, 130, 200]):
            raise ValueError("Window differs from public policy")
        if e["block_hash"] != request["block_hash"] or e["selected"] != request["selected"]:
            raise ValueError("Noncanonical grant or roster")
        digest = keccak(
            encode(
                [
                    "uint256",
                    "address",
                    "address",
                    "bytes32",
                    "uint256",
                    "bytes32",
                    "uint256[]",
                    "uint8",
                    "uint16",
                ],
                [
                    e["chain_id"],
                    e["lease"],
                    e["source"],
                    bytes.fromhex(e["cfg"]),
                    request["round"],
                    bytes.fromhex(e["mixed"]),
                    e["selected"],
                    mask,
                    request["window"],
                ],
            )
        ).hex()
        if digest != request["digest"] or digest != e["grant"]:
            raise ValueError("Grant digest mismatch")
        self.current, self.evidence = dict(request), e
        return dict(round=request["round"], digest=digest, read_block=e["read_block"])

    def authorize(self, header):
        if header["round"] == 0:
            return
        if self.current and time.time() > self.current.get("expires_at", float("inf")):
            raise ValueError("Round grant expired")
        if not self.current or header["round"] != self.current["round"]:
            raise ValueError("No verified grant for this round")
        if self.identity.startswith("w") and int(self.identity[1:]) not in self.current["selected"]:
            raise ValueError("Worker absent from the certified roster")
        if header["kind"] in ("relay_projection", "relayed_projection"):
            edge = (
                self.identity
                if self.identity.startswith("e")
                else (
                    header["recipient"] if header["sender"] == self.identity else header["sender"]
                )
            )
            if not edge.startswith("e") or not self.current["mask"] & (1 << int(edge[1:])):
                raise ValueError("Collector absent from the paid grant")
        if (
            header["kind"] == "relay_projection"
            and not header.get("response", False)
            and header["body"]["worker"] not in self.current["selected"]
        ):
            raise ValueError("Unselected relay target")


class DistributedShaper:
    """Transport adapter enforcing remote credit and verified grants without emulation."""

    def __init__(self, budget, guard=None):
        self.byte_budget, self.guard = budget, guard

    def authorize(self, header):
        self.byte_budget.local.deadline = header["deadline_ns"] / 1e9
        self.byte_budget.local.round_id = header["round"]
        if self.guard:
            self.guard.authorize(header)

    def latency(self, header, deadline):
        pass

    def limited(self, header):
        return False
