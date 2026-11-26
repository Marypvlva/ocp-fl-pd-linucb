"""Actual local-EVM OCP-FL contract and independently auditable certificates.

EthereumTester is a research chain, not a Byzantine-consensus deployment.
The oracle is allowlisted on-chain; BLS verification is enforced off-chain.
"""

from dataclasses import asdict
from hashlib import sha256
import json
import secrets
from time import time
from eth_tester import EthereumTester, PyEVMBackend

from web3 import Web3, EthereumTesterProvider
from web3.logs import DISCARD
from eth_hash.auto import keccak

from .paths import ROOT
from .ocp_randomness import LocalTestBeacon, commitment, select_clients


def json_value(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def verify_certificate(ticket, spec):
    """Replay an exported transcript; caller must independently pin chain/contract."""
    required = {
        "round",
        "chain_id",
        "contract",
        "config_hash",
        "clients",
        "sample_size",
        "commit_end",
        "reveal_end",
        "beacon_round",
        "beacon_time",
        "commits",
        "reveals",
        "non_reveals",
        "beacon_proof",
        "mixed",
        "selected",
        "block_number",
        "block_hash",
    }
    if set(ticket) != required or ticket["config_hash"] != spec.config_hash.hex():
        raise ValueError("Certificate/config mismatch")
    t = ticket["round"]
    if (
        type(t) is not int
        or t < 1
        or not ticket["commit_end"] < ticket["reveal_end"] < ticket["beacon_time"]
    ):
        raise ValueError("Invalid ceremony windows")
    if ticket["beacon_time"] != spec.time_for(ticket["beacon_round"]):
        raise ValueError("Oracle changed the precommitted beacon round/time")
    clients = ticket["clients"]
    if len(set(clients)) != len(clients):
        raise ValueError("Duplicate registered client")
    commits = {}
    for event in ticket["commits"]:
        who = event["client"]
        if who not in clients or who in commits or event["timestamp"] >= ticket["commit_end"]:
            raise ValueError("Duplicate, unknown or late commitment")
        commits[who] = event["commitment"]
    reveals = {}
    mixed = int.from_bytes(
        spec.verify(ticket["beacon_proof"], ticket["beacon_round"], ticket["reveal_end"]), "big"
    )
    for event in ticket["reveals"]:
        who = event["client"]
        if (
            who not in commits
            or who in reveals
            or not ticket["commit_end"] <= event["timestamp"] < ticket["reveal_end"]
        ):
            raise ValueError("Invalid or late reveal")
        seed = bytes.fromhex(event["seed"])
        if commitment(seed, who, t, ticket["chain_id"], ticket["contract"]).hex() != commits[who]:
            raise ValueError("Commitment opening mismatch")
        reveals[who] = seed
        mixed ^= int.from_bytes(seed, "big")
    missing = sorted(set(commits) - set(reveals))
    if ticket["non_reveals"] != missing:
        raise ValueError("Non-reveal evidence differs")
    mixed = mixed.to_bytes(32, "big")
    selected = select_clients(mixed, len(clients), ticket["sample_size"])
    if ticket["mixed"] != mixed.hex() or ticket["selected"] != list(selected):
        raise ValueError("Coordinator changed the mixed seed or selected roster")
    return selected


class OcpChain:
    def __init__(self, web3, contract, oracle, spec, finality_blocks=2):
        if type(finality_blocks) is not int or finality_blocks < 1:
            raise ValueError("Invalid finality threshold")
        self.w3, self.contract, self.oracle, self.spec = web3, contract, oracle, spec
        self.clients = contract.functions.population().call()
        self.size = contract.functions.sampleSize().call()
        self.chain_id = web3.eth.chain_id
        self.code_hash = keccak(bytes(web3.eth.get_code(contract.address)))
        if bytes(contract.functions.beaconConfigHash().call()) != spec.config_hash:
            raise ValueError("Wrong pinned beacon configuration")
        self.finality = finality_blocks
        self.receipts = []
        self.validated = set()
        self.open_blocks = {}
        self.final_blocks = {}
        self.model_blocks = {}
        self.model_events = []
        self.late_events = []

    @classmethod
    def local(cls, population=8, size=4, spec=None, finality_blocks=2, *, genesis_timestamp=None):
        if type(population) is not int or not 2 <= population <= 16:
            raise ValueError("Local population 2..16")
        # PyEVM otherwise mixes wall-clock time into new headers. A future
        # genesis keeps the local research clock entirely virtual during a run.
        paired_test = genesis_timestamp is not None
        now = int(time())
        if genesis_timestamp is None:
            genesis_timestamp = now + 86400
        if genesis_timestamp <= now:
            raise ValueError("Local virtual genesis must be in the future")
        backend = PyEVMBackend(
            genesis_parameters={"timestamp": genesis_timestamp, "gas_limit": 30000000}
        )
        w3 = Web3(EthereumTesterProvider(EthereumTester(backend=backend)))
        tester = w3.provider.ethereum_tester
        while len(w3.eth.accounts) < population + 1:
            # Ephemeral local-test accounts, never used on an external chain.
            key = (
                sha256(f"adaptive-planning-local-test-{len(w3.eth.accounts)}".encode()).hexdigest()
                if paired_test
                else secrets.token_hex(32)
            )
            address = tester.add_account(key)
            w3.eth.wait_for_transaction_receipt(
                w3.eth.send_transaction(
                    {"from": w3.eth.accounts[0], "to": address, "value": 10**22}
                )
            )
        spec = spec or LocalTestBeacon().spec
        artifact = json.loads((ROOT / "contracts/artifacts/OcpfScheduler.json").read_text())
        if (
            artifact["source_sha256"]
            != sha256((ROOT / "contracts/OcpfScheduler.sol").read_bytes()).hexdigest()
        ):
            raise ValueError("Contract artifact is stale; rebuild with npm run build")
        factory = w3.eth.contract(
            abi=artifact["abi"], bytecode=artifact["evm"]["bytecode"]["object"]
        )
        oracle = w3.eth.accounts[0]
        receipt = w3.eth.wait_for_transaction_receipt(
            factory.constructor(
                w3.eth.accounts[1 : population + 1], size, oracle, spec.config_hash
            ).transact({"from": oracle, "gas": 12000000})
        )
        if receipt.status != 1:
            raise ValueError("Contract deployment reverted")
        obj = cls(
            w3,
            w3.eth.contract(address=receipt.contractAddress, abi=artifact["abi"]),
            oracle,
            spec,
            finality_blocks,
        )
        obj.receipts.append(obj.receipt_record(receipt, "deploy"))
        return obj

    def receipt_record(self, receipt, operation):
        return dict(
            operation=operation,
            tx_hash=receipt.transactionHash.hex(),
            block_number=receipt.blockNumber,
            block_hash=receipt.blockHash.hex(),
            timestamp=self.w3.eth.get_block(receipt.blockNumber).timestamp,
            gas_used=receipt.gasUsed,
            status=receipt.status,
        )

    def transact(self, function, sender=None, operation="transaction"):
        receipt = self.w3.eth.wait_for_transaction_receipt(
            function.transact({"from": sender or self.oracle, "gas": 12000000})
        )
        self.receipts.append(self.receipt_record(receipt, operation))
        if receipt.status != 1:
            raise ValueError("EVM transaction reverted: " + operation)
        return receipt

    @property
    def timestamp(self):
        return self.w3.eth.get_block("latest").timestamp

    def advance_local(self, timestamp):
        if not isinstance(self.w3.provider, EthereumTesterProvider):
            raise ValueError("Virtual-time advance is for local EthereumTester only")
        if timestamp > self.timestamp:
            self.w3.provider.ethereum_tester.time_travel(timestamp)
            self.w3.provider.ethereum_tester.mine_blocks(1)

    def open_round(self, t, commit_end, reveal_end, beacon_round):
        receipt = self.transact(
            self.contract.functions.openRound(
                t, commit_end, reveal_end, beacon_round, self.spec.time_for(beacon_round)
            ),
            operation="open",
        )
        self.open_blocks[t] = receipt.blockNumber
        self.final_blocks.pop(t, None)
        self.model_blocks.pop(t, None)
        return receipt

    def commit(self, t, identity, seed):
        address = self.clients[identity]
        digest = commitment(seed, address, t, self.chain_id, self.contract.address)
        return self.transact(self.contract.functions.commit(t, digest), address, "commit")

    def reveal(self, t, identity, seed):
        receipt = self.transact(
            self.contract.functions.reveal(t, seed), self.clients[identity], "reveal"
        )
        self.late_events.extend(self._receipt_events(receipt, "LateReveal"))
        return receipt

    def _receipt_events(self, receipt, name):
        timestamp = self.w3.eth.get_block(receipt.blockNumber).timestamp
        return [
            {
                **json_value(dict(event["args"])),
                "timestamp": timestamp,
                "block_number": receipt.blockNumber,
                "block_hash": receipt.blockHash.hex(),
            }
            for event in getattr(self.contract.events, name)().process_receipt(
                receipt, errors=DISCARD
            )
        ]

    def finalize(self, t, proof):
        meta = self.contract.functions.roundMeta(t).call()
        value = self.spec.verify(proof, meta[2], meta[1])
        receipt = self.transact(
            self.contract.functions.finalize(
                t,
                value,
                bytes.fromhex(proof["signature"]),
                bytes.fromhex(proof["previous_signature"]),
            ),
            operation="finalize",
        )
        self.final_blocks[t] = receipt.blockNumber
        if isinstance(self.w3.provider, EthereumTesterProvider):
            self.w3.provider.ethereum_tester.mine_blocks(self.finality - 1)
        return receipt

    def _events(self, name, t):
        start = self.open_blocks.get(t, 0)
        if name == "ModelRecorded":
            end = self.model_blocks.get(t, "latest")
        elif name == "LateReveal":
            end = self.open_blocks[t + 1] - 1 if t + 1 in self.open_blocks else "latest"
        else:
            end = self.final_blocks.get(t, "latest")
        events = getattr(self.contract.events, name)().get_logs(
            from_block=start, to_block=end, argument_filters={"round": t}
        )
        return [
            {
                **json_value(dict(e["args"])),
                "timestamp": self.w3.eth.get_block(e["blockNumber"]).timestamp,
                "block_number": e["blockNumber"],
                "block_hash": e["blockHash"].hex(),
            }
            for e in events
        ]

    def certificate(self, t):
        meta = self.contract.functions.roundMeta(t).call()
        finals = self._events("RoundFinal", t)
        opens = self._events("RoundOpened", t)
        if not meta[4] or len(finals) != 1 or len(opens) != 1:
            raise ValueError("Missing unique on-chain round finalization")
        final = finals[0]
        if self.w3.eth.block_number - final["block_number"] + 1 < self.finality:
            raise ValueError("Insufficient chain finality")
        commits = [
            {k: e[k] for k in ("client", "commitment", "timestamp")}
            for e in self._events("Commit", t)
        ]
        reveals = [
            {k: e[k] for k in ("client", "seed", "timestamp")} for e in self._events("Reveal", t)
        ]
        ticket = dict(
            round=t,
            chain_id=self.chain_id,
            contract=self.contract.address,
            config_hash=self.spec.config_hash.hex(),
            clients=list(self.clients),
            sample_size=self.size,
            commit_end=meta[0],
            reveal_end=meta[1],
            beacon_round=meta[2],
            beacon_time=meta[3],
            commits=commits,
            reveals=reveals,
            non_reveals=sorted(e["client"] for e in self._events("NonReveal", t)),
            beacon_proof=dict(
                round=meta[2],
                signature=bytes(self.contract.functions.beaconSignature(t).call()).hex(),
                previous_signature=bytes(self.contract.functions.previousSignature(t).call()).hex(),
                randomness=bytes(meta[5]).hex(),
            ),
            mixed=bytes(meta[6]).hex(),
            selected=list(self.contract.functions.selected(t).call()),
            block_number=final["block_number"],
            block_hash=final["block_hash"],
        )
        if (
            opens[0]["commitEnd"],
            opens[0]["revealEnd"],
            opens[0]["beaconRound"],
            opens[0]["beaconTime"],
        ) != tuple(meta[:4]):
            raise ValueError("Round metadata differs from opening event")
        return ticket

    def authorize(self, ticket, round_id, workers):
        if (
            not isinstance(ticket, dict)
            or ticket.get("round") != round_id
            or workers != len(self.clients)
        ):
            raise ValueError("Missing, reused or foreign schedule")
        if (
            ticket.get("chain_id") != self.chain_id
            or ticket.get("contract") != self.contract.address
            or ticket.get("clients") != self.clients
            or ticket.get("sample_size") != self.size
        ):
            raise ValueError("Untrusted chain/contract/registry")
        if (
            self.w3.eth.get_block(ticket["block_number"]).hash.hex() != ticket["block_hash"]
            or keccak(bytes(self.w3.eth.get_code(self.contract.address))) != self.code_hash
        ):
            raise ValueError("Reorg or contract code mismatch")
        if self.w3.eth.block_number - ticket["block_number"] + 1 < self.finality:
            raise ValueError("Insufficient chain finality")
        canonical = self.certificate(round_id)
        if canonical != ticket:
            raise ValueError("Schedule differs from canonical chain events")
        digest = sha256(json.dumps(ticket, sort_keys=True).encode()).digest()
        if digest not in self.validated:
            verify_certificate(ticket, self.spec)
            self.validated.add(digest)
        return tuple(ticket["selected"])

    def run_local_ceremony(self, t, beacon=None, missing=(), seeds=None):
        beacon = beacon or LocalTestBeacon()
        if self.spec.mode != "local_test_bls" or beacon.spec != self.spec:
            raise ValueError("Automated virtual-time ceremony is local-test only")
        n = len(self.clients)
        seeds = [secrets.token_bytes(32) for _ in self.clients] if seeds is None else seeds
        if (
            len(seeds) != n
            or len(set(missing)) != len(missing)
            or any(type(i) is not int or not 0 <= i < n for i in missing)
        ):
            raise ValueError("Invalid seed population/non-reveal set")
        commit_end = self.timestamp + 2 * n + 10
        reveal_end = commit_end + 2 * n + 10
        beacon_round = self.spec.round_after(reveal_end)
        self.open_round(t, commit_end, reveal_end, beacon_round)
        for i, seed in enumerate(seeds):
            self.commit(t, i, seed)
        self.advance_local(commit_end)
        for i, seed in enumerate(seeds):
            if i not in missing:
                self.reveal(t, i, seed)
        self.advance_local(self.spec.time_for(beacon_round))
        self.finalize(t, beacon.fetch(beacon_round))
        ticket = self.certificate(t)
        self.authorize(ticket, t, n)
        return ticket

    def record_model(self, t, model_bytes):
        digest = keccak(model_bytes)
        receipt = self.transact(
            self.contract.functions.recordModel(t, digest), operation="model_hash"
        )
        self.model_blocks[t] = receipt.blockNumber
        self.model_events.extend(self._receipt_events(receipt, "ModelRecorded"))
        return digest.hex()

    def export(self):
        # Historical eth_getLogs in PyEVM repeatedly scans for old transactions.
        # Decode events from their original receipts, and separately ensure those
        # receipts still belong to the canonical chain before exporting them.
        for receipt in self.receipts:
            if self.w3.eth.get_block(receipt["block_number"]).hash.hex() != receipt["block_hash"]:
                raise ValueError("Receipt became noncanonical before export")
        return dict(
            backend="EthereumTester/PyEVM (single-process research EVM, no consensus network)",
            chain_id=self.chain_id,
            contract=self.contract.address,
            code_hash=self.code_hash.hex(),
            beacon_spec=asdict(self.spec),
            finality_blocks=self.finality,
            model_events=list(self.model_events),
            receipts=self.receipts,
            gas_used=sum(r["gas_used"] for r in self.receipts),
            late_reveals=list(self.late_events),
        )
