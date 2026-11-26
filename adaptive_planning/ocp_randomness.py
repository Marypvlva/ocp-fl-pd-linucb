"""OCP-FL primitives: Ethereum Keccak, unbiased shuffle and verified drand G2."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from functools import lru_cache
import json
from urllib.request import urlopen

from eth_hash.auto import keccak
from py_ecc.bls import G2Basic

COMMIT_DOMAIN = b"OCP-FL/commit/v1"
SHUFFLE_DOMAIN = b"OCP-FL/shuffle/v1"


@lru_cache(maxsize=2048)
def _verify_bls(public_key, digest, signature):
    return G2Basic.Verify(public_key, digest, signature)


def uint256(value):
    if type(value) is not int or not 0 <= value < 2**256:
        raise ValueError("Invalid uint256")
    return value.to_bytes(32, "big")


def commitment(seed, client, round_id, chain_id, contract):
    if not isinstance(seed, bytes) or len(seed) != 32 or round_id < 1:
        raise ValueError("Invalid commitment")
    addresses = [bytes.fromhex(a.removeprefix("0x")) for a in (contract, client)]
    if any(len(a) != 20 for a in addresses):
        raise ValueError("Invalid Ethereum address")
    return keccak(
        COMMIT_DOMAIN + uint256(chain_id) + addresses[0] + seed + addresses[1] + uint256(round_id)
    )


def select_clients(seed, population, size, word_source=None):
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise ValueError("A 256-bit mixed seed is required")
    if type(population) is not int or type(size) is not int or not 1 <= size <= population <= 16:
        raise ValueError("Invalid population/sample size")
    order = list(range(population))
    counter = 0
    for bound in range(population, 1, -1):
        # Reject the short bottom interval, instead of modulo-biased sampling.
        threshold = 2**256 % bound
        while True:
            word = (
                int.from_bytes(keccak(SHUFFLE_DOMAIN + seed + uint256(counter)), "big")
                if word_source is None
                else word_source(counter)
            )
            counter += 1
            uint256(word)
            if word >= threshold:
                break
            if counter > 100000:
                raise ValueError("Randomness source failed rejection sampling")
        j = word % bound
        order[bound - 1], order[j] = order[j], order[bound - 1]
    return tuple(order[:size])


@dataclass(frozen=True)
class BeaconSpec:
    chain_hash: str
    public_key: str
    period: int
    genesis_time: int
    scheme: str = "pedersen-bls-chained"
    mode: str = "drand"

    def __post_init__(self):
        if len(bytes.fromhex(self.chain_hash)) != 32 or len(bytes.fromhex(self.public_key)) != 48:
            raise ValueError("Invalid pinned beacon key/hash")
        if self.scheme != "pedersen-bls-chained" or self.mode not in ("drand", "local_test_bls"):
            raise ValueError("Unsupported beacon scheme")
        if (
            type(self.period) is not int
            or self.period <= 0
            or type(self.genesis_time) is not int
            or self.genesis_time < 0
        ):
            raise ValueError("Invalid beacon timing")

    @property
    def config_hash(self):
        return keccak(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode())

    def time_for(self, round_id):
        if type(round_id) is not int or not 1 <= round_id < 2**64:
            raise ValueError("Invalid drand round")
        return self.genesis_time + (round_id - 1) * self.period

    def round_after(self, timestamp):
        return max(1, (timestamp - self.genesis_time) // self.period + 2)

    def verify(self, proof, expected_round, reveal_end):
        if set(proof) != {"round", "signature", "previous_signature", "randomness"}:
            raise ValueError("Malformed beacon proof")
        if (
            type(proof["round"]) is not int
            or proof["round"] != expected_round
            or self.time_for(expected_round) <= reveal_end
        ):
            raise ValueError("Beacon must be fixed before commit and issued after reveal closes")
        signature = bytes.fromhex(proof["signature"])
        previous = bytes.fromhex(proof["previous_signature"])
        if (
            len(signature) != 96
            or len(previous) not in (32, 96)
            or (expected_round > 1 and len(previous) != 96)
        ):
            raise ValueError("Invalid BLS proof length")
        digest = sha256(previous + expected_round.to_bytes(8, "big")).digest()
        if not _verify_bls(bytes.fromhex(self.public_key), digest, signature):
            raise ValueError("Invalid drand BLS signature")
        randomness = sha256(signature).digest()
        if proof["randomness"] != randomness.hex():
            raise ValueError("Beacon randomness does not match signature")
        return randomness


class DrandBeacon:
    def __init__(self, spec, endpoint="https://api.drand.sh"):
        if spec.mode != "drand" or not endpoint.startswith("https://"):
            raise ValueError("Pinned drand spec and HTTPS endpoint required")
        self.spec, self.endpoint = spec, endpoint.rstrip("/")

    def fetch(self, round_id):
        self.spec.time_for(round_id)
        with urlopen(
            f"{self.endpoint}/{self.spec.chain_hash}/public/{round_id}", timeout=15
        ) as response:
            payload = response.read(8193)
        if len(payload) > 8192:
            raise ValueError("Oversized drand response")
        raw = json.loads(payload)
        proof = {k: raw[k] for k in ("round", "signature", "previous_signature", "randomness")}
        self.spec.verify(proof, round_id, self.spec.time_for(round_id) - 1)
        return proof


class LocalTestBeacon:
    """Known test key, virtual time, no threshold or production unpredictability."""

    def __init__(self):
        self.spec = BeaconSpec(
            sha256(b"OCP-FL/local-test-beacon").hexdigest(),
            G2Basic.SkToPk(17).hex(),
            1,
            0,
            mode="local_test_bls",
        )

    def fetch(self, round_id):
        previous = bytes(96)
        signature = G2Basic.Sign(17, sha256(previous + round_id.to_bytes(8, "big")).digest())
        return dict(
            round=round_id,
            signature=signature.hex(),
            previous_signature=previous.hex(),
            randomness=sha256(signature).hexdigest(),
        )
