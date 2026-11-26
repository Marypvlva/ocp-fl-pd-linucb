"""Prepare a research source contract on an OPERATOR-SUPPLIED consensus RPC.

No cloud resources are created. Transactions run only with --execute.
Private signing keys are read from OCP_SIGNING_KEYS and are never printed.
"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from web3 import Web3
from adaptive_planning.ocp_randomness import BeaconSpec


def signing(w3):
    from web3.middleware import SignAndSendRawMiddlewareBuilder
    from eth_account import Account
    from web3.middleware import ExtraDataToPOAMiddleware

    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    for key in json.loads(os.environ.get("OCP_SIGNING_KEYS", "[]")):
        account = Account.from_key(key)
        w3.middleware_onion.add(
            SignAndSendRawMiddlewareBuilder.build(account), name="sign-" + account.address
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rpc", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--clients", nargs="+", required=True)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--mode", default="joint")
    p.add_argument("--beacon", type=Path, default=ROOT / "configs/drand_mainnet.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not a.rpc.startswith("https://"):
        raise ValueError("External RPC must use trusted HTTPS")
    if not 2 <= len(a.clients) <= 16 or not 1 <= a.k <= len(a.clients):
        raise ValueError("Population/sample size")
    spec = BeaconSpec(**json.loads(a.beacon.read_text()))
    w3 = Web3(Web3.HTTPProvider(a.rpc))
    signing(w3)
    # Fail before deploying when consensus-finalized state is unavailable.
    finalized = w3.eth.get_block("finalized")
    print(
        json.dumps(
            dict(chain_id=w3.eth.chain_id, finalized_number=finalized.number, execute=a.execute)
        )
    )
    if not a.execute:
        return
    if a.out.exists():
        raise ValueError("Use a new output config path")
    abi = json.loads((ROOT / "contracts/artifacts/OcpfScheduler.json").read_text())
    factory = w3.eth.contract(abi=abi["abi"], bytecode=abi["evm"]["bytecode"]["object"])
    tx = factory.constructor(a.clients, a.k, a.oracle, spec.config_hash).transact(
        {"from": a.oracle}
    )
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    if receipt.status != 1:
        raise ValueError("Source deployment reverted")
    a.out.write_text(
        json.dumps(
            dict(
                rpc=a.rpc,
                confirmations=2,
                beacon=spec.__dict__,
                sources={a.mode: dict(contract=receipt.contractAddress, oracle=a.oracle)},
            ),
            indent=2,
        )
        + "\n"
    )
    a.out.chmod(0o600)


if __name__ == "__main__":
    main()
