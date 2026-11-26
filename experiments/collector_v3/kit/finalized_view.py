"""Read a previously fully verified certificate through its immutable, canonical block anchor."""
from hashlib import sha256
import json
from eth_hash.auto import keccak


def fingerprint(ticket):
    return sha256(json.dumps(ticket,sort_keys=True,separators=(',',':')).encode()).digest()


class FinalizedView:
    def __init__(self, chain, verified_ceremonies):
        self.chain=chain
        self.known={}
        for ticket in verified_ceremonies:
            # OcpChain.run_local_ceremony already authorized this exact ticket using canonical events and BLS.
            digest=sha256(json.dumps(ticket,sort_keys=True).encode()).digest()
            if digest not in chain.validated:
                raise ValueError('Only previously verified certificates can enter the read cache')
            self.known[ticket['round']]=fingerprint(ticket)

    def authorize(self,ticket,round_id,workers):
        chain=self.chain
        if (not isinstance(ticket,dict) or ticket.get('round')!=round_id or workers!=len(chain.clients)
                or self.known.get(round_id)!=fingerprint(ticket)):
            raise ValueError('Modified certificate, round or registry')
        address=chain.contract.address
        if (chain.w3.eth.chain_id!=ticket['chain_id'] or address!=ticket['contract']
                or chain.w3.eth.get_block(ticket['block_number']).hash.hex()!=ticket['block_hash']
                or chain.w3.eth.block_number-ticket['block_number']+1<chain.finality
                or keccak(bytes(chain.w3.eth.get_code(address)))!=chain.code_hash):
            raise ValueError('Noncanonical verified certificate')
        return tuple(ticket['selected'])
