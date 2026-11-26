"""Issue and verify a per-round on-chain reservation before starting TCP."""
from pathlib import Path
from hashlib import sha256
import json
from eth_abi import encode
from eth_hash.auto import keccak


def deploy(chain,config,mirror,caps):
    base=Path(__file__).resolve().parent
    spec=json.loads((base/'CollectorLeases.json').read_text())
    assert spec['source_sha256']==sha256((base/'CollectorLeases.sol').read_bytes()).hexdigest()
    builder=chain.w3.eth.contract(abi=spec['abi'],bytecode=spec['bytecode'])
    cfg=keccak(json.dumps(config,sort_keys=True,separators=(',',':')).encode())
    tx=builder.constructor(chain.contract.address,cfg,mirror,list(map(int,caps))).transact({'from':chain.oracle})
    receipt=chain.w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status==1
    return chain.w3.eth.contract(address=receipt.contractAddress,abi=spec['abi']),cfg


def reserve(chain,book,cfg,ticket,arm,window):
    rd=ticket['round'];members=list(chain.authorize(ticket,rd,len(chain.clients)))
    mask=sum(1<<g for g in arm)
    expected=keccak(encode(['uint256','address','address','bytes32','uint256','bytes32','uint256[]','uint8','uint16'],
              [chain.chain_id,book.address,chain.contract.address,cfg,rd,bytes.fromhex(ticket['mixed']),members,mask,window]))
    tx=book.functions.reserve(rd,mask,window).transact({'from':chain.oracle})
    receipt=chain.w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status==1 and book.functions.grant(rd).call()==expected
    return dict(digest=expected.hex(),gas=receipt.gasUsed,block=receipt.blockNumber,mask=mask)


def check_rejections(chain,book,ticket,gateway,window):
    from eth_tester.exceptions import TransactionFailed
    rd=ticket['round']
    before=(book.functions.mirrorLeft().call(),[book.functions.siteLeft(i).call() for i in range(4)])
    mask=sum(1<<g for g in gateway)
    same=book.functions.reserve(rd,mask,window)
    r=chain.w3.eth.wait_for_transaction_receipt(same.transact({'from':chain.oracle}))
    assert r.status==1
    for args in ((rd,mask,window+1),(rd,0,window),(rd+1000,mask,window)):
        try:
            book.functions.reserve(*args).call({'from':chain.oracle})
        except TransactionFailed:
            continue
        raise AssertionError('Invalid new reservation accepted')
    assert before==(book.functions.mirrorLeft().call(),[book.functions.siteLeft(i).call() for i in range(4)])
