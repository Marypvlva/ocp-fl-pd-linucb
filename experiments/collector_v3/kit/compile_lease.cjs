const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const solc = require(require.resolve('solc', {paths:[path.resolve(process.argv[2] || '.')]}));
const filename = 'CollectorLeases.sol';
const file = fs.readFileSync(path.join(__dirname, filename), 'utf8');
const settings = {optimizer:{enabled:true,runs:200},outputSelection:{'*':{'*':['abi','evm.bytecode.object']}}};
const result = JSON.parse(solc.compile(JSON.stringify({language:'Solidity',sources:{[filename]:{content:file}},settings})));
for (const error of result.errors || []) if (error.severity === 'error') throw Error(error.formattedMessage);
const contract = result.contracts[filename].CollectorLeases;
const artifact = {compiler:solc.version(),source_sha256:crypto.createHash('sha256').update(file).digest('hex'),
                  abi:contract.abi,bytecode:contract.evm.bytecode.object,settings};
fs.writeFileSync(path.join(__dirname,'CollectorLeases.json'),JSON.stringify(artifact,null,2)+'\n');
