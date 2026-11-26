// SPDX-License-Identifier: UNLICENSED
pragma solidity 0.8.30;

/// Research OCP-FL. The allowlisted oracle submits BLS proofs; off-chain
/// verifiers MUST check these proofs before training. No on-chain BLS verifier.
contract OcpfScheduler {
    address public immutable oracle;
    bytes32 public immutable beaconConfigHash;
    uint256 public immutable sampleSize;
    address[] private clients;
    mapping(address => bool) public registered;
    uint256 public lastRound;
    struct Round {
        uint256 commitEnd; uint256 revealEnd; uint256 beaconRound;
        uint256 beaconTime; bool finalized; bytes32 beacon;
        bytes32 mixed; bytes32 modelHash;
    }
    mapping(uint256 => Round) private rounds;
    mapping(uint256 => mapping(address => bytes32)) public commitments;
    mapping(uint256 => mapping(address => bool)) public revealed;
    mapping(uint256 => mapping(address => bytes32)) public seeds;
    mapping(uint256 => bytes) public beaconSignature;
    mapping(uint256 => bytes) public previousSignature;
    mapping(uint256 => uint256[]) private schedules;
    mapping(address => uint256) public missedReveals;
    event RoundOpened(uint256 indexed round, uint256 commitEnd, uint256 revealEnd, uint256 beaconRound, uint256 beaconTime);
    event Commit(uint256 indexed round, address indexed client, bytes32 commitment);
    event Reveal(uint256 indexed round, address indexed client, bytes32 seed);
    event LateReveal(uint256 indexed round, address indexed client, bytes32 seed);
    event NonReveal(uint256 indexed round, address indexed client);
    event RoundFinal(uint256 indexed round, bytes32 beacon, bytes32 mixed, uint256[] selected, bool noReveals);
    event ModelRecorded(uint256 indexed round, bytes32 modelHash);
    modifier onlyOracle() { require(msg.sender == oracle, "oracle only"); _; }
    constructor(address[] memory population_, uint256 k, address oracle_, bytes32 configHash) {
        require(population_.length >= 2 && population_.length <= 16, "population 2..16");
        require(k > 0 && k <= population_.length && oracle_ != address(0), "invalid setup");
        for (uint256 i; i < population_.length; ++i) {
            require(population_[i] != address(0) && !registered[population_[i]], "duplicate client");
            registered[population_[i]] = true; clients.push(population_[i]);
        }
        sampleSize = k; oracle = oracle_; beaconConfigHash = configHash;
    }
    function population() external view returns (address[] memory) { return clients; }
    function roundMeta(uint256 t) external view returns (Round memory) { return rounds[t]; }
    function selected(uint256 t) external view returns (uint256[] memory) { return schedules[t]; }
    function commitmentFor(bytes32 seed, address client, uint256 t) public view returns (bytes32) {
        return keccak256(abi.encodePacked("OCP-FL/commit/v1", block.chainid, address(this), seed, client, t));
    }
    function openRound(uint256 t, uint256 commitEnd, uint256 revealEnd, uint256 beaconRound_, uint256 beaconTime) external onlyOracle {
        require(t == lastRound + 1 && (t == 1 || rounds[t-1].finalized), "round order");
        require(block.timestamp < commitEnd && commitEnd < revealEnd && revealEnd < beaconTime, "window order");
        require(beaconRound_ > 0, "beacon round");
        rounds[t].commitEnd = commitEnd; rounds[t].revealEnd = revealEnd;
        rounds[t].beaconRound = beaconRound_; rounds[t].beaconTime = beaconTime;
        lastRound = t; emit RoundOpened(t, commitEnd, revealEnd, beaconRound_, beaconTime);
    }
    function commit(uint256 t, bytes32 commitment) external {
        require(registered[msg.sender] && t == lastRound && t > 0, "client or round");
        require(block.timestamp < rounds[t].commitEnd, "commit closed");
        require(commitment != bytes32(0) && commitments[t][msg.sender] == bytes32(0), "duplicate commit");
        commitments[t][msg.sender] = commitment; emit Commit(t, msg.sender, commitment);
    }
    function reveal(uint256 t, bytes32 seed) external returns (bool) {
        require(t == lastRound && t > 0 && commitments[t][msg.sender] != bytes32(0), "no commit");
        require(!revealed[t][msg.sender] && block.timestamp >= rounds[t].commitEnd, "reveal phase");
        require(commitmentFor(seed, msg.sender, t) == commitments[t][msg.sender], "wrong seed");
        if (block.timestamp >= rounds[t].revealEnd) { emit LateReveal(t, msg.sender, seed); return false; }
        revealed[t][msg.sender] = true; seeds[t][msg.sender] = seed;
        emit Reveal(t, msg.sender, seed); return true;
    }
    function finalize(uint256 t, bytes32 beacon, bytes calldata signature, bytes calldata previous) external onlyOracle {
        Round storage r = rounds[t];
        require(t == lastRound && t > 0 && !r.finalized && block.timestamp >= r.beaconTime, "not ready");
        require(signature.length == 96 && previous.length <= 96, "proof length");
        bytes32 mixed = beacon; uint256 count;
        for (uint256 i; i < clients.length; ++i) {
            address client = clients[i];
            if (revealed[t][client]) { mixed ^= seeds[t][client]; ++count; }
            else if (commitments[t][client] != bytes32(0)) { ++missedReveals[client]; emit NonReveal(t, client); }
        }
        uint256[] memory order = new uint256[](clients.length);
        for (uint256 i; i < order.length; ++i) order[i] = i;
        uint256 counter;
        for (uint256 bound = order.length; bound > 1; --bound) {
            uint256 word; uint256 threshold = addmod(type(uint256).max, 1, bound);
            do { word = uint256(keccak256(abi.encodePacked("OCP-FL/shuffle/v1", mixed, counter++))); } while (word < threshold);
            uint256 j = word % bound;
            (order[bound-1], order[j]) = (order[j], order[bound-1]);
        }
        for (uint256 i; i < sampleSize; ++i) schedules[t].push(order[i]);
        r.beacon = beacon; r.mixed = mixed; r.finalized = true;
        beaconSignature[t] = signature; previousSignature[t] = previous;
        emit RoundFinal(t, beacon, mixed, schedules[t], count == 0);
    }
    function recordModel(uint256 t, bytes32 digest) external onlyOracle {
        require(rounds[t].finalized && rounds[t].modelHash == bytes32(0) && digest != bytes32(0), "model state");
        rounds[t].modelHash = digest; emit ModelRecorded(t, digest);
    }
}
