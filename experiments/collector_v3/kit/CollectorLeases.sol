// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

interface ICertifiedRounds {
    struct Round {
        uint256 commitEnd; uint256 revealEnd; uint256 beaconRound; uint256 beaconTime;
        bool finalized; bytes32 beacon; bytes32 mixed; bytes32 modelHash;
    }
    function roundMeta(uint256 t) external view returns (Round memory);
    function selected(uint256 t) external view returns (uint256[] memory);
}

contract CollectorLeases {
    ICertifiedRounds public immutable source;
    address public immutable planner;
    bytes32 public immutable policyConfig;
    uint256 public lastRound;
    uint256 public mirrorLeft;
    uint256[4] public siteLeft;
    mapping(uint256 => bytes32) public grant;
    event Reserved(uint256 indexed round, uint8 mask, uint16 windowMs, bytes32 digest);

    constructor(address rounds, bytes32 config, uint256 mirror, uint256[4] memory caps) {
        require(rounds.code.length > 0 && config != bytes32(0), "configuration");
        source = ICertifiedRounds(rounds);
        planner = msg.sender;
        policyConfig = config;
        mirrorLeft = mirror;
        siteLeft = caps;
    }

    function reserve(uint256 t, uint8 mask, uint16 windowMs) external returns (bytes32 digest) {
        require(msg.sender == planner && mask > 0 && mask < 16 && windowMs > 0, "request");
        ICertifiedRounds.Round memory round = source.roundMeta(t);
        require(round.finalized, "round");
        digest = keccak256(abi.encode(block.chainid, address(this), address(source), policyConfig,
                                      t, round.mixed, source.selected(t), mask, windowMs));
        if (grant[t] != bytes32(0)) {
            require(grant[t] == digest, "conflict");
            return digest;
        }
        require(t == lastRound + 1 && round.modelHash == bytes32(0), "order");
        uint256 count;
        for (uint8 i = 0; i < 4; ++i) {
            if ((mask & (uint8(1) << i)) != 0) {
                count++;
                require(siteLeft[i] > 0, "site quota");
                siteLeft[i]--;
            }
        }
        require(count <= 2, "too many collectors");
        if (count == 2) {
            require(mirrorLeft > 0, "mirror quota");
            mirrorLeft--;
        }
        grant[t] = digest;
        lastRound = t;
        emit Reserved(t, mask, windowMs, digest);
    }
}
