# Panoptic Protocol Security Audit Findings

## Executive Summary

This audit covers the Panoptic Protocol - a DeFi options protocol built on Uniswap V3/V4. After thorough analysis of the core contracts, I've identified several potential vulnerabilities ranging from medium to high severity.

---

## HIGH SEVERITY FINDINGS

### H-1: Oracle Manipulation via Clamped Tick Updates

**Location:** `contracts/types/OraclePack.sol` - `clampTick()` and `insertObservation()`

**Description:**
The internal oracle uses a clamping mechanism (`MAX_CLAMP_DELTA = 149 ticks`) to limit how much the price can move between observations. While this is designed to prevent manipulation, it creates a vulnerability where an attacker can gradually manipulate the oracle over multiple blocks.

```solidity
function clampTick(
    int24 newTick,
    OraclePack _oraclePack,
    int24 clampDelta
) internal pure returns (int24 clamped) {
    int24 _lastTick = _oraclePack.lastTick();
    if (newTick > _lastTick + clampDelta) {
        clamped = _lastTick + clampDelta;
    } else if (newTick < _lastTick - clampDelta) {
        clamped = _lastTick - clampDelta;
    } else {
        clamped = newTick;
    }
}
```

**Attack Vector:**
1. Attacker manipulates Uniswap pool price by 149 ticks
2. Waits for epoch change (64 seconds)
3. Repeats manipulation, moving oracle 149 ticks per epoch
4. After ~10 epochs (~10 minutes), oracle can be moved ~1490 ticks (~15% price change)
5. Liquidate positions that appear insolvent at manipulated oracle price

**Impact:** Malicious liquidations of solvent positions, theft of liquidation bonuses.

**Recommendation:** 
- Implement cumulative deviation tracking across multiple epochs
- Add a secondary TWAP check from Uniswap's built-in oracle
- Increase the epoch duration or reduce clamp delta

---

### H-2: Delegation Balance Manipulation During Liquidation

**Location:** `contracts/CollateralTracker.sol` - `delegate()`, `revoke()`, `settleLiquidation()`

**Description:**
The delegation mechanism adds `type(uint248).max` virtual shares to a user's balance during liquidation. However, there's a race condition where interest accrual between `delegate()` and `revoke()` can consume real shares, leading to incorrect accounting.

```solidity
function delegate(address delegatee) external onlyPanopticPool {
    uint256 interestShares = previewWithdraw(_owedInterest(delegatee));
    uint256 balance = balanceOf[delegatee];
    uint256 balanceConsumedByInterest = interestShares > balance ? balance : 0;
    balanceOf[delegatee] += type(uint248).max - balanceConsumedByInterest;
}
```

**Issue:** If interest accrues between the `delegate()` call and subsequent operations, the `balanceConsumedByInterest` calculation becomes stale, potentially allowing the liquidatee to retain more shares than they should.

**Impact:** Protocol loss through incorrect share accounting during liquidations.

**Recommendation:**
- Use transient storage to lock interest accrual during liquidation
- Perform atomic delegation/revocation with interest settlement

---

### H-3: Premium Haircut Bypass via Position Ordering

**Location:** `contracts/RiskEngine.sol` - `haircutPremia()`

**Description:**
The haircut mechanism iterates through positions in order, applying haircuts proportionally. An attacker can structure their positions such that the haircut is applied to positions with minimal premium, leaving high-premium positions intact.

```solidity
for (uint256 i = 0; i < positionIdList.length; i++) {
    TokenId tokenId = positionIdList[i];
    for (uint256 leg = 0; leg < tokenId.countLegs(); ++leg) {
        if (tokenId.isLong(leg) == 1 && LeftRightSigned.unwrap(_premiasByLeg[i][leg]) != 0) {
            // haircut calculation...
        }
    }
}
```

**Attack Vector:**
1. Create multiple positions with varying premium amounts
2. When facing liquidation, ensure low-premium positions are processed first
3. Haircut is exhausted on low-value positions
4. High-premium positions escape haircut

**Impact:** Reduced protocol loss recovery during liquidations.

**Recommendation:**
- Sort positions by premium amount before applying haircuts
- Apply haircuts proportionally across all positions simultaneously

---

## MEDIUM SEVERITY FINDINGS

### M-1: Interest Rate Model Drift During Inactivity

**Location:** `contracts/RiskEngine.sol` - `_borrowRate()`

**Description:**
The interest rate model caps elapsed time at `IRM_MAX_ELAPSED_TIME = 4096 seconds`. During periods of low activity, the rate can drift significantly from the target, and the cap prevents proper convergence.

```solidity
int256 elapsed = Math.min(
    int256(block.timestamp) - int256(previousTime),
    IRM_MAX_ELAPSED_TIME
);
```

**Impact:** Interest rates may not properly reflect market conditions after periods of inactivity.

**Recommendation:** Implement a catch-up mechanism or use a different rate model for extended inactivity periods.

---

### M-2: Cross-Margining Buffer Manipulation

**Location:** `contracts/RiskEngine.sol` - `_crossBufferRatio()` and `isAccountSolvent()`

**Description:**
The cross-buffer ratio decreases linearly from `crossBuffer` to 0 between 50% and 90% utilization. An attacker can manipulate pool utilization to reduce their collateral requirements.

```solidity
function _crossBufferRatio(int256 utilization, uint256 crossBuffer) internal view returns (uint256) {
    uint256 utilizationScaled = uint256(utilization * 1_000);
    if (utilizationScaled < TARGET_POOL_UTIL) return crossBuffer;
    if (utilizationScaled > SATURATED_POOL_UTIL) return 0;
    return ((crossBuffer * (SATURATED_POOL_UTIL - utilizationScaled)) / (SATURATED_POOL_UTIL - TARGET_POOL_UTIL));
}
```

**Attack Vector:**
1. Deposit large amounts to reduce utilization below 50%
2. Open leveraged positions with maximum cross-margining benefit
3. Withdraw deposits, increasing utilization
4. Positions now have insufficient collateral but may not be immediately liquidatable

**Impact:** Undercollateralized positions, potential protocol loss.

**Recommendation:**
- Use utilization at position opening for collateral calculations
- Implement utilization-based position limits

---

### M-3: Safe Mode Bypass via EMA Convergence

**Location:** `contracts/RiskEngine.sol` - `isSafeMode()`

**Description:**
Safe mode is triggered based on EMA divergence. However, after a price shock, EMAs will naturally converge, potentially exiting safe mode before the market has stabilized.

```solidity
bool externalShock = Math.abs(currentTick - spotEMA) > MAX_TICKS_DELTA;
bool internalDisagreement = Math.abs(spotEMA - fastEMA) > (MAX_TICKS_DELTA / 2);
bool highDivergence = Math.abs(medianTick - slowEMA) > (MAX_TICKS_DELTA * 2);
```

**Impact:** Premature exit from safe mode could allow risky operations during volatile periods.

**Recommendation:**
- Implement a minimum safe mode duration
- Add hysteresis to safe mode entry/exit thresholds

---

### M-4: Liquidation Bonus Calculation Rounding

**Location:** `contracts/RiskEngine.sol` - `getLiquidationBonus()`

**Description:**
The liquidation bonus calculation involves multiple divisions and conversions between tokens. Rounding errors can accumulate, potentially resulting in incorrect bonus amounts.

```solidity
uint256 requiredRatioX128 = Math.mulDiv(tokenData0.leftSlot(), 2 ** 128, thresholdCross);
uint256 bonus0U = Math.mulDiv128(bonusCross, requiredRatioX128);
bonus0 = int256(bonus0U);
bonus1 = int256(PanopticMath.convert0to1(bonusCross - bonus0U, atSqrtPriceX96));
```

**Impact:** Liquidators may receive slightly more or less than intended, affecting protocol economics.

**Recommendation:** Use consistent rounding direction throughout calculations.

---

### M-5: Position Hash Collision Risk

**Location:** `contracts/PanopticPool.sol` - `_updatePositionsHash()`

**Description:**
The positions hash uses XOR of keccak256 hashes. While collision probability is low, a determined attacker could potentially find colliding position IDs.

```solidity
uint256 newHash = PanopticMath.updatePositionsHash(
    s_positionsHash[account],
    tokenId,
    addFlag
);
```

**Impact:** Potential bypass of position validation checks.

**Recommendation:** Use a more collision-resistant accumulator or maintain a mapping of positions.

---

## LOW SEVERITY FINDINGS

### L-1: Unchecked Return Value in BuilderWallet

**Location:** `contracts/RiskEngine.sol` - `BuilderWallet.sweep()`

**Description:**
The sweep function checks the return value of `transfer()` but doesn't handle the case where the token doesn't return a boolean (non-standard ERC20).

```solidity
bool ok = IERC20(token).transfer(to, bal);
if (!ok) {
    revert Errors.TransferFailed(token, address(this), bal, bal);
}
```

**Recommendation:** Use SafeTransferLib for all token transfers.

---

### L-2: Epoch-Based Timekeeping Precision Loss

**Location:** `contracts/types/OraclePack.sol`

**Description:**
The oracle uses 64-second epochs (`block.timestamp >> 6`), which can cause up to 63 seconds of timing imprecision.

**Impact:** Minor timing inconsistencies in oracle updates.

**Recommendation:** Document this behavior clearly; consider finer granularity if precision is critical.

---

### L-3: Missing Zero-Address Checks

**Location:** Multiple contracts

**Description:**
Several functions don't validate that address parameters are non-zero.

**Recommendation:** Add zero-address validation for critical parameters.

---

## INFORMATIONAL FINDINGS

### I-1: Gas Optimization Opportunities

- `_checkSolvencyAtTicks()` could cache repeated calculations
- Position iteration in `_calculateAccumulatedPremia()` could be optimized
- Multiple storage reads in `_accrueInterest()` could be consolidated

### I-2: Documentation Gaps

- The interaction between safe mode levels and position restrictions could be clearer
- Cross-margining rules for complex positions need more documentation

### I-3: Centralization Risks

- Guardian can lock pools indefinitely via `lockPool()`
- Builder factory owner has significant control over fee distribution

---

## Recommendations Summary

1. **Critical:** Implement additional oracle manipulation protections
2. **High:** Review and harden the delegation/revocation mechanism
3. **Medium:** Add utilization-based position limits
4. **Medium:** Implement safe mode hysteresis
5. **General:** Comprehensive fuzzing and invariant testing recommended

---

*Audit conducted: December 21, 2025*
*Auditor: Kiro Security Analysis*


---

## ADDITIONAL HIGH SEVERITY FINDINGS

### H-4: Reentrancy in Force Exercise Flow

**Location:** `contracts/PanopticPool.sol` - `_forceExercise()`

**Description:**
The force exercise flow delegates virtual shares, burns positions, then refunds and revokes. If the underlying token has callbacks (e.g., ERC777), an attacker could potentially reenter during the refund phase.

```solidity
function _forceExercise(...) internal {
    // 1. Delegate virtual shares
    ct0.delegate(account);
    ct1.delegate(account);
    
    // 2. Burn options (interacts with Uniswap)
    _burnOptions(...);
    
    // 3. Refund (potential callback point)
    ct0.refund(account, msg.sender, refundAmounts.rightSlot());
    ct1.refund(account, msg.sender, refundAmounts.leftSlot());
    
    // 4. Revoke
    ct0.revoke(account);
    ct1.revoke(account);
}
```

**Attack Vector:**
1. Create a position with an ERC777 token as collateral
2. When force exercised, use the token callback during refund
3. Reenter to manipulate state before revoke is called

**Impact:** Potential double-spending of virtual shares or manipulation of liquidation outcomes.

**Recommendation:**
- Add reentrancy guards to all external-facing functions
- Use checks-effects-interactions pattern strictly
- Consider using transient storage for reentrancy locks

---

### H-5: Spread Calculation Underflow in Calendar Spreads

**Location:** `contracts/RiskEngine.sol` - `_computeSpread()`

**Description:**
The spread calculation for calendar spreads uses unchecked arithmetic that could underflow in edge cases:

```solidity
unchecked {
    int24 deltaWidth = _tokenId.width(index) - _tokenId.width(partnerIndex);
    if (deltaWidth < 0) deltaWidth = -deltaWidth;
    
    if (tokenType == 0) {
        spreadRequirement +=
            (amountsMoved.rightSlot() *
                uint256(int256(deltaWidth * _tokenId.tickSpacing()))) /
            80000;
    }
}
```

**Issue:** If `deltaWidth * tickSpacing` produces a negative value that's then cast to uint256, it will wrap to a very large number.

**Impact:** Incorrect collateral requirements, potentially allowing undercollateralized positions.

**Recommendation:** Add explicit bounds checking before the cast.

---

### H-6: Premium Settlement Race Condition

**Location:** `contracts/PanopticPool.sol` - `_settlePremium()` and `_updateSettlementPostBurn()`

**Description:**
When settling premium, there's a window where the settled tokens are updated but the position's premium accumulator hasn't been updated yet. A malicious actor could exploit this timing.

```solidity
// In _updateSettlementPostBurn:
settledTokens = settledTokens.sub(availablePremium);  // State change 1
// ... other operations ...
s_options[owner][tokenId][leg] = ...;  // State change 2 (later)
```

**Attack Vector:**
1. Monitor mempool for premium settlement transactions
2. Front-run with a transaction that reads the stale premium accumulator
3. Extract value from the timing gap

**Impact:** Premium theft through front-running.

**Recommendation:** Use atomic updates or implement commit-reveal schemes for premium settlement.

---

## ADDITIONAL MEDIUM SEVERITY FINDINGS

### M-6: Liquidity Spread Check Bypass

**Location:** `contracts/PanopticPool.sol` - `_checkLiquiditySpread()`

**Description:**
The liquidity spread check returns early if both `netLiquidity` and `removedLiquidity` are zero:

```solidity
if (netLiquidity == 0 && removedLiquidity == 0) return totalLiquidity;
```

This allows closing short positions without spread checks, but could be exploited in edge cases where liquidity was manipulated to zero.

**Impact:** Potential bypass of spread limits in manipulated pools.

**Recommendation:** Add additional validation for the zero-liquidity case.

---

### M-7: Tick Delta Accumulation in Dispatch

**Location:** `contracts/PanopticPool.sol` - `dispatch()`

**Description:**
The dispatch function accumulates tick deltas across multiple operations:

```solidity
cumulativeTickDeltas = LeftRightSigned
    .wrap(0)
    .addToRightSlot(
        cumulativeTickDeltas.rightSlot() +
            int128(Math.abs(int24(cumulativeTickDeltas.leftSlot()) - finalTick))
    )
    .addToLeftSlot(finalTick);
```

The check `cumulativeTickDeltas.rightSlot() > int256(uint256(2 * riskParameters.tickDeltaLiquidation()))` could be bypassed by structuring operations to minimize apparent price impact while still achieving significant cumulative movement.

**Impact:** Price manipulation through carefully structured multi-operation transactions.

**Recommendation:** Track absolute tick movement, not just deltas from previous operation.

---

### M-8: Interest Insolvency Penalty Gaming

**Location:** `contracts/CollateralTracker.sol` - `_accrueInterest()`

**Description:**
When a user owes more interest than their balance, they receive an "insolvency penalty" but their debt continues to compound:

```solidity
if (shares > userBalance) {
    if (!isDeposit) {
        burntInterestValue = Math.mulDiv(userBalance, _totalAssets, totalSupply()).toUint128();
        _burn(_owner, userBalance);
        // DO NOT update index - debt continues compounding
        userBorrowIndex = userState.rightSlot();
    }
}
```

**Attack Vector:**
1. Borrow maximum amount
2. Let interest accrue beyond balance
3. Deposit small amount to trigger insolvency penalty
4. Debt continues compounding from original point
5. Repeat to accumulate unbounded debt that will never be paid

**Impact:** Bad debt accumulation in the protocol.

**Recommendation:** Cap maximum debt or implement debt write-off mechanisms.

---

### M-9: Builder Code Validation Timing

**Location:** `contracts/RiskEngine.sol` - `getFeeRecipient()`

**Description:**
The builder code validation only checks if the wallet contract exists:

```solidity
function getFeeRecipient(uint256 builderCode) external view returns (address feeRecipient) {
    feeRecipient = _computeBuilderWallet(builderCode);
    if (builderCode != 0) {
        if (feeRecipient.code.length == 0) revert Errors.InvalidBuilderCode();
    }
}
```

However, `getRiskParameters()` uses `_computeBuilderWallet()` directly without this validation, allowing fees to be directed to non-existent addresses.

**Impact:** Fees could be lost to undeployed builder wallets.

**Recommendation:** Validate builder code in `getRiskParameters()` as well.

---

## STATE FLOW ANALYSIS

### Critical State Transitions

1. **Position Minting Flow:**
   ```
   dispatch() → _mintOptions() → SFPM.mintTokenizedPosition() → _updateSettlementPostMint() → _payCommissionAndWriteData()
   ```
   - Risk: Premium accumulator updates must be atomic with liquidity changes

2. **Liquidation Flow:**
   ```
   dispatchFrom() → _liquidate() → delegate() → _burnAllOptionsFrom() → getLiquidationBonus() → haircutPremia() → settleLiquidation() → revoke()
   ```
   - Risk: Multiple state changes with external calls between them

3. **Oracle Update Flow:**
   ```
   pokeOracle() → computeInternalMedian() → insertObservation() → updateEMAs()
   ```
   - Risk: Clamping allows gradual manipulation

### Invariants That Should Hold

1. `totalSupply() + _internalSupply == sum(balanceOf[all_users])` (excluding delegated shares)
2. `s_settledTokens[chunk] >= sum(available_premium_for_all_sellers_in_chunk)`
3. `s_assetsInAMM + s_depositedAssets == total_protocol_assets`
4. For any user: `collateral_balance >= maintenance_requirement` (when not in liquidation)

---

## TESTING RECOMMENDATIONS

### Fuzzing Targets

1. Oracle manipulation sequences
2. Liquidation bonus calculations with extreme values
3. Interest rate model under various utilization patterns
4. Cross-margining calculations with complex position combinations
5. Premium settlement with concurrent operations

### Invariant Tests

1. Total supply consistency
2. Settled tokens never negative
3. Collateral requirements always positive
4. Oracle tick bounds
5. Position hash uniqueness

### Integration Tests

1. Multi-block oracle manipulation attempts
2. Flash loan attack simulations
3. Sandwich attack scenarios on liquidations
4. MEV extraction attempts on premium settlements

---

*Additional findings added: December 21, 2025*
