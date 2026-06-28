package com.example.bottleneckreader

class PacketClockDecoder {
    data class Result(
        val bits: String? = null,
        val bitLlrs: FloatArray? = null,
        val debugInfo: DebugInfo? = null,
        val packetKind: PacketKind? = null,
        val payloadErasures: Int = 0,
        val failure: FailureReason? = null,
    )

    data class DebugInfo(
        val periodNs: Long = 0L,
        val measuredPeriodNs: Long = 0L,
        val preambleEstimateMode: String = "",
        val preambleFirstIntervalNs: Long = 0L,
        val preambleSecondIntervalNs: Long = 0L,
        val periodsElapsed: Long = 0L,
        val emittedSymbol: Long = 0L,
        val decision: String = "",
        val rejectReason: String = "",
        val symbolStartNs: Long = 0L,
        val symbolEmitNs: Long = 0L,
        val sampleCount: Int = 0,
        val sampleWeightSum: Float = 0f,
        val averageSamplePhase: Float = 0f,
        val averages: FloatArray = FloatArray(0),
        val peaks: FloatArray = FloatArray(0),
        val reliabilities: FloatArray = FloatArray(0),
        val onThreshold: Float = 0f,
        val offThreshold: Float = 0f,
    )

    enum class PacketKind {
        Preamble,
        Payload,
    }

    enum class FailureReason(val message: String) {
        MarkerLost("Decode failed: marker lost"),
        StreamInterrupted("Decode failed: stream interrupted"),
        PreambleLost("Decode failed: preamble lost"),
        SymbolSkipped("Decode failed: symbol period skipped"),
        NoSymbolSamples("Decode failed: no samples in symbol window"),
    }

    val debugState: String
        get() = when (phase) {
            Phase.WaitingZeros -> "zeros $zeroFrames/$REQUIRED_ZERO_FRAMES"
            Phase.WaitingPreamble -> "preamble $preambleIndex/${PREAMBLE.lastIndex}"
            Phase.Active -> "active ${periodNs / 1_000_000L}ms"
        }

    private enum class Phase {
        WaitingZeros,
        WaitingPreamble,
        Active,
    }

    private var phase = Phase.WaitingZeros
    private var zeroFrames = 0
    private var preambleIndex = 1
    private var preambleStarted = false
    private var lastPreambleAtNs = 0L
    private val preambleIntervalsNs = LongArray(PREAMBLE.size - 2)
    private var preambleIntervalCount = 0
    private var preambleMisses = 0
    private var lastPreambleScores: FloatArray? = null
    private var pendingPreambleEdgeAtNs = 0L
    private var pendingPreambleEdgeScore = 0f
    private var lastPreambleEstimateMode = ""
    private var lastPreambleFirstIntervalNs = 0L
    private var lastPreambleSecondIntervalNs = 0L
    private var periodNs = DEFAULT_PERIOD_NS
    private var clockStartNs = 0L
    private var symbolStartNs = 0L
    private var nextEmitAtNs = 0L
    private var emittedSymbols = 0L
    private var bitStates: BooleanArray? = null
    private val symbolScoreSums = FloatArray(BIT_COUNT)
    private val symbolScoreMax = FloatArray(BIT_COUNT)
    private var symbolWeightSum = 0f
    private var symbolPhaseWeightSum = 0f
    private var symbolSampleCount = 0

    @Synchronized
    fun accept(timestampNs: Long, scores: FloatArray?): Result? {
        if (scores == null || scores.size != BIT_COUNT) {
            return acceptMissingDetection(timestampNs)
        }

        val packet = scoresToPacket(scores)
        return when (phase) {
            Phase.WaitingZeros -> acceptWaitingZeros(scores)
            Phase.WaitingPreamble -> acceptPreamble(scores, packet, timestampNs)
            Phase.Active -> acceptActive(scores, timestampNs)
        }
    }

    @Synchronized
    fun reset(): Result? {
        val hadLock = phase != Phase.WaitingZeros
        resetInternal()
        return if (hadLock) Result(failure = FailureReason.StreamInterrupted) else null
    }

    @Synchronized
    fun finishMessage() {
        resetInternal()
    }

    private fun acceptWaitingZeros(scores: FloatArray): Result? {
        if (isStableZero(scores)) {
            zeroFrames += 1
            if (zeroFrames >= REQUIRED_ZERO_FRAMES) {
                resetPreamble()
                lastPreambleScores = scores.copyOf()
                phase = Phase.WaitingPreamble
            }
        } else {
            zeroFrames = 0
        }
        return null
    }

    private fun acceptMissingDetection(timestampNs: Long): Result? {
        return when (phase) {
            Phase.WaitingZeros -> null
            Phase.WaitingPreamble -> {
                if (preambleStarted) countPreambleMiss() else null
            }
            Phase.Active -> acceptActive(scores = null, timestampNs = timestampNs)
        }
    }

    private fun acceptPreamble(scores: FloatArray, packet: String, timestampNs: Long): Result? {
        val expected = PREAMBLE[preambleIndex]
        val previous = PREAMBLE[preambleIndex - 1]
        observeExpectedPreambleEdge(scores, previous, expected, timestampNs)

        return when {
            matchesSoft(scores, expected) -> {
                preambleMisses = 0
                preambleStarted = true
                val observedAtNs = consumePreambleEdgeTimestampOr(timestampNs)
                var lastIntervalNs = 0L
                if (lastPreambleAtNs != 0L) {
                    lastIntervalNs = observedAtNs - lastPreambleAtNs
                    if (preambleIntervalCount < preambleIntervalsNs.size) {
                        preambleIntervalsNs[preambleIntervalCount] = lastIntervalNs
                        preambleIntervalCount += 1
                    }
                }
                val previousPreambleAtNs = lastPreambleAtNs
                lastPreambleAtNs = observedAtNs
                preambleIndex += 1
                var debugInfo: DebugInfo? = null

                if (preambleIndex == PREAMBLE.size) {
                    val measuredPeriodNs = estimatePreamblePeriod()
                    periodNs = stabilizeMeasuredPeriod(measuredPeriodNs)
                    val finalPreambleAtNs = correctedFinalPreambleAt(
                        observedFinalAtNs = observedAtNs,
                        previousPreambleAtNs = previousPreambleAtNs,
                        lastIntervalNs = lastIntervalNs,
                    )
                    debugInfo = DebugInfo(
                        periodNs = periodNs,
                        measuredPeriodNs = measuredPeriodNs,
                        preambleEstimateMode = lastPreambleEstimateMode,
                        preambleFirstIntervalNs = lastPreambleFirstIntervalNs,
                        preambleSecondIntervalNs = lastPreambleSecondIntervalNs,
                    )
                    clockStartNs = finalPreambleAtNs + periodNs
                    symbolStartNs = clockStartNs
                    nextEmitAtNs = symbolStartNs + periodNs
                    emittedSymbols = 0L
                    clearSymbolSamples()
                    phase = Phase.Active
                }
                Result(bits = expected, debugInfo = debugInfo, packetKind = PacketKind.Preamble)
            }
            matchesSoft(scores, previous) -> null
            packet == ZERO_PACKET && isStableZero(scores) -> {
                if (preambleIndex != 1) countPreambleMiss() else null
            }
            else -> countPreambleMiss()
        }
    }

    private fun countPreambleMiss(): Result? {
        preambleMisses += 1
        if (preambleMisses <= MAX_PREAMBLE_MISSES) return null
        resetInternal()
        return Result(failure = FailureReason.PreambleLost)
    }

    private fun isStableZero(scores: FloatArray): Boolean {
        var sum = 0f
        var max = Float.NEGATIVE_INFINITY
        for (score in scores) {
            sum += score
            if (score > max) max = score
        }
        return max <= PREAMBLE_ZERO_MAX_SCORE && sum / BIT_COUNT <= PREAMBLE_ZERO_AVG_SCORE
    }

    private fun observeExpectedPreambleEdge(
        scores: FloatArray,
        previousPattern: String,
        expectedPattern: String,
        timestampNs: Long,
    ) {
        val previousScores = lastPreambleScores
        if (previousScores == null) {
            lastPreambleScores = scores.copyOf()
            return
        }

        var changedScore = 0f
        var changedCount = 0
        var unexpectedScore = 0f
        for (index in 0 until BIT_COUNT) {
            val previousBit = previousPattern[index]
            val expectedBit = expectedPattern[index]
            val delta = scores[index] - previousScores[index]
            previousScores[index] = scores[index]
            if (previousBit != expectedBit) {
                changedCount += 1
                changedScore += if (expectedBit == '1') delta.coerceAtLeast(0f) else (-delta).coerceAtLeast(0f)
            } else {
                unexpectedScore += kotlin.math.abs(delta)
            }
        }
        if (changedCount == 0) return

        val edgeScore = changedScore / changedCount - unexpectedScore * PREAMBLE_UNEXPECTED_EDGE_PENALTY
        if (edgeScore >= PREAMBLE_EDGE_SCORE_THRESHOLD && edgeScore > pendingPreambleEdgeScore) {
            pendingPreambleEdgeAtNs = timestampNs
            pendingPreambleEdgeScore = edgeScore
        }
    }

    private fun consumePreambleEdgeTimestampOr(fallbackNs: Long): Long {
        val timestampNs = if (pendingPreambleEdgeAtNs != 0L) pendingPreambleEdgeAtNs else fallbackNs
        pendingPreambleEdgeAtNs = 0L
        pendingPreambleEdgeScore = 0f
        return timestampNs
    }

    private fun matchesSoft(scores: FloatArray, expected: String): Boolean {
        var wrong = 0
        for (index in 0 until BIT_COUNT) {
            val score = scores[index]
            val ok = if (expected[index] == '1') {
                score >= PREAMBLE_ON_MATCH_SCORE
            } else {
                score <= PREAMBLE_OFF_MATCH_SCORE
            }
            if (!ok) {
                wrong += 1
                if (wrong > MAX_PREAMBLE_HAMMING_DISTANCE) return false
            }
        }
        return true
    }

    private fun stabilizeMeasuredPeriod(measuredPeriodNs: Long): Long {
        val clamped = measuredPeriodNs.coerceIn(MIN_PERIOD_NS, MAX_PERIOD_NS)
        var best = ALLOWED_PERIODS_NS[0]
        var bestDistance = kotlin.math.abs(clamped - best)
        for (index in 1 until ALLOWED_PERIODS_NS.size) {
            val candidate = ALLOWED_PERIODS_NS[index]
            val distance = kotlin.math.abs(clamped - candidate)
            if (distance < bestDistance) {
                best = candidate
                bestDistance = distance
            }
        }
        return best
    }

    private fun estimatePreamblePeriod(): Long {
        lastPreambleEstimateMode = "allowed_default"
        lastPreambleFirstIntervalNs = 0L
        lastPreambleSecondIntervalNs = 0L
        if (preambleIntervalCount == 0) return DEFAULT_PERIOD_NS
        lastPreambleFirstIntervalNs = preambleIntervalsNs[0]
        if (preambleIntervalCount > 1) {
            lastPreambleSecondIntervalNs = preambleIntervalsNs[1]
        }

        var bestPeriod = ALLOWED_PERIODS_NS[0]
        var bestError = Long.MAX_VALUE
        for (candidate in ALLOWED_PERIODS_NS) {
            var error = 0L
            for (index in 0 until preambleIntervalCount) {
                error += kotlin.math.abs(preambleIntervalsNs[index] - candidate)
            }
            if (error < bestError) {
                bestError = error
                bestPeriod = candidate
            }
        }
        lastPreambleEstimateMode = "allowed_${1_000_000_000L / bestPeriod}hz"
        return bestPeriod
    }

    private fun correctedFinalPreambleAt(
        observedFinalAtNs: Long,
        previousPreambleAtNs: Long,
        lastIntervalNs: Long,
    ): Long {
        if (previousPreambleAtNs == 0L || lastIntervalNs == 0L) return observedFinalAtNs
        val expectedFinalAtNs = previousPreambleAtNs + periodNs
        val observedError = kotlin.math.abs(observedFinalAtNs - expectedFinalAtNs)
        return if (observedError > periodNs * PREAMBLE_ANCHOR_ERROR_PERCENT / 100L) {
            expectedFinalAtNs
        } else {
            observedFinalAtNs
        }
    }

    private fun acceptActive(scores: FloatArray?, timestampNs: Long): Result? {
        if (timestampNs < nextEmitAtNs) {
            if (scores != null) sampleActiveSymbol(scores, timestampNs)
            return null
        }

        val periodsElapsed = ((timestampNs - nextEmitAtNs) / periodNs).coerceAtLeast(0L)
        if (periodsElapsed > MAX_ERASURE_BURST) {
            resetInternal()
            return Result(failure = FailureReason.SymbolSkipped)
        }
        val skippedSymbols = periodsElapsed.toInt().coerceAtMost(MAX_ERASURE_BURST)
        val rejectReason = when {
            periodsElapsed > 0L -> "late_$periodsElapsed"
            else -> symbolSampleRejectReason()
        }
        val packet = if (rejectReason == null) symbolPacket() else null
        val result = Result(
            bits = packet?.bits,
            bitLlrs = packet?.bitLlrs,
            debugInfo = packet?.debugInfo?.copy(
                emittedSymbol = emittedSymbols,
                decision = "packet",
                rejectReason = "",
                symbolStartNs = symbolStartNs,
                symbolEmitNs = nextEmitAtNs,
            ) ?: DebugInfo(
                periodNs = periodNs,
                periodsElapsed = periodsElapsed,
                emittedSymbol = emittedSymbols,
                decision = "erasure",
                rejectReason = rejectReason ?: "unknown",
                symbolStartNs = symbolStartNs,
                symbolEmitNs = nextEmitAtNs,
                sampleCount = symbolSampleCount,
                sampleWeightSum = symbolWeightSum,
                averageSamplePhase = averageSamplePhase(),
            ),
            packetKind = PacketKind.Payload,
            payloadErasures = if (packet == null) skippedSymbols + 1 else 0,
        )
        emittedSymbols += skippedSymbols + 1L
        updateActiveBoundariesFromClock()
        clearSymbolSamples()
        if (scores != null) sampleActiveSymbol(scores, timestampNs)
        return result
    }

    private fun sampleActiveSymbol(scores: FloatArray, timestampNs: Long) {
        val elapsed = timestampNs - symbolStartNs
        if (elapsed < 0L || elapsed >= periodNs) return
        val phaseInSymbol = elapsed.toFloat() / periodNs.toFloat()
        val sampleWeight = symbolSampleWeight(phaseInSymbol)
        if (sampleWeight <= 0f) return

        for (index in 0 until BIT_COUNT) {
            val score = scores[index]
            symbolScoreSums[index] += score * sampleWeight
            if (score > symbolScoreMax[index]) {
                symbolScoreMax[index] = score
            }
        }
        symbolWeightSum += sampleWeight
        symbolPhaseWeightSum += phaseInSymbol * sampleWeight
        symbolSampleCount += 1
    }

    private fun averageSamplePhase(): Float {
        return if (symbolWeightSum > 0f) symbolPhaseWeightSum / symbolWeightSum else 0f
    }

    private fun symbolSampleRejectReason(): String? {
        if (symbolSampleCount <= 0) return "no_samples"
        if (symbolWeightSum <= 0f) return "zero_weight"
        return null
    }

    private fun updateActiveBoundariesFromClock() {
        symbolStartNs = clockStartNs + emittedSymbols * periodNs
        nextEmitAtNs = clockStartNs + (emittedSymbols + 1L) * periodNs
    }

    private fun symbolSampleWeight(phaseInSymbol: Float): Float {
        val edgeLimit = sampleEdgeLimit()
        val distanceFromCenter = kotlin.math.abs(phaseInSymbol - 0.5f)
        if (distanceFromCenter >= edgeLimit) return 0f
        val normalized = 1f - distanceFromCenter / edgeLimit
        return normalized * normalized
    }

    private fun sampleEdgeLimit(): Float {
        return when {
            periodNs <= FAST_PERIOD_NS -> FAST_SYMBOL_EDGE_LIMIT
            periodNs <= MEDIUM_PERIOD_NS -> MEDIUM_SYMBOL_EDGE_LIMIT
            else -> SLOW_SYMBOL_EDGE_LIMIT
        }
    }

    private fun symbolPacket(): SymbolPacket {
        val bitLlrs = FloatArray(BIT_COUNT)
        val reliabilities = FloatArray(BIT_COUNT)
        val averages = FloatArray(BIT_COUNT)
        val peaks = symbolScoreMax.copyOf()
        val onThreshold = onThreshold()
        val bits = buildString(BIT_COUNT) {
            for (index in 0 until BIT_COUNT) {
                val average = symbolScoreSums[index] / symbolWeightSum
                val peak = symbolScoreMax[index]
                averages[index] = average
                val rawLlr = scoreToRawLlr(average = average, peak = peak, onThreshold = onThreshold)
                bitLlrs[index] = rawLlr
                val bit = debugBit(rawLlr)
                reliabilities[index] = llrReliability(rawLlr)
                append(bit)
            }
        }
        return SymbolPacket(
            bits = bits,
            bitLlrs = bitLlrs,
            debugInfo = DebugInfo(
                periodNs = periodNs,
                decision = "candidate",
                symbolStartNs = symbolStartNs,
                symbolEmitNs = nextEmitAtNs,
                sampleCount = symbolSampleCount,
                sampleWeightSum = symbolWeightSum,
                averageSamplePhase = averageSamplePhase(),
                averages = averages,
                peaks = peaks,
                reliabilities = reliabilities.copyOf(),
                onThreshold = onThreshold,
                offThreshold = SYMBOL_OFF_THRESHOLD,
            ),
        )
    }

    private fun onThreshold(): Float {
        return if (periodNs <= FAST_PERIOD_NS) FAST_SYMBOL_ON_THRESHOLD else SYMBOL_ON_THRESHOLD
    }

    private fun scoreToRawLlr(average: Float, peak: Float, onThreshold: Float): Float {
        val midpoint = (SYMBOL_OFF_THRESHOLD + onThreshold) * 0.5f
        val scale = (onThreshold - SYMBOL_OFF_THRESHOLD).coerceAtLeast(0.05f)
        val averageLlr = (midpoint - average) / scale * SCORE_LLR_SCALE
        val peakBoost = if (
            symbolSampleCount >= MIN_PEAK_BOOST_SAMPLES &&
            symbolWeightSum >= MIN_PEAK_BOOST_WEIGHT &&
            peak > SYMBOL_STRONG_ON_THRESHOLD &&
            average > midpoint
        ) {
            -((peak - SYMBOL_STRONG_ON_THRESHOLD) / SCORE_PEAK_SCALE).coerceIn(0f, SCORE_PEAK_LLR_BOOST)
        } else {
            0f
        }
        val sampleConfidence = when {
            symbolSampleCount <= 1 -> SINGLE_SAMPLE_LLR_SCALE
            symbolSampleCount == 2 -> DOUBLE_SAMPLE_LLR_SCALE
            else -> 1f
        }
        return ((averageLlr + peakBoost) * sampleConfidence).coerceIn(-SCORE_LLR_CLAMP, SCORE_LLR_CLAMP)
    }

    private fun debugBit(rawLlr: Float): Char {
        return when {
            rawLlr > DEBUG_BIT_ERASURE_LLR -> '0'
            rawLlr < -DEBUG_BIT_ERASURE_LLR -> '1'
            else -> '?'
        }
    }

    private fun llrReliability(rawLlr: Float): Float {
        return (kotlin.math.abs(rawLlr) / SCORE_LLR_CLAMP).coerceIn(0f, 1f)
    }

    private fun clearSymbolSamples() {
        java.util.Arrays.fill(symbolScoreSums, 0f)
        java.util.Arrays.fill(symbolScoreMax, Float.NEGATIVE_INFINITY)
        symbolWeightSum = 0f
        symbolPhaseWeightSum = 0f
        symbolSampleCount = 0
    }

    private fun resetInternal() {
        phase = Phase.WaitingZeros
        zeroFrames = 0
        resetPreamble()
        periodNs = DEFAULT_PERIOD_NS
        clockStartNs = 0L
        symbolStartNs = 0L
        nextEmitAtNs = 0L
        emittedSymbols = 0L
        clearSymbolSamples()
        bitStates = null
    }

    private fun resetPreamble() {
        preambleIndex = 1
        preambleStarted = false
        lastPreambleAtNs = 0L
        java.util.Arrays.fill(preambleIntervalsNs, 0L)
        preambleIntervalCount = 0
        preambleMisses = 0
        lastPreambleScores = null
        pendingPreambleEdgeAtNs = 0L
        pendingPreambleEdgeScore = 0f
        lastPreambleEstimateMode = ""
        lastPreambleFirstIntervalNs = 0L
        lastPreambleSecondIntervalNs = 0L
    }

    private fun scoresToPacket(scores: FloatArray): String {
        val states = bitStates ?: BooleanArray(BIT_COUNT).also { fresh ->
            for (index in 0 until BIT_COUNT) {
                fresh[index] = scores[index] >= ON_THRESHOLD
            }
            bitStates = fresh
        }

        for (index in 0 until BIT_COUNT) {
            val score = scores[index]
            states[index] = if (states[index]) {
                score > OFF_THRESHOLD
            } else {
                score >= ON_THRESHOLD
            }
        }

        return buildString(BIT_COUNT) {
            states.forEach { append(if (it) '1' else '0') }
        }
    }

    private data class SymbolPacket(
        val bits: String,
        val bitLlrs: FloatArray,
        val debugInfo: DebugInfo,
    )

    private companion object {
        const val BIT_COUNT = 5
        const val REQUIRED_ZERO_FRAMES = 3
        const val MAX_PREAMBLE_HAMMING_DISTANCE = 1
        const val MAX_PREAMBLE_MISSES = 4
        const val ZERO_PACKET = "00000"
        val PREAMBLE = arrayOf("00000", "01010", "10101", "11111")
        const val ON_THRESHOLD = 1.0f
        const val OFF_THRESHOLD = 0.6f
        const val PREAMBLE_ZERO_MAX_SCORE = 0.74f
        const val PREAMBLE_ZERO_AVG_SCORE = 0.46f
        const val PREAMBLE_ON_MATCH_SCORE = 0.82f
        const val PREAMBLE_OFF_MATCH_SCORE = 0.82f
        const val PREAMBLE_EDGE_SCORE_THRESHOLD = 0.28f
        const val PREAMBLE_UNEXPECTED_EDGE_PENALTY = 0.18f
        const val SYMBOL_ON_THRESHOLD = 0.94f
        const val FAST_SYMBOL_ON_THRESHOLD = 1.05f
        const val SYMBOL_STRONG_ON_THRESHOLD = 1.12f
        const val SYMBOL_OFF_THRESHOLD = 0.56f
        const val SCORE_LLR_SCALE = 3.0f
        const val SCORE_LLR_CLAMP = 3.0f
        const val SCORE_PEAK_SCALE = 0.30f
        const val SCORE_PEAK_LLR_BOOST = 0.8f
        const val MIN_PEAK_BOOST_SAMPLES = 2
        const val MIN_PEAK_BOOST_WEIGHT = 0.45f
        const val SINGLE_SAMPLE_LLR_SCALE = 0.68f
        const val DOUBLE_SAMPLE_LLR_SCALE = 0.88f
        const val DEBUG_BIT_ERASURE_LLR = 0.30f
        const val FAST_PERIOD_NS = 135_000_000L
        const val PREAMBLE_ANCHOR_ERROR_PERCENT = 18L
        const val MEDIUM_PERIOD_NS = 180_000_000L
        val ALLOWED_PERIODS_NS = longArrayOf(500_000_000L, 250_000_000L, 125_000_000L)
        const val FAST_SYMBOL_EDGE_LIMIT = 0.38f
        const val MEDIUM_SYMBOL_EDGE_LIMIT = 0.42f
        const val SLOW_SYMBOL_EDGE_LIMIT = 0.46f
        const val DEFAULT_PERIOD_NS = 125_000_000L
        const val MIN_PERIOD_NS = 100_000_000L
        const val MAX_PERIOD_NS = 650_000_000L
        const val MAX_ERASURE_BURST = 24
    }
}
