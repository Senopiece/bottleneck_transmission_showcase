package com.example.bottleneckreader

class PacketClockDecoder {
    data class Result(
        val bits: String? = null,
        val bitReliability: FloatArray? = null,
        val debugInfo: DebugInfo? = null,
        val packetKind: PacketKind? = null,
        val failure: FailureReason? = null,
    )

    data class DebugInfo(
        val periodNs: Long = 0L,
        val measuredPeriodNs: Long = 0L,
        val periodsElapsed: Long = 0L,
        val sampleCount: Int = 0,
        val sampleWeightSum: Float = 0f,
        val averages: FloatArray = FloatArray(0),
        val peaks: FloatArray = FloatArray(0),
        val reliabilities: FloatArray = FloatArray(0),
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
    private var periodNs = DEFAULT_PERIOD_NS
    private var symbolStartNs = 0L
    private var nextEmitAtNs = 0L
    private var bitStates: BooleanArray? = null
    private val symbolScoreSums = FloatArray(BIT_COUNT)
    private val symbolScoreMax = FloatArray(BIT_COUNT)
    private var symbolWeightSum = 0f
    private var symbolSampleCount = 0

    fun accept(timestampNs: Long, scores: FloatArray?): Result? {
        if (scores == null || scores.size != BIT_COUNT) {
            return acceptMissingDetection(timestampNs)
        }

        val packet = scoresToPacket(scores)
        return when (phase) {
            Phase.WaitingZeros -> acceptWaitingZeros(packet)
            Phase.WaitingPreamble -> acceptPreamble(packet, timestampNs)
            Phase.Active -> acceptActive(scores, timestampNs)
        }
    }

    fun reset(): Result? {
        val hadLock = phase != Phase.WaitingZeros
        resetInternal()
        return if (hadLock) Result(failure = FailureReason.StreamInterrupted) else null
    }

    fun finishMessage() {
        resetInternal()
    }

    private fun acceptWaitingZeros(packet: String): Result? {
        if (packet == ZERO_PACKET) {
            zeroFrames += 1
            if (zeroFrames >= REQUIRED_ZERO_FRAMES) {
                resetPreamble()
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

    private fun acceptPreamble(packet: String, timestampNs: Long): Result? {
        val expected = PREAMBLE[preambleIndex]
        val previous = PREAMBLE[preambleIndex - 1]

        return when {
            matches(packet, expected) -> {
                preambleMisses = 0
                preambleStarted = true
                var lastIntervalNs = 0L
                if (lastPreambleAtNs != 0L) {
                    lastIntervalNs = timestampNs - lastPreambleAtNs
                    if (preambleIntervalCount < preambleIntervalsNs.size) {
                        preambleIntervalsNs[preambleIntervalCount] = lastIntervalNs
                        preambleIntervalCount += 1
                    }
                }
                val previousPreambleAtNs = lastPreambleAtNs
                lastPreambleAtNs = timestampNs
                preambleIndex += 1
                var debugInfo: DebugInfo? = null

                if (preambleIndex == PREAMBLE.size) {
                    val measuredPeriodNs = estimatePreamblePeriod()
                    periodNs = stabilizeMeasuredPeriod(measuredPeriodNs)
                    val finalPreambleAtNs = correctedFinalPreambleAt(
                        observedFinalAtNs = timestampNs,
                        previousPreambleAtNs = previousPreambleAtNs,
                        lastIntervalNs = lastIntervalNs,
                    )
                    debugInfo = DebugInfo(
                        periodNs = periodNs,
                        measuredPeriodNs = measuredPeriodNs,
                    )
                    symbolStartNs = finalPreambleAtNs + periodNs
                    nextEmitAtNs = symbolStartNs + periodNs
                    clearSymbolSamples()
                    phase = Phase.Active
                }
                Result(bits = expected, debugInfo = debugInfo, packetKind = PacketKind.Preamble)
            }
            matches(packet, previous) -> null
            packet == ZERO_PACKET -> {
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

    private fun stabilizeMeasuredPeriod(measuredPeriodNs: Long): Long {
        if (measuredPeriodNs > FAST_PREAMBLE_PERIOD_NS) return measuredPeriodNs
        return (measuredPeriodNs * FAST_PREAMBLE_PERIOD_SCALE)
            .toLong()
            .coerceIn(MIN_PERIOD_NS, MAX_PERIOD_NS)
    }

    private fun estimatePreamblePeriod(): Long {
        if (preambleIntervalCount == 0) return DEFAULT_PERIOD_NS
        if (preambleIntervalCount == 1) {
            return preambleIntervalsNs[0].coerceIn(MIN_PERIOD_NS, MAX_PERIOD_NS)
        }

        val first = preambleIntervalsNs[0]
        val second = preambleIntervalsNs[1]
        val longer = kotlin.math.max(first, second)
        val shorter = kotlin.math.min(first, second)
        val measured = if (shorter * 100L < longer * SHORT_PREAMBLE_INTERVAL_PERCENT) {
            // The all-on final preamble can be observed early during camera exposure overlap.
            // In that case the longer interval is a better period estimate than the average.
            (longer * EARLY_FINAL_PERIOD_SCALE).toLong()
        } else {
            (first + second) / 2L
        }
        return measured.coerceIn(MIN_PERIOD_NS, MAX_PERIOD_NS)
    }

    private fun correctedFinalPreambleAt(
        observedFinalAtNs: Long,
        previousPreambleAtNs: Long,
        lastIntervalNs: Long,
    ): Long {
        if (previousPreambleAtNs == 0L || lastIntervalNs == 0L) return observedFinalAtNs
        return if (lastIntervalNs * 100L < periodNs * EARLY_FINAL_ANCHOR_PERCENT) {
            previousPreambleAtNs + periodNs
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
        val packet = if (periodsElapsed > 0L || symbolSampleCount == 0) null else symbolPacket()
        val result = Result(
            bits = packet?.bits,
            bitReliability = packet?.bitReliability,
            debugInfo = packet?.debugInfo ?: DebugInfo(
                periodNs = periodNs,
                periodsElapsed = periodsElapsed,
                sampleCount = symbolSampleCount,
                sampleWeightSum = symbolWeightSum,
            ),
            packetKind = PacketKind.Payload,
        )
        val emittedBoundaryNs = nextEmitAtNs
        symbolStartNs = emittedBoundaryNs
        nextEmitAtNs = emittedBoundaryNs + periodNs
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
        symbolSampleCount += 1
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
        val reliabilities = FloatArray(BIT_COUNT)
        val averages = FloatArray(BIT_COUNT)
        val peaks = symbolScoreMax.copyOf()
        val bits = buildString(BIT_COUNT) {
            for (index in 0 until BIT_COUNT) {
                val average = symbolScoreSums[index] / symbolWeightSum
                val peak = symbolScoreMax[index]
                averages[index] = average
                val bit = when {
                    average >= SYMBOL_ON_THRESHOLD -> '1'
                    peak >= SYMBOL_STRONG_ON_THRESHOLD && average >= SYMBOL_WEAK_ON_AVERAGE -> '1'
                    average <= SYMBOL_OFF_THRESHOLD && peak <= SYMBOL_OFF_PEAK_LIMIT -> '0'
                    else -> '?'
                }
                reliabilities[index] = bitReliability(bit, average, peak)
                append(bit)
            }
        }
        return SymbolPacket(
            bits = bits,
            bitReliability = reliabilities,
            debugInfo = DebugInfo(
                periodNs = periodNs,
                sampleCount = symbolSampleCount,
                sampleWeightSum = symbolWeightSum,
                averages = averages,
                peaks = peaks,
                reliabilities = reliabilities.copyOf(),
            ),
        )
    }

    private fun bitReliability(bit: Char, average: Float, peak: Float): Float {
        return when (bit) {
            '1' -> {
                val avgConfidence = ((average - SYMBOL_WEAK_ON_AVERAGE) / (SYMBOL_STRONG_ON_THRESHOLD - SYMBOL_WEAK_ON_AVERAGE))
                    .coerceIn(0f, 1f)
                val peakConfidence = ((peak - SYMBOL_ON_THRESHOLD) / (SYMBOL_STRONG_ON_THRESHOLD - SYMBOL_ON_THRESHOLD))
                    .coerceIn(0f, 1f)
                (0.35f + 0.45f * avgConfidence + 0.20f * peakConfidence).coerceIn(0f, 1f)
            }
            '0' -> {
                val avgConfidence = ((SYMBOL_OFF_THRESHOLD - average) / SYMBOL_OFF_THRESHOLD).coerceIn(0f, 1f)
                val peakConfidence = ((SYMBOL_OFF_PEAK_LIMIT - peak) / SYMBOL_OFF_PEAK_LIMIT).coerceIn(0f, 1f)
                (0.25f + 0.50f * avgConfidence + 0.25f * peakConfidence).coerceIn(0f, 1f)
            }
            else -> 0f
        }
    }

    private fun clearSymbolSamples() {
        java.util.Arrays.fill(symbolScoreSums, 0f)
        java.util.Arrays.fill(symbolScoreMax, Float.NEGATIVE_INFINITY)
        symbolWeightSum = 0f
        symbolSampleCount = 0
    }

    private fun resetInternal() {
        phase = Phase.WaitingZeros
        zeroFrames = 0
        resetPreamble()
        periodNs = DEFAULT_PERIOD_NS
        symbolStartNs = 0L
        nextEmitAtNs = 0L
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

    private fun matches(packet: String, expected: String): Boolean {
        if (packet == expected) return true
        var distance = 0
        for (index in 0 until BIT_COUNT) {
            if (packet[index] != expected[index]) {
                distance += 1
                if (distance > MAX_PREAMBLE_HAMMING_DISTANCE) return false
            }
        }
        return true
    }

    private data class SymbolPacket(
        val bits: String,
        val bitReliability: FloatArray,
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
        const val SYMBOL_ON_THRESHOLD = 0.94f
        const val SYMBOL_STRONG_ON_THRESHOLD = 1.12f
        const val SYMBOL_WEAK_ON_AVERAGE = 0.72f
        const val SYMBOL_OFF_THRESHOLD = 0.56f
        const val SYMBOL_OFF_PEAK_LIMIT = 0.86f
        const val FAST_PERIOD_NS = 135_000_000L
        const val FAST_PREAMBLE_PERIOD_NS = 145_000_000L
        const val FAST_PREAMBLE_PERIOD_SCALE = 1.07f
        const val SHORT_PREAMBLE_INTERVAL_PERCENT = 62L
        const val EARLY_FINAL_PERIOD_SCALE = 0.88f
        const val EARLY_FINAL_ANCHOR_PERCENT = 88L
        const val MEDIUM_PERIOD_NS = 180_000_000L
        const val FAST_SYMBOL_EDGE_LIMIT = 0.38f
        const val MEDIUM_SYMBOL_EDGE_LIMIT = 0.42f
        const val SLOW_SYMBOL_EDGE_LIMIT = 0.46f
        const val DEFAULT_PERIOD_NS = 125_000_000L
        const val MIN_PERIOD_NS = 33_000_000L
        const val MAX_PERIOD_NS = 1_000_000_000L
    }
}
