package com.example.bottleneckreader

class PacketClockDecoder {
    data class Result(
        val bits: String? = null,
        val packetKind: PacketKind? = null,
        val failure: FailureReason? = null,
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
    private var periodSumNs = 0L
    private var periodSamples = 0
    private var preambleMisses = 0
    private var periodNs = DEFAULT_PERIOD_NS
    private var symbolStartNs = 0L
    private var nextEmitAtNs = 0L
    private var bitStates: BooleanArray? = null
    private val symbolScoreSums = FloatArray(BIT_COUNT)
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
                if (lastPreambleAtNs != 0L) {
                    periodSumNs += timestampNs - lastPreambleAtNs
                    periodSamples += 1
                }
                lastPreambleAtNs = timestampNs
                preambleIndex += 1
                val result = Result(bits = expected, packetKind = PacketKind.Preamble)

                if (preambleIndex == PREAMBLE.size) {
                    periodNs = if (periodSamples > 0) {
                        (periodSumNs / periodSamples).coerceIn(MIN_PERIOD_NS, MAX_PERIOD_NS)
                    } else {
                        DEFAULT_PERIOD_NS
                    }
                    symbolStartNs = timestampNs + periodNs
                    nextEmitAtNs = symbolStartNs + periodNs
                    clearSymbolSamples()
                    phase = Phase.Active
                }
                result
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

    private fun acceptActive(scores: FloatArray?, timestampNs: Long): Result? {
        if (timestampNs < nextEmitAtNs) {
            if (scores != null) sampleActiveSymbol(scores, timestampNs)
            return null
        }

        val periodsElapsed = ((timestampNs - nextEmitAtNs) / periodNs).coerceAtLeast(0L)
        val result = Result(
            bits = if (periodsElapsed > 0L || symbolSampleCount == 0) null else symbolPacket(),
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
        if (phaseInSymbol < SYMBOL_SAMPLE_START || phaseInSymbol > SYMBOL_SAMPLE_END) return

        for (index in 0 until BIT_COUNT) {
            symbolScoreSums[index] += scores[index]
        }
        symbolSampleCount += 1
    }

    private fun symbolPacket(): String {
        return buildString(BIT_COUNT) {
            for (index in 0 until BIT_COUNT) {
                val average = symbolScoreSums[index] / symbolSampleCount
                append(if (average >= SYMBOL_ON_THRESHOLD) '1' else '0')
            }
        }
    }

    private fun clearSymbolSamples() {
        java.util.Arrays.fill(symbolScoreSums, 0f)
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
        periodSumNs = 0L
        periodSamples = 0
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

    private companion object {
        const val BIT_COUNT = 5
        const val REQUIRED_ZERO_FRAMES = 3
        const val MAX_PREAMBLE_HAMMING_DISTANCE = 1
        const val MAX_PREAMBLE_MISSES = 4
        const val ZERO_PACKET = "00000"
        val PREAMBLE = arrayOf("00000", "01010", "10101", "11111")
        const val ON_THRESHOLD = 1.0f
        const val OFF_THRESHOLD = 0.6f
        const val SYMBOL_ON_THRESHOLD = 0.82f
        const val SYMBOL_SAMPLE_START = 0.30f
        const val SYMBOL_SAMPLE_END = 0.78f
        const val DEFAULT_PERIOD_NS = 125_000_000L
        const val MIN_PERIOD_NS = 33_000_000L
        const val MAX_PERIOD_NS = 1_000_000_000L
    }
}
