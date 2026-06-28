package com.example.bottleneckreader

import java.util.Locale

class FountainDecoderController {
    enum class State {
        WaitingPreamble,
        Decoding,
        Complete,
        Failed,
    }

    data class ProcessResult(
        val state: State,
        val messageBits: BooleanArray? = null,
        val confidence: FloatArray? = null,
        val progress: Float = 0f,
        val measurements: Int = 0,
        val failureReason: String? = null,
        val debug: String = "",
    )

    private val decoder = FountainDecoder()
    private var state = State.WaitingPreamble
    private var payloadPackets = 0
    private var observedPayloadPackets = 0
    private var bestProgress = 0f
    private var saturatedPumpTicks = 0
    private var frozenPosterior: FloatArray? = null
    private var frozenProgress = 0f
    private val predictiveBadWindow = FloatArray(PREDICTIVE_BAD_WINDOW_SIZE)
    private var predictiveBadWindowIndex = 0
    private var predictiveBadWindowCount = 0
    private var consecutivePredictiveHits = 0
    private var lastPredictiveScore = 0f
    private var lastPredictiveBad = 0f
    private var lastPredictiveBadWindow = 0f
    private var lastPredictiveSkipped = false

    @Synchronized
    fun startPreamble() {
        reset()
        state = State.Decoding
    }

    @Synchronized
    fun processPayload(bits: String?, bitLlrs: FloatArray?): ProcessResult {
        if (state == State.Failed || state == State.Complete) {
            return ProcessResult(state = state)
        }
        if (state == State.WaitingPreamble) {
            return ProcessResult(state = state)
        }

        val packetIndex = payloadPackets
        payloadPackets += 1
        val observedPayload = bitLlrs != null && bitLlrs.size == FountainDecoder.PACKET_BITS
        if (observedPayload) observedPayloadPackets += 1
        val useBitLlrs = if (observedPayload && predictiveGuardRejects(packetIndex, bitLlrs)) {
            if (consecutivePredictiveHits >= PREDICTIVE_FAIL_CONSECUTIVE_HITS) {
                state = State.Failed
                return ProcessResult(
                    state = State.Failed,
                    failureReason = "Decode failed: stream became inconsistent",
                    debug = buildPredictiveFailureDebug(packetIndex, bits),
                )
            }
            null
        } else {
            bitLlrs
        }
        val snapshot = decoder.addPacket(packetIndex = packetIndex, bits = bits, bitLlrs = useBitLlrs)
        val debugLine = if (Diagnostics.enabled) buildDebugLine(packetIndex, bits, snapshot) else ""

        if (snapshot.progress > bestProgress) bestProgress = snapshot.progress
        updateFrozenPosterior(snapshot)

        return evaluateSnapshot(snapshot, debugLine)
    }

    @Synchronized
    fun pump(): ProcessResult {
        if (state == State.Failed || state == State.Complete) {
            return ProcessResult(state = state)
        }
        if (state == State.WaitingPreamble) {
            return ProcessResult(state = state)
        }

        val snapshot = decoder.pump(BP_ITERATIONS_PER_PUMP)
        if (snapshot.progress > bestProgress) bestProgress = snapshot.progress
        updateFrozenPosterior(snapshot)
        val debugLine = if (Diagnostics.enabled) buildPumpDebugLine(snapshot) else ""
        return evaluateSnapshot(snapshot, debugLine)
    }

    private fun predictiveGuardRejects(packetIndex: Int, bitLlrs: FloatArray?): Boolean {
        lastPredictiveSkipped = false
        val reference = frozenPosterior
        if (reference == null || bitLlrs == null || frozenProgress < PREDICTIVE_MIN_PROGRESS) {
            lastPredictiveScore = 0f
            lastPredictiveBad = 0f
            lastPredictiveBadWindow = predictiveBadWindowSum()
            consecutivePredictiveHits = 0
            return false
        }
        val packetLogScore = decoder.predictivePacketLogScore(
            packetIndex = packetIndex,
            bitLlrs = bitLlrs,
            referencePosterior = reference,
        )
        lastPredictiveScore = packetLogScore
        lastPredictiveBad = (-packetLogScore).coerceAtLeast(0f)
        pushPredictiveBadScore(lastPredictiveBad)
        lastPredictiveBadWindow = predictiveBadWindowSum()
        val hit = lastPredictiveBad >= PREDICTIVE_PACKET_BAD_MIN &&
            lastPredictiveBadWindow >= PREDICTIVE_BAD_WINDOW_THRESHOLD
        if (hit) {
            consecutivePredictiveHits += 1
            lastPredictiveSkipped = true
        } else {
            consecutivePredictiveHits = 0
        }
        return hit
    }

    private fun updateFrozenPosterior(snapshot: FountainDecoder.Snapshot) {
        if (snapshot.progress >= PREDICTIVE_MIN_PROGRESS && snapshot.progress >= frozenProgress) {
            frozenPosterior = decoder.copyPosterior()
            frozenProgress = snapshot.progress
        }
    }

    private fun pushPredictiveBadScore(value: Float) {
        predictiveBadWindow[predictiveBadWindowIndex] = value
        predictiveBadWindowIndex = (predictiveBadWindowIndex + 1) % predictiveBadWindow.size
        if (predictiveBadWindowCount < predictiveBadWindow.size) {
            predictiveBadWindowCount += 1
        }
    }

    private fun predictiveBadWindowSum(): Float {
        var sum = 0f
        for (index in 0 until predictiveBadWindowCount) {
            sum += predictiveBadWindow[index]
        }
        return sum
    }

    private fun evaluateSnapshot(
        snapshot: FountainDecoder.Snapshot,
        debugLine: String,
    ): ProcessResult {
        if (snapshot.readyToFinalize && snapshot.parityViolations != 0) {
            state = State.Failed
            return ProcessResult(
                state = State.Failed,
                failureReason = "Decode failed: parity check failed",
                debug = appendFailureDebug(debugLine, "parity_failed"),
            )
        }
        if (snapshot.totalMeasurements >= FountainDecoder.MAX_MEASUREMENTS && !snapshot.readyToFinalize) {
            saturatedPumpTicks += 1
        } else {
            saturatedPumpTicks = 0
        }
        if (saturatedPumpTicks >= MAX_SATURATED_PUMP_TICKS && !snapshot.complete) {
            state = State.Failed
            return ProcessResult(
                state = State.Failed,
                failureReason = "Decode failed: confidence did not converge",
                debug = appendFailureDebug(debugLine, "no_converge"),
            )
        }
        state = if (snapshot.complete) State.Complete else State.Decoding
        return ProcessResult(
            state = state,
            messageBits = snapshot.bits,
            confidence = snapshot.certainties,
            progress = snapshot.progress,
            measurements = snapshot.measurements,
            debug = debugLine,
        )
    }

    private fun buildDebugLine(packetIndex: Int, bits: String?, snapshot: FountainDecoder.Snapshot): String {
        val packet = snapshot.packetDebug
        return listOf(
            "payload=$payloadPackets",
            "observed=$observedPayloadPackets",
            "packetIndex=$packetIndex",
            "raw=${bits ?: "erasure"}",
            "llr=${formatFloatArray(packet?.measurementLlrs)}",
            "added=${packet?.addedFactors ?: 0}",
            "skipped=${packet?.skippedFactors ?: 0}",
            "deg=${formatDegreeHistogram(packet?.degreeHistogram)}",
            "measurements=${snapshot.measurements}",
            "totalMeasurements=${snapshot.totalMeasurements}",
            "progress=${fmt(snapshot.progress)}",
            "best=${fmt(bestProgress)}",
            "frozen=${fmt(frozenProgress)}",
            "predLog=${fmt(lastPredictiveScore)}",
            "predBad=${fmt(lastPredictiveBad)}",
            "badwin4=${fmt(lastPredictiveBadWindow)}",
            "predHits=$consecutivePredictiveHits",
            "predSkip=$lastPredictiveSkipped",
            "expectedErrors=${fmt(snapshot.expectedErrors)}",
            "ready=${snapshot.readyToFinalize}",
            "minLlr=${fmt(snapshot.minAbsLlr)}",
            "avgLlr=${fmt(snapshot.avgAbsLlr)}",
            "maxLlr=${fmt(snapshot.maxAbsLlr)}",
            "agree=${fmt(snapshot.channelAgreement)}",
            "agreeW=${fmt(snapshot.channelMatchedWeight)}/${fmt(snapshot.channelTotalWeight)}",
            "chanBad=${snapshot.channelMismatchedFactors}",
            "parityBad=${snapshot.parityViolations}",
            "complete=${snapshot.complete}",
            "hard=${snapshot.hardBits}",
        ).joinToString(separator = " ")
    }

    private fun buildPumpDebugLine(snapshot: FountainDecoder.Snapshot): String {
        return listOf(
            "pump=bp",
            "payload=$payloadPackets",
            "observed=$observedPayloadPackets",
            "measurements=${snapshot.measurements}",
            "totalMeasurements=${snapshot.totalMeasurements}",
            "progress=${fmt(snapshot.progress)}",
            "best=${fmt(bestProgress)}",
            "frozen=${fmt(frozenProgress)}",
            "predLog=${fmt(lastPredictiveScore)}",
            "predBad=${fmt(lastPredictiveBad)}",
            "badwin4=${fmt(lastPredictiveBadWindow)}",
            "predHits=$consecutivePredictiveHits",
            "expectedErrors=${fmt(snapshot.expectedErrors)}",
            "ready=${snapshot.readyToFinalize}",
            "parityBad=${snapshot.parityViolations}",
            "complete=${snapshot.complete}",
            "saturated=$saturatedPumpTicks",
            "hard=${snapshot.hardBits}",
        ).joinToString(separator = " ")
    }

    private fun appendFailureDebug(debugLine: String, failure: String): String {
        return if (debugLine.isEmpty()) "" else "$debugLine fail=$failure"
    }

    private fun buildPredictiveFailureDebug(packetIndex: Int, bits: String?): String {
        if (!Diagnostics.enabled) return ""
        return listOf(
            "payload=$payloadPackets",
            "observed=$observedPayloadPackets",
            "packetIndex=$packetIndex",
            "raw=${bits ?: "erasure"}",
            "frozen=${fmt(frozenProgress)}",
            "predLog=${fmt(lastPredictiveScore)}",
            "predBad=${fmt(lastPredictiveBad)}",
            "badwin4=${fmt(lastPredictiveBadWindow)}",
            "predHits=$consecutivePredictiveHits",
            "fail=inconsistent",
        ).joinToString(separator = " ")
    }

    private fun fmt(value: Float): String {
        return String.format(Locale.US, "%.3f", value)
    }

    private fun formatFloatArray(values: FloatArray?): String {
        if (values == null || values.isEmpty()) return "-"
        return values.joinToString(separator = "|") { fmt(it) }
    }

    private fun formatDegreeHistogram(values: IntArray?): String {
        if (values == null || values.isEmpty()) return "-"
        val parts = ArrayList<String>(values.size)
        for (degree in values.indices) {
            val count = values[degree]
            if (count > 0) parts += "$degree:$count"
        }
        return parts.joinToString(separator = "|")
    }

    @Synchronized
    fun reset() {
        decoder.reset()
        state = State.WaitingPreamble
        payloadPackets = 0
        observedPayloadPackets = 0
        bestProgress = 0f
        saturatedPumpTicks = 0
        frozenPosterior = null
        frozenProgress = 0f
        java.util.Arrays.fill(predictiveBadWindow, 0f)
        predictiveBadWindowIndex = 0
        predictiveBadWindowCount = 0
        consecutivePredictiveHits = 0
        lastPredictiveScore = 0f
        lastPredictiveBad = 0f
        lastPredictiveBadWindow = 0f
        lastPredictiveSkipped = false
    }

    companion object {
        private const val BP_ITERATIONS_PER_PUMP = 1
        private const val MAX_SATURATED_PUMP_TICKS = 240
        private const val PREDICTIVE_MIN_PROGRESS = 0.25f
        private const val PREDICTIVE_BAD_WINDOW_SIZE = 4
        private const val PREDICTIVE_BAD_WINDOW_THRESHOLD = 2.0f
        private const val PREDICTIVE_PACKET_BAD_MIN = 0.15f
        private const val PREDICTIVE_FAIL_CONSECUTIVE_HITS = 2
    }
}
