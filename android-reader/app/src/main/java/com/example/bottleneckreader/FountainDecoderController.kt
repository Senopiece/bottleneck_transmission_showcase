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
    private var lowProgressPackets = 0
    private var bestProgress = 0f
    private var saturatedPumpTicks = 0

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
        val snapshot = decoder.addPacket(packetIndex = packetIndex, bits = bits, bitLlrs = bitLlrs)
        val debugLine = if (Diagnostics.enabled) buildDebugLine(packetIndex, bits, snapshot) else ""

        if (!observedPayload) {
            lowProgressPackets = (lowProgressPackets - 1).coerceAtLeast(0)
        } else if (snapshot.progress + PROGRESS_BACKTRACK_TOLERANCE < bestProgress) {
            lowProgressPackets += 1
        } else {
            lowProgressPackets = 0
        }
        if (snapshot.progress > bestProgress) bestProgress = snapshot.progress

        return evaluateSnapshot(snapshot, debugLine, allowInconsistentFail = true)
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
        val debugLine = if (Diagnostics.enabled) buildPumpDebugLine(snapshot) else ""
        return evaluateSnapshot(snapshot, debugLine, allowInconsistentFail = false)
    }

    private fun evaluateSnapshot(
        snapshot: FountainDecoder.Snapshot,
        debugLine: String,
        allowInconsistentFail: Boolean,
    ): ProcessResult {
        if (allowInconsistentFail &&
            payloadPackets >= MIN_PACKETS_BEFORE_CONSISTENCY_FAIL &&
            lowProgressPackets >= MAX_BACKTRACK_PACKETS
        ) {
            state = State.Failed
            return ProcessResult(
                state = State.Failed,
                failureReason = "Decode failed: stream became inconsistent",
                debug = appendFailureDebug(debugLine, "inconsistent"),
            )
        }
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
            "backtrack=$lowProgressPackets",
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
        lowProgressPackets = 0
        bestProgress = 0f
        saturatedPumpTicks = 0
    }

    companion object {
        private const val MIN_PACKETS_BEFORE_CONSISTENCY_FAIL = 12
        private const val MAX_BACKTRACK_PACKETS = 10
        private const val PROGRESS_BACKTRACK_TOLERANCE = 0.18f
        private const val BP_ITERATIONS_PER_PUMP = 1
        private const val MAX_SATURATED_PUMP_TICKS = 240
    }
}
