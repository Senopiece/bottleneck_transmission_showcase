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
    private var lowAgreementPackets = 0

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

        val packetIndex = PREAMBLE_SIZE + payloadPackets
        payloadPackets += 1
        val snapshot = decoder.addPacket(packetIndex = packetIndex, bits = bits, bitLlrs = bitLlrs)
        val debugLine = if (Diagnostics.enabled) buildDebugLine(packetIndex, bits, snapshot) else ""
        val observedPayload = bitLlrs != null && bitLlrs.size == FountainDecoder.PACKET_BITS
        if (observedPayload) observedPayloadPackets += 1

        if (!observedPayload) {
            lowProgressPackets = (lowProgressPackets - 1).coerceAtLeast(0)
            lowAgreementPackets = (lowAgreementPackets - 1).coerceAtLeast(0)
        } else if (snapshot.progress + PROGRESS_BACKTRACK_TOLERANCE < bestProgress) {
            lowProgressPackets += 1
        } else {
            lowProgressPackets = 0
        }
        if (snapshot.progress > bestProgress) bestProgress = snapshot.progress
        if (observedPayload &&
            observedPayloadPackets >= MIN_OBSERVED_PACKETS_BEFORE_AGREEMENT_FAIL &&
            snapshot.channelAgreement < MIN_RUNNING_CHANNEL_AGREEMENT
        ) {
            lowAgreementPackets += 1
        } else if (observedPayload) {
            lowAgreementPackets = 0
        }

        if (payloadPackets >= MIN_PACKETS_BEFORE_CONSISTENCY_FAIL && lowProgressPackets >= MAX_BACKTRACK_PACKETS) {
            state = State.Failed
            return ProcessResult(
                state = State.Failed,
                failureReason = "Decode failed: stream became inconsistent",
                debug = appendFailureDebug(debugLine, "inconsistent"),
            )
        }
        if (lowAgreementPackets >= MAX_LOW_AGREEMENT_PACKETS) {
            state = State.Failed
            return ProcessResult(
                state = State.Failed,
                failureReason = "Decode failed: payload does not match stream timing",
                debug = appendFailureDebug(debugLine, "low_agreement"),
            )
        }
        if (snapshot.measurements >= MAX_MEASUREMENTS_WITHOUT_DECODE && !snapshot.complete) {
            state = State.Failed
            return ProcessResult(
                state = State.Failed,
                failureReason = "Decode failed: confidence did not converge",
                debug = appendFailureDebug(debugLine, "no_converge"),
            )
        }
        if (payloadPackets >= MAX_PAYLOAD_PACKETS_WITHOUT_DECODE && !snapshot.complete) {
            state = State.Failed
            return ProcessResult(
                state = State.Failed,
                failureReason = "Decode failed: too many erased or weak packets",
                debug = appendFailureDebug(debugLine, "too_many_packets"),
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
            "progress=${fmt(snapshot.progress)}",
            "best=${fmt(bestProgress)}",
            "backtrack=$lowProgressPackets",
            "lowAgree=$lowAgreementPackets",
            "expectedErrors=${fmt(snapshot.expectedErrors)}",
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
        lowAgreementPackets = 0
    }

    companion object {
        private const val PREAMBLE_SIZE = 4
        private const val MIN_PACKETS_BEFORE_CONSISTENCY_FAIL = 12
        private const val MAX_BACKTRACK_PACKETS = 10
        private const val MIN_OBSERVED_PACKETS_BEFORE_AGREEMENT_FAIL = 18
        private const val MIN_RUNNING_CHANNEL_AGREEMENT = 0.80f
        private const val MAX_LOW_AGREEMENT_PACKETS = 5
        private const val PROGRESS_BACKTRACK_TOLERANCE = 0.18f
        private const val MAX_MEASUREMENTS_WITHOUT_DECODE = 360
        private const val MAX_PAYLOAD_PACKETS_WITHOUT_DECODE = 120
    }
}
