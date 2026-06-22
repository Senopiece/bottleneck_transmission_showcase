package com.example.bottleneckreader

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong

class ReaderViewModel : ViewModel() {
    private val ids = AtomicLong(0)
    private val packetIds = AtomicLong(0)
    private val messageIds = AtomicLong(0)
    private val progressFailureIds = AtomicLong(0)
    private val packetDecoder = PacketClockDecoder()
    private val scoreLogger = ScoreLogger()
    private val messagePackets = ArrayList<String?>(MESSAGE_PACKET_COUNT)
    private var preambleProgressPackets = 0
    private var lastDebugFailureMessage: String? = null
    private var lastDebugFailureAtNs = 0L

    private val _frame = MutableStateFlow<DetectionFrame?>(null)
    val frame: StateFlow<DetectionFrame?> = _frame.asStateFlow()

    private val _notices = MutableStateFlow<List<ReaderNotice>>(emptyList())
    val notices: StateFlow<List<ReaderNotice>> = _notices.asStateFlow()

    private val _packetEvents = MutableStateFlow<List<PacketEvent>>(emptyList())
    val packetEvents: StateFlow<List<PacketEvent>> = _packetEvents.asStateFlow()

    private val _decodeProgress = MutableStateFlow(DecodeProgress(requiredPackets = TOTAL_PROGRESS_PACKETS))
    val decodeProgress: StateFlow<DecodeProgress> = _decodeProgress.asStateFlow()

    private val _decodedMessage = MutableStateFlow<DecodedMessage?>(null)
    val decodedMessage: StateFlow<DecodedMessage?> = _decodedMessage.asStateFlow()

    private val _decoderTiming = MutableStateFlow(DecoderTimingWindow())
    val decoderTiming: StateFlow<DecoderTimingWindow> = _decoderTiming.asStateFlow()

    private val _problem = MutableStateFlow<CameraProblem?>(null)
    val problem: StateFlow<CameraProblem?> = _problem.asStateFlow()

    private val _restartToken = MutableStateFlow(0)
    val restartToken: StateFlow<Int> = _restartToken.asStateFlow()

    fun onReaderEvent(event: ReaderEvent) {
        when (event) {
            is ReaderEvent.Detection -> onDetection(event.frame)
            is ReaderEvent.DecoderTiming -> onDecoderTiming(event.elapsedMs)
            is ReaderEvent.Notice -> enqueueNotice(event.message)
            is ReaderEvent.CameraIssue -> {
                _problem.value = event.problem
                enqueueNotice(event.problem.title)
                resetPacketDecoder(System.nanoTime())
            }
            ReaderEvent.SlowDecoderTerminated -> {
                val problem = CameraProblem(
                    title = "Decoder too slow",
                    message = "Frame processing could not keep up with 30 fps. The camera stream was terminated.",
                )
                _problem.value = problem
                enqueueNotice("Stream terminated: decoder too slow")
                resetPacketDecoder(System.nanoTime())
            }
        }
    }

    fun notifyResumed(reason: String) {
        enqueueNotice("Resumed: $reason")
    }

    fun resumeCamera(reason: String) {
        _restartToken.update { it + 1 }
        enqueueNotice("Resumed: $reason")
    }

    fun markStreamInterrupted() {
        resetPacketDecoder(System.nanoTime())
    }

    fun retryCamera() {
        _problem.value = null
        _restartToken.update { it + 1 }
        enqueueNotice("Camera restart requested")
    }

    fun dismissProblem() {
        _problem.value = null
    }

    fun stopDecoding() {
        if (!_decodeProgress.value.visible || _decodeProgress.value.failed) return
        while (messagePackets.size < MESSAGE_PACKET_COUNT) {
            messagePackets += null
        }
        updateDecodeProgress()
        decodeMessage()
        messagePackets.clear()
        preambleProgressPackets = 0
        packetDecoder.finishMessage()
    }

    private fun enqueueNotice(message: String) {
        val id = ids.incrementAndGet()
        _notices.update { it + ReaderNotice(id = id, message = message) }
        viewModelScope.launch {
            delay(NOTICE_VISIBLE_MS)
            _notices.update { notices ->
                notices.map { if (it.id == id) it.copy(exiting = true) else it }
            }
            delay(NOTICE_EXIT_MS)
            _notices.update { notices -> notices.filterNot { it.id == id } }
        }
    }

    private fun onDetection(frame: DetectionFrame) {
        _frame.value = frame
        val scores = frame.ledScores.takeIf { it.size == LED_COUNT }
        val event = packetDecoder.accept(frame.timestampNs, scores)
        if (Diagnostics.enabled) {
            scoreLogger.log(frame.timestampNs, scores, event, packetDecoder.debugState)
        }
        event?.let { result ->
            onPacketResult(frame.timestampNs, result)
            notifyDebugDecodeFailure(result.failure, frame.timestampNs)
        }
    }

    private fun resetPacketDecoder(timestampNs: Long) {
        val event = packetDecoder.reset()
        if (Diagnostics.enabled) {
            scoreLogger.log(timestampNs, null, event, packetDecoder.debugState)
        }
        if (event == null) {
            preambleProgressPackets = 0
            messagePackets.clear()
            updateDecodeProgress(visible = false)
        } else {
            val result = event
            onPacketResult(timestampNs, result)
            notifyDebugDecodeFailure(result.failure, timestampNs)
        }
    }

    private fun onPacketResult(timestampNs: Long, result: PacketClockDecoder.Result) {
        val failure = result.failure
        if (failure != null) {
            failProgress(failure.message)
            return
        }

        when (result.packetKind) {
            PacketClockDecoder.PacketKind.Preamble -> onPreamblePacket(timestampNs, result.bits)
            PacketClockDecoder.PacketKind.Payload -> onPayloadPacket(timestampNs, result.bits)
            null -> Unit
        }
    }

    private fun onPreamblePacket(timestampNs: Long, bits: String?) {
        if (bits == null) return
        appendPacketEvent(timestampNs, bits)
        if (preambleProgressPackets == 0) {
            messagePackets.clear()
            _decodedMessage.value = null
        }
        preambleProgressPackets = (preambleProgressPackets + 1).coerceAtMost(PREAMBLE_PROGRESS_PACKETS)
        updateDecodeProgress()
        logDebug("preamble=$bits progress=${progressCount()}/$TOTAL_PROGRESS_PACKETS")
    }

    private fun onPayloadPacket(timestampNs: Long, bits: String?) {
        appendPacketEvent(timestampNs, bits)
        if (preambleProgressPackets < PREAMBLE_PROGRESS_PACKETS) {
            failProgress("Decode failed: payload arrived before complete preamble")
            return
        }
        if (messagePackets.size >= MESSAGE_PACKET_COUNT) return
        messagePackets += bits
        updateDecodeProgress()
        logDebug("payload=${bits ?: "erasure"} progress=${progressCount()}/$TOTAL_PROGRESS_PACKETS")
        if (messagePackets.size == MESSAGE_PACKET_COUNT) {
            decodeMessage()
            messagePackets.clear()
            packetDecoder.finishMessage()
        }
    }

    private fun appendPacketEvent(timestampNs: Long, bits: String?) {
        val event = PacketEvent(
            id = packetIds.incrementAndGet(),
            timestampNs = timestampNs,
            bits = bits,
        )
        _packetEvents.update { events -> (listOf(event) + events).take(MAX_PACKET_EVENTS) }
    }

    private fun decodeMessage() {
        if (messagePackets.size < MESSAGE_PACKET_COUNT) return
        val codeword = messagePackets.joinToString(separator = "") { it ?: ERASURE_PACKET }
        if (codeword.length < CODEWORD_BITS) return
        val decoded = decodeSparseParityCodeword(codeword)
        if (decoded == null) {
            failProgress("Decode failed: parity checks did not converge")
            logDebug("message decode failed: parity checks did not converge")
            return
        }
        val pixels = BooleanArray(MESSAGE_BITS) { index -> decoded[index] }
        _decodedMessage.value = DecodedMessage(
            id = messageIds.incrementAndGet(),
            bits = pixels,
        )
        logDebug("message decoded")
        preambleProgressPackets = 0
        updateDecodeProgress(visible = false)
    }

    private fun updateDecodeProgress(
        visible: Boolean = progressCount() > 0,
        failed: Boolean = false,
        failureId: Long = _decodeProgress.value.failureId,
    ) {
        _decodeProgress.value = DecodeProgress(
            receivedPackets = progressCount(),
            requiredPackets = TOTAL_PROGRESS_PACKETS,
            visible = visible,
            failed = failed,
            failureId = failureId,
        )
    }

    private fun progressCount(): Int {
        return preambleProgressPackets + messagePackets.size
    }

    private fun failProgress(message: String) {
        notifyDebugDecodeFailure(message, System.nanoTime())
        logDebug(message)
        packetDecoder.finishMessage()
        if (progressCount() == 0) {
            preambleProgressPackets = 0
            messagePackets.clear()
            updateDecodeProgress(visible = false)
            return
        }
        val failureId = progressFailureIds.incrementAndGet()
        updateDecodeProgress(
            visible = true,
            failed = true,
            failureId = failureId,
        )
        viewModelScope.launch {
            delay(PROGRESS_FAILURE_VISIBLE_MS)
            preambleProgressPackets = 0
            messagePackets.clear()
            updateDecodeProgress(visible = false, failed = false, failureId = failureId)
        }
    }

    private fun notifyDebugDecodeFailure(reason: PacketClockDecoder.FailureReason?, timestampNs: Long) {
        if (reason == null) return
        notifyDebugDecodeFailure(reason.message, timestampNs)
    }

    private fun notifyDebugDecodeFailure(message: String, timestampNs: Long) {
        if (!Diagnostics.enabled) return
        if (message == lastDebugFailureMessage && timestampNs - lastDebugFailureAtNs < DEBUG_FAILURE_NOTICE_COOLDOWN_NS) {
            return
        }
        lastDebugFailureMessage = message
        lastDebugFailureAtNs = timestampNs
        logDebug(message)
        enqueueNotice(message)
    }

    private fun logDebug(message: String) {
        if (!Diagnostics.enabled) return
        Log.d(DEBUG_TAG, message)
    }

    private fun decodeSparseParityCodeword(codeword: String): BooleanArray? {
        val bits = BooleanArray(CODEWORD_BITS)
        val known = BooleanArray(CODEWORD_BITS)
        for (index in 0 until CODEWORD_BITS) {
            when (codeword[index]) {
                '0' -> {
                    bits[index] = false
                    known[index] = true
                }
                '1' -> {
                    bits[index] = true
                    known[index] = true
                }
            }
        }
        solveErasures(bits, known)
        if (known.any { !it }) return null
        return decodeKnownCodeword(bits)
    }

    private fun solveErasures(bits: BooleanArray, known: BooleanArray) {
        var changed: Boolean
        do {
            changed = false
            for (check in 0 until PARITY_BITS) {
                val indexes = LDPC_GROUPS[check] + intArrayOf(MESSAGE_BITS + check)
                var unknownIndex = -1
                var unknownCount = 0
                var parity = false
                for (index in indexes) {
                    if (known[index]) {
                        parity = parity != bits[index]
                    } else {
                        unknownIndex = index
                        unknownCount += 1
                    }
                }
                if (unknownCount == 1) {
                    bits[unknownIndex] = parity
                    known[unknownIndex] = true
                    changed = true
                }
            }
        } while (changed)
    }

    private fun decodeKnownCodeword(bits: BooleanArray): BooleanArray? {
        val votes = IntArray(CODEWORD_BITS)

        repeat(LDPC_MAX_ITERATIONS) {
            var missCount = 0
            java.util.Arrays.fill(votes, 0)

            for (check in 0 until PARITY_BITS) {
                var parity = bits[MESSAGE_BITS + check]
                val group = LDPC_GROUPS[check]
                for (dataIndex in group) {
                    parity = parity != bits[dataIndex]
                }
                if (parity) {
                    missCount += 1
                    votes[MESSAGE_BITS + check] += 1
                    for (dataIndex in group) {
                        votes[dataIndex] += 1
                    }
                }
            }

            if (missCount == 0) return bits

            var bestIndex = -1
            var bestVotes = 0
            for (index in votes.indices) {
                if (votes[index] > bestVotes) {
                    bestVotes = votes[index]
                    bestIndex = index
                }
            }

            if (bestVotes < MIN_BIT_FLIP_VOTES || bestIndex < 0) return null
            bits[bestIndex] = !bits[bestIndex]
        }

        return if (parityMissCount(bits) == 0) bits else null
    }

    private fun parityMissCount(bits: BooleanArray): Int {
        var missCount = 0
        for (check in 0 until PARITY_BITS) {
            var parity = bits[MESSAGE_BITS + check]
            val group = LDPC_GROUPS[check]
            for (dataIndex in group) {
                parity = parity != bits[dataIndex]
            }
            if (parity) missCount += 1
        }
        return missCount
    }

    private fun onDecoderTiming(elapsedMs: Float) {
        if (!Diagnostics.enabled) return
        _decoderTiming.update { current ->
            val samples = (current.samplesMs + elapsedMs).takeLast(TIMING_WINDOW_SIZE)
            var min = Float.POSITIVE_INFINITY
            var max = Float.NEGATIVE_INFINITY
            var sum = 0f
            for (sample in samples) {
                if (sample < min) min = sample
                if (sample > max) max = sample
                sum += sample
            }
            DecoderTimingWindow(
                samplesMs = samples,
                avgMs = if (samples.isEmpty()) 0f else sum / samples.size,
                minMs = if (samples.isEmpty()) 0f else min,
                maxMs = if (samples.isEmpty()) 0f else max,
            )
        }
    }

    private companion object {
        const val LED_COUNT = 5
        const val MAX_PACKET_EVENTS = 3
        const val PREAMBLE_PROGRESS_PACKETS = 3
        const val MESSAGE_BITS = 36
        const val PARITY_BITS = 24
        const val CODEWORD_BITS = MESSAGE_BITS + PARITY_BITS
        const val MESSAGE_PACKET_COUNT = CODEWORD_BITS / LED_COUNT
        const val TOTAL_PROGRESS_PACKETS = PREAMBLE_PROGRESS_PACKETS + MESSAGE_PACKET_COUNT
        const val LDPC_MAX_ITERATIONS = 8
        const val MIN_BIT_FLIP_VOTES = 2
        const val TIMING_WINDOW_SIZE = 90
        const val NOTICE_VISIBLE_MS = 3_200L
        const val NOTICE_EXIT_MS = 420L
        const val PROGRESS_FAILURE_VISIBLE_MS = 760L
        const val DEBUG_FAILURE_NOTICE_COOLDOWN_NS = 1_500_000_000L
        const val DEBUG_TAG = "ReaderDecode"
        const val ERASURE_PACKET = "?????"

        val LDPC_GROUPS: Array<IntArray> = Array(PARITY_BITS) { index ->
            intArrayOf(
                (index * 5) % MESSAGE_BITS,
                (index * 5 + 7) % MESSAGE_BITS,
                (index * 5 + 13) % MESSAGE_BITS,
                (index * 5 + 23) % MESSAGE_BITS,
            )
        }
    }
}
