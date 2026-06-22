package com.example.bottleneckreader

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
    private val packetDecoder = PacketClockDecoder()
    private val scoreLogger = ScoreLogger()
    private val messagePackets = ArrayList<String>(MESSAGE_PACKET_COUNT)

    private val _frame = MutableStateFlow<DetectionFrame?>(null)
    val frame: StateFlow<DetectionFrame?> = _frame.asStateFlow()

    private val _notices = MutableStateFlow<List<ReaderNotice>>(emptyList())
    val notices: StateFlow<List<ReaderNotice>> = _notices.asStateFlow()

    private val _packetEvents = MutableStateFlow<List<PacketEvent>>(emptyList())
    val packetEvents: StateFlow<List<PacketEvent>> = _packetEvents.asStateFlow()

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
            onPacketEvent(frame.timestampNs, result.bits)
        }
    }

    private fun resetPacketDecoder(timestampNs: Long) {
        val event = packetDecoder.reset()
        if (Diagnostics.enabled) {
            scoreLogger.log(timestampNs, null, event, packetDecoder.debugState)
        }
        event?.let { result ->
            onPacketEvent(timestampNs, result.bits)
        }
    }

    private fun onPacketEvent(timestampNs: Long, bits: String?) {
        appendPacketEvent(timestampNs, bits)
        if (bits == null) {
            messagePackets.clear()
            return
        }
        if (messagePackets.size >= MESSAGE_PACKET_COUNT) return
        if (messagePackets.isEmpty()) {
            _decodedMessage.value = null
        }
        messagePackets += bits
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
        val codeword = messagePackets.joinToString(separator = "")
        if (codeword.length < CODEWORD_BITS) return
        val decoded = decodeSparseParityCodeword(codeword) ?: return
        val pixels = BooleanArray(MESSAGE_BITS) { index -> decoded[index] }
        _decodedMessage.value = DecodedMessage(
            id = messageIds.incrementAndGet(),
            bits = pixels,
        )
    }

    private fun decodeSparseParityCodeword(codeword: String): BooleanArray? {
        val bits = BooleanArray(CODEWORD_BITS) { index -> codeword[index] == '1' }
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
        const val MESSAGE_BITS = 36
        const val PARITY_BITS = 24
        const val CODEWORD_BITS = MESSAGE_BITS + PARITY_BITS
        const val MESSAGE_PACKET_COUNT = CODEWORD_BITS / LED_COUNT
        const val LDPC_MAX_ITERATIONS = 8
        const val MIN_BIT_FLIP_VOTES = 2
        const val TIMING_WINDOW_SIZE = 90
        const val NOTICE_VISIBLE_MS = 3_200L
        const val NOTICE_EXIT_MS = 420L

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
