package com.example.bottleneckreader

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong

class ReaderViewModel : ViewModel() {
    private val packetIds = AtomicLong(0)
    private val messageIds = AtomicLong(0)
    private val progressFailureIds = AtomicLong(0)
    private val packetDecoder = PacketClockDecoder()
    private val fountainDecoderController = FountainDecoderController()
    private val scoreLogger = ScoreLogger()
    private var preambleProgressPackets = 0
    private var lastDebugFailureMessage: String? = null
    private var lastDebugFailureAtNs = 0L
    private var bpPumpJob: Job? = null

    private val _frame = MutableStateFlow<DetectionFrame?>(null)
    val frame: StateFlow<DetectionFrame?> = _frame.asStateFlow()

    private val _packetEvents = MutableStateFlow<List<PacketEvent>>(emptyList())
    val packetEvents: StateFlow<List<PacketEvent>> = _packetEvents.asStateFlow()

    private val _decodeProgress = MutableStateFlow(DecodeProgress())
    val decodeProgress: StateFlow<DecodeProgress> = _decodeProgress.asStateFlow()

    private val _decodedMessage = MutableStateFlow<DecodedMessage?>(null)
    val decodedMessage: StateFlow<DecodedMessage?> = _decodedMessage.asStateFlow()

    private val _resultEventId = MutableStateFlow(0L)
    val resultEventId: StateFlow<Long> = _resultEventId.asStateFlow()

    private val _failureEventId = MutableStateFlow(0L)
    val failureEventId: StateFlow<Long> = _failureEventId.asStateFlow()

    private val _failureEventPhase = MutableStateFlow(DecodePhase.Idle)
    val failureEventPhase: StateFlow<DecodePhase> = _failureEventPhase.asStateFlow()

    private val _liveMessageBits = MutableStateFlow(List(FountainDecoder.MESSAGE_BITS) { false })
    val liveMessageBits: StateFlow<List<Boolean>> = _liveMessageBits.asStateFlow()

    private val _liveBitConfidences = MutableStateFlow(List(FountainDecoder.MESSAGE_BITS) { 0f })
    val liveBitConfidences: StateFlow<List<Float>> = _liveBitConfidences.asStateFlow()

    private val _liveDecoding = MutableStateFlow(false)
    val liveDecoding: StateFlow<Boolean> = _liveDecoding.asStateFlow()

    private val _decoderTiming = MutableStateFlow(DecoderTimingWindow())
    val decoderTiming: StateFlow<DecoderTimingWindow> = _decoderTiming.asStateFlow()

    private val _problem = MutableStateFlow<CameraProblem?>(null)
    val problem: StateFlow<CameraProblem?> = _problem.asStateFlow()

    private val _restartToken = MutableStateFlow(0)
    val restartToken: StateFlow<Int> = _restartToken.asStateFlow()

    @Synchronized
    fun onReaderEvent(event: ReaderEvent) {
        when (event) {
            is ReaderEvent.Detection -> onDetection(event.frame)
            is ReaderEvent.DecoderTiming -> onDecoderTiming(event.elapsedMs)
            is ReaderEvent.CameraIssue -> {
                _problem.value = event.problem
                resetPacketDecoder(System.nanoTime())
            }
        }
    }

    fun notifyResumed(reason: String) {
        logDebug("Resumed: $reason")
    }

    fun resumeCamera(reason: String) {
        _restartToken.update { it + 1 }
        logDebug("Resumed: $reason")
    }

    @Synchronized
    fun markStreamInterrupted() {
        resetPacketDecoder(System.nanoTime())
    }

    @Synchronized
    fun retryCamera() {
        _problem.value = null
        _restartToken.update { it + 1 }
    }

    fun dismissProblem() {
        _problem.value = null
    }

    @Synchronized
    fun closeDecodedMessage() {
        _decodedMessage.value = null
        _restartToken.update { it + 1 }
    }

    @Synchronized
    fun stopDecoding() {
        if (!_decodeProgress.value.visible && !_decodeProgress.value.failed) return
        preambleProgressPackets = 0
        _decodedMessage.value = null
        stopBpPump()
        hideDecodeProgressThenReset()
        packetDecoder.finishMessage()
        fountainDecoderController.reset()
        resetLiveDecoderStateAfterExit()
    }

    private fun onDetection(frame: DetectionFrame) {
        if (_decodeProgress.value.failed) return

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
            _decodedMessage.value = null
            resetLiveDecoderState()
            stopBpPump()
            hideDecodeProgressThenReset()
            fountainDecoderController.reset()
        } else {
            stopBpPump()
            fountainDecoderController.reset()
            val result = event
            onPacketResult(timestampNs, result)
            notifyDebugDecodeFailure(result.failure, timestampNs)
        }
    }

    private fun onPacketResult(timestampNs: Long, result: PacketClockDecoder.Result) {
        if (_decodeProgress.value.failed) return

        val failure = result.failure
        if (failure != null) {
            fountainDecoderController.reset()
            failProgress(failure.message)
            return
        }

        when (result.packetKind) {
            PacketClockDecoder.PacketKind.Preamble -> onPreamblePacket(timestampNs, result.bits, result.debugInfo)
            PacketClockDecoder.PacketKind.Payload -> {
                val erasures = result.payloadErasures
                if (erasures > 0) {
                    repeat(erasures) {
                        onPayloadPacket(timestampNs, bits = null, bitLlrs = null, clockDebug = result.debugInfo)
                    }
                } else {
                    onPayloadPacket(timestampNs, result.bits, result.bitLlrs, result.debugInfo)
                }
            }
            null -> Unit
        }
    }

    private fun resetLiveDecoderState() {
        _liveBitConfidences.value = List(FountainDecoder.MESSAGE_BITS) { 0f }
        _liveMessageBits.value = List(FountainDecoder.MESSAGE_BITS) { false }
        _liveDecoding.value = false
    }

    private fun onPreamblePacket(
        timestampNs: Long,
        bits: String?,
        clockDebug: PacketClockDecoder.DebugInfo?,
    ) {
        if (bits == null || _decodeProgress.value.failed) return
        appendPacketEvent(timestampNs, bits)
        if (preambleProgressPackets == 0) {
            _decodedMessage.value = null
            resetLiveDecoderState()
            fountainDecoderController.startPreamble()
        }
        preambleProgressPackets = (preambleProgressPackets + 1).coerceAtMost(PREAMBLE_PROGRESS_PACKETS)
        updateDecodeProgress(
            confidenceProgress = preambleUiProgress(),
            phase = DecodePhase.Preamble,
        )
        logDebug("preamble=$bits progress=$preambleProgressPackets/$PREAMBLE_PROGRESS_PACKETS clock ${formatClockDebug(clockDebug)}")
        if (preambleProgressPackets >= PREAMBLE_PROGRESS_PACKETS) {
            startBpPump()
        }
    }

    private fun onPayloadPacket(
        timestampNs: Long,
        bits: String?,
        bitLlrs: FloatArray?,
        clockDebug: PacketClockDecoder.DebugInfo?,
    ) {
        if (_decodeProgress.value.failed) return
        appendPacketEvent(timestampNs, bits)
        if (preambleProgressPackets < PREAMBLE_PROGRESS_PACKETS) {
            failProgress("Decode failed: payload arrived before complete preamble")
            return
        }
        val result = fountainDecoderController.processPayload(bits, bitLlrs?.takeIf { it.size == LED_COUNT })
        logDebug(
            "clock ${formatClockDebug(clockDebug)} fountain " +
                result.debug.ifEmpty { "payload=${bits ?: "erasure"} measurements=${result.measurements} progress=${result.progress}" },
        )
        if (Diagnostics.enabled) {
            scoreLogger.log(timestampNs, null, null, packetDecoder.debugState, result.debug)
        }

        applyFountainResult(result)
    }

    @Synchronized
    private fun onBpPumpTick() {
        if (_decodeProgress.value.failed || preambleProgressPackets < PREAMBLE_PROGRESS_PACKETS) return
        val result = fountainDecoderController.pump()
        if (result.state == FountainDecoderController.State.WaitingPreamble) return
        logDebug(result.debug)
        if (Diagnostics.enabled && result.debug.isNotEmpty()) {
            scoreLogger.log(System.nanoTime(), null, null, packetDecoder.debugState, result.debug)
        }
        applyFountainResult(result)
    }

    private fun applyFountainResult(result: FountainDecoderController.ProcessResult) {
        val confidence = result.confidence
        val messageBits = result.messageBits
        if (result.state == FountainDecoderController.State.Failed) {
            failProgress(result.failureReason ?: "Decode failed")
            return
        }
        if (confidence != null && messageBits != null) {
            _liveBitConfidences.value = confidence.toList()
            _liveMessageBits.value = messageBits.toList()
            _liveDecoding.value = true
        }
        updateDecodeProgress(
            visible = true,
            confidenceProgress = result.progress,
            phase = DecodePhase.Decoding,
        )
        if (result.state == FountainDecoderController.State.Complete && messageBits != null) {
            _decodedMessage.value = DecodedMessage(
                id = messageIds.incrementAndGet(),
                bits = BooleanArray(messageBits.size) { index -> messageBits[index] },
            )
            _resultEventId.value = progressFailureIds.incrementAndGet()
            stopBpPump()
            packetDecoder.finishMessage()
            preambleProgressPackets = 0
            updateDecodeProgress(visible = false, confidenceProgress = 1f, phase = DecodePhase.Idle)
            resetLiveDecoderStateAfterExit()
            logDebug("fountain message decoded")
        }
    }

    private fun startBpPump() {
        if (bpPumpJob?.isActive == true) return
        bpPumpJob = viewModelScope.launch {
            while (true) {
                delay(BP_PUMP_INTERVAL_MS)
                onBpPumpTick()
            }
        }
    }

    private fun stopBpPump() {
        bpPumpJob?.cancel()
        bpPumpJob = null
    }

    private fun appendPacketEvent(timestampNs: Long, bits: String?) {
        val event = PacketEvent(
            id = packetIds.incrementAndGet(),
            timestampNs = timestampNs,
            bits = bits,
        )
        _packetEvents.update { events -> (listOf(event) + events).take(MAX_PACKET_EVENTS) }
    }

    private fun updateDecodeProgress(
        visible: Boolean = preambleProgressPackets > 0,
        confidenceProgress: Float = _decodeProgress.value.confidenceProgress,
        failed: Boolean = false,
        failureId: Long = _decodeProgress.value.failureId,
        phase: DecodePhase = _decodeProgress.value.phase,
    ) {
        _decodeProgress.value = DecodeProgress(
            confidenceProgress = confidenceProgress.coerceIn(0f, 1f),
            visible = visible,
            failed = failed,
            failureId = failureId,
            phase = phase,
        )
    }

    private fun preambleUiProgress(): Float {
        return preambleProgressPackets.toFloat() / PREAMBLE_PROGRESS_PACKETS
    }

    private fun hideDecodeProgressThenReset(failureId: Long = _decodeProgress.value.failureId) {
        val visibleProgress = _decodeProgress.value.confidenceProgress
        val wasVisible = _decodeProgress.value.visible || _decodeProgress.value.failed
        updateDecodeProgress(
            visible = false,
            confidenceProgress = visibleProgress,
            failed = false,
            failureId = failureId,
            phase = DecodePhase.Idle,
        )
        if (!wasVisible) {
            updateDecodeProgress(
                visible = false,
                confidenceProgress = 0f,
                failed = false,
                failureId = failureId,
                phase = DecodePhase.Idle,
            )
            return
        }
        viewModelScope.launch {
            delay(PROGRESS_EXIT_RESET_MS)
            if (!_decodeProgress.value.visible && _decodeProgress.value.failureId == failureId) {
                updateDecodeProgress(
                    visible = false,
                    confidenceProgress = 0f,
                    failed = false,
                    failureId = failureId,
                    phase = DecodePhase.Idle,
                )
            }
        }
    }

    private fun resetLiveDecoderStateAfterExit() {
        viewModelScope.launch {
            delay(PROGRESS_EXIT_RESET_MS)
            resetLiveDecoderState()
        }
    }

    private fun failProgress(message: String) {
        notifyDebugDecodeFailure(message, System.nanoTime())
        logDebug(message)
        packetDecoder.finishMessage()
        fountainDecoderController.reset()
        stopBpPump()
        _decodedMessage.value = null
        if (!_decodeProgress.value.visible) {
            preambleProgressPackets = 0
            hideDecodeProgressThenReset()
            resetLiveDecoderStateAfterExit()
            return
        }
        val failureId = progressFailureIds.incrementAndGet()
        val failurePhase = if (preambleProgressPackets < PREAMBLE_PROGRESS_PACKETS) DecodePhase.Preamble else DecodePhase.Decoding
        _failureEventPhase.value = failurePhase
        _failureEventId.value = failureId
        val visibleProgress = _decodeProgress.value.confidenceProgress
        updateDecodeProgress(
            visible = true,
            confidenceProgress = visibleProgress,
            failed = true,
            failureId = failureId,
            phase = failurePhase,
        )
        viewModelScope.launch {
            delay(PROGRESS_FAILURE_VISIBLE_MS)
            if (_decodeProgress.value.failureId != failureId || !_decodeProgress.value.failed) {
                return@launch
            }
            preambleProgressPackets = 0
            stopBpPump()
            hideDecodeProgressThenReset(failureId = failureId)
            resetLiveDecoderStateAfterExit()
        }
    }

    private fun notifyDebugDecodeFailure(reason: PacketClockDecoder.FailureReason?, timestampNs: Long) {
        if (reason == null) return
        notifyDebugDecodeFailure(reason.message, timestampNs)
    }

    private fun notifyDebugDecodeFailure(message: String, timestampNs: Long) {
        if (!Diagnostics.enabled) return
        if (message == lastDebugFailureMessage && timestampNs - lastDebugFailureAtNs < DEBUG_FAILURE_LOG_COOLDOWN_NS) {
            return
        }
        lastDebugFailureMessage = message
        lastDebugFailureAtNs = timestampNs
        logDebug(message)
    }

    private fun logDebug(message: String) {
        if (!Diagnostics.enabled) return
        Log.d(DEBUG_TAG, message)
    }

    private fun formatClockDebug(debug: PacketClockDecoder.DebugInfo?): String {
        if (debug == null) return "-"
        return "sym=${debug.emittedSymbol} decision=${debug.decision.ifEmpty { "-" }} " +
            "reject=${debug.rejectReason.ifEmpty { "-" }} " +
            "periodMs=${debug.periodNs / 1_000_000L} samples=${debug.sampleCount} " +
            "measuredMs=${fmt(debug.measuredPeriodNs / 1_000_000f)} " +
            "preMode=${debug.preambleEstimateMode.ifEmpty { "-" }} " +
            "preI=${fmt(debug.preambleFirstIntervalNs / 1_000_000f)}|" +
            "${fmt(debug.preambleSecondIntervalNs / 1_000_000f)} " +
            "w=${fmt(debug.sampleWeightSum)} phase=${fmt(debug.averageSamplePhase)} " +
            "avg=${formatFloatArray(debug.averages)} peak=${formatFloatArray(debug.peaks)} " +
            "rel=${formatFloatArray(debug.reliabilities)} " +
            "th=${fmt(debug.onThreshold)}/${fmt(debug.offThreshold)}"
    }

    private fun fmt(value: Float): String {
        return String.format(java.util.Locale.US, "%.3f", value)
    }

    private fun formatFloatArray(values: FloatArray): String {
        if (values.isEmpty()) return "-"
        return values.joinToString(separator = "|") { fmt(it) }
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
        const val BP_PUMP_INTERVAL_MS = 16L
        const val TIMING_WINDOW_SIZE = 90
        const val PROGRESS_FAILURE_VISIBLE_MS = 0L
        const val PROGRESS_EXIT_RESET_MS = 220L
        const val DEBUG_FAILURE_LOG_COOLDOWN_NS = 1_500_000_000L
        const val DEBUG_TAG = "ReaderDecode"
    }
}
