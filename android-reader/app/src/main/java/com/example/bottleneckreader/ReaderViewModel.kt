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
    private val messageReliability = ArrayList<FloatArray?>(MESSAGE_PACKET_COUNT)
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
            messageReliability += null
        }
        updateDecodeProgress()
        decodeMessage()
        messagePackets.clear()
        messageReliability.clear()
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
            messageReliability.clear()
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
            PacketClockDecoder.PacketKind.Payload -> onPayloadPacket(timestampNs, result.bits, result.bitReliability)
            null -> Unit
        }
    }

    private fun onPreamblePacket(timestampNs: Long, bits: String?) {
        if (bits == null) return
        appendPacketEvent(timestampNs, bits)
        if (preambleProgressPackets == 0) {
            messagePackets.clear()
            messageReliability.clear()
            _decodedMessage.value = null
        }
        preambleProgressPackets = (preambleProgressPackets + 1).coerceAtMost(PREAMBLE_PROGRESS_PACKETS)
        updateDecodeProgress()
        logDebug("preamble=$bits progress=${progressCount()}/$TOTAL_PROGRESS_PACKETS")
    }

    private fun onPayloadPacket(timestampNs: Long, bits: String?, bitReliability: FloatArray?) {
        appendPacketEvent(timestampNs, bits)
        if (preambleProgressPackets < PREAMBLE_PROGRESS_PACKETS) {
            failProgress("Decode failed: payload arrived before complete preamble")
            return
        }
        if (messagePackets.size >= MESSAGE_PACKET_COUNT) return
        messagePackets += bits
        messageReliability += bitReliability?.takeIf { it.size == LED_COUNT }?.copyOf()
        updateDecodeProgress()
        logDebug("payload=${bits ?: "erasure"} progress=${progressCount()}/$TOTAL_PROGRESS_PACKETS")
        if (messagePackets.size == MESSAGE_PACKET_COUNT) {
            decodeMessage()
            messagePackets.clear()
            messageReliability.clear()
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
        val erasureBits = codeword.count { it == '?' }
        val erasurePackets = messagePackets.count { it == null }
        val reliability = flattenReliability()
        logDebug("decode codeword erasures=$erasureBits packets=$erasurePackets codeword=$codeword")
        if (erasureBits > MAX_DECODE_ERASURE_BITS || erasurePackets > MAX_DECODE_ERASURE_PACKETS) {
            failProgress("Decode failed: insufficient reliable packets")
            logDebug("message decode rejected: erasures=$erasureBits packets=$erasurePackets")
            return
        }
        val decoded = decodeSparseParityCodeword(codeword, reliability)
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

    private fun flattenReliability(): FloatArray {
        val reliability = FloatArray(CODEWORD_BITS)
        var outputIndex = 0
        for (packetIndex in 0 until MESSAGE_PACKET_COUNT) {
            val packetReliability = messageReliability.getOrNull(packetIndex)
            for (bitIndex in 0 until LED_COUNT) {
                if (outputIndex >= CODEWORD_BITS) break
                reliability[outputIndex] = packetReliability?.getOrNull(bitIndex) ?: 0f
                outputIndex += 1
            }
        }
        return reliability
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
            messageReliability.clear()
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
            messageReliability.clear()
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

    private fun decodeSparseParityCodeword(codeword: String, reliability: FloatArray): BooleanArray? {
        val bits = BooleanArray(CODEWORD_BITS)
        val known = BooleanArray(CODEWORD_BITS)
        val bitReliability = FloatArray(CODEWORD_BITS)
        for (index in 0 until CODEWORD_BITS) {
            when (codeword[index]) {
                '0' -> {
                    bits[index] = false
                    known[index] = true
                    bitReliability[index] = reliability.getOrElse(index) { DEFAULT_HARD_BIT_RELIABILITY }
                }
                '1' -> {
                    bits[index] = true
                    known[index] = true
                    bitReliability[index] = reliability.getOrElse(index) { DEFAULT_HARD_BIT_RELIABILITY }
                }
            }
        }
        val bpDecoded = decodeMinSumCodeword(codeword, bitReliability)
        if (bpDecoded != null) {
            val cost = decodedInputCost(bpDecoded, codeword, bitReliability)
            if (cost.knownFlips <= BP_MAX_ACCEPT_KNOWN_FLIPS && cost.weightedCost <= BP_MAX_ACCEPT_FLIP_COST) {
                return bpDecoded
            }
            logDebug("bp rejected flips=${cost.knownFlips} cost=${cost.weightedCost}")
        }

        solveErasures(bits, known, bitReliability)
        if (parityMissCount(bits) == 0) {
            val erasureSolvedCost = decodedInputCost(bits, codeword, bitReliability)
            if (erasureSolvedCost.knownFlips == 0) {
                logDebug("erasures solved without known flips")
                return bits
            }
        }
        if (solveErasureSystem(bits, known, bitReliability)) {
            val erasureSolvedCost = decodedInputCost(bits, codeword, bitReliability)
            if (erasureSolvedCost.knownFlips == 0) {
                logDebug("erasure system solved")
                return bits
            }
        }

        val erasureBits = codeword.count { it == '?' }
        if (erasureBits > FALLBACK_MAX_ERASURE_BITS) {
            logDebug("fallback skipped erasures=$erasureBits")
            return null
        }

        val fallbackDecoded = decodeWeightedCodeword(bits, bitReliability) ?: return null
        val fallbackCost = decodedInputCost(fallbackDecoded, codeword, bitReliability)
        return if (
            fallbackCost.knownFlips <= FALLBACK_MAX_ACCEPT_KNOWN_FLIPS &&
            fallbackCost.weightedCost <= FALLBACK_MAX_ACCEPT_FLIP_COST
        ) {
            fallbackDecoded
        } else {
            logDebug("fallback rejected flips=${fallbackCost.knownFlips} cost=${fallbackCost.weightedCost}")
            null
        }
    }

    private fun decodeMinSumCodeword(codeword: String, reliability: FloatArray): BooleanArray? {
        val channel = Array(PARITY_BITS) { FloatArray(CHECK_DEGREE) }
        val checkToVariable = Array(PARITY_BITS) { FloatArray(CHECK_DEGREE) }
        val posterior = FloatArray(CODEWORD_BITS)

        for (check in 0 until PARITY_BITS) {
            val indexes = LDPC_CHECK_INDEXES[check]
            for (edge in 0 until CHECK_DEGREE) {
                channel[check][edge] = initialLlr(codeword[indexes[edge]], reliability.getOrElse(indexes[edge]) { 0f })
            }
        }

        repeat(BP_MAX_ITERATIONS) { iteration ->
            for (check in 0 until PARITY_BITS) {
                val variableToCheck = channel[check]
                for (edge in 0 until CHECK_DEGREE) {
                    var sign = 1f
                    var minMagnitude = Float.POSITIVE_INFINITY
                    for (otherEdge in 0 until CHECK_DEGREE) {
                        if (otherEdge == edge) continue
                        val message = variableToCheck[otherEdge]
                        if (message < 0f) sign = -sign
                        val magnitude = kotlin.math.abs(message)
                        if (magnitude < minMagnitude) minMagnitude = magnitude
                    }
                    checkToVariable[check][edge] = sign * minMagnitude * BP_NORMALIZATION
                }
            }

            java.util.Arrays.fill(posterior, 0f)
            for (index in 0 until CODEWORD_BITS) {
                posterior[index] = initialLlr(codeword[index], reliability.getOrElse(index) { 0f })
            }
            for (check in 0 until PARITY_BITS) {
                val indexes = LDPC_CHECK_INDEXES[check]
                for (edge in 0 until CHECK_DEGREE) {
                    posterior[indexes[edge]] += checkToVariable[check][edge]
                }
            }

            val bits = BooleanArray(CODEWORD_BITS) { index -> posterior[index] < 0f }
            if (parityMissCount(bits) == 0) {
                logDebug("bp decoded iterations=${iteration + 1}")
                return bits
            }

            for (check in 0 until PARITY_BITS) {
                val indexes = LDPC_CHECK_INDEXES[check]
                for (edge in 0 until CHECK_DEGREE) {
                    channel[check][edge] = (
                        initialLlr(codeword[indexes[edge]], reliability.getOrElse(indexes[edge]) { 0f }) +
                            variableCheckSum(indexes[edge], check, checkToVariable)
                        ).coerceIn(-BP_MAX_LLR, BP_MAX_LLR)
                }
            }
        }

        logDebug("bp failed syndrome=${parityMissCount(BooleanArray(CODEWORD_BITS) { posterior[it] < 0f })}")
        return null
    }

    private fun initialLlr(bit: Char, reliability: Float): Float {
        val magnitude = (BP_BASE_LLR + reliability.coerceIn(0f, 1f) * BP_RELIABILITY_LLR).coerceAtMost(BP_MAX_LLR)
        return when (bit) {
            '0' -> magnitude
            '1' -> -magnitude
            else -> 0f
        }
    }

    private fun variableCheckSum(
        variableIndex: Int,
        excludedCheck: Int,
        checkToVariable: Array<FloatArray>,
    ): Float {
        var sum = 0f
        for (check in 0 until PARITY_BITS) {
            if (check == excludedCheck) continue
            val indexes = LDPC_CHECK_INDEXES[check]
            for (edge in 0 until CHECK_DEGREE) {
                if (indexes[edge] == variableIndex) {
                    sum += checkToVariable[check][edge]
                    break
                }
            }
        }
        return sum
    }

    private fun solveErasures(bits: BooleanArray, known: BooleanArray, reliability: FloatArray) {
        var changed: Boolean
        do {
            changed = false
            for (check in 0 until PARITY_BITS) {
                val indexes = LDPC_GROUPS[check] + intArrayOf(MESSAGE_BITS + check)
                var unknownIndex = -1
                var unknownCount = 0
                var parity = false
                var minKnownReliability = 1f
                for (index in indexes) {
                    if (known[index]) {
                        parity = parity != bits[index]
                        if (reliability[index] < minKnownReliability) {
                            minKnownReliability = reliability[index]
                        }
                    } else {
                        unknownIndex = index
                        unknownCount += 1
                    }
                }
                if (unknownCount == 1) {
                    bits[unknownIndex] = parity
                    known[unknownIndex] = true
                    reliability[unknownIndex] = (minKnownReliability * ERASURE_SOLVE_RELIABILITY_SCALE)
                        .coerceIn(0f, MAX_SOLVED_ERASURE_RELIABILITY)
                    changed = true
                }
            }
        } while (changed)
    }

    private fun solveErasureSystem(bits: BooleanArray, known: BooleanArray, reliability: FloatArray): Boolean {
        val unknownIndexes = IntArray(CODEWORD_BITS)
        var unknownCount = 0
        for (index in 0 until CODEWORD_BITS) {
            if (!known[index]) {
                unknownIndexes[unknownCount] = index
                unknownCount += 1
            }
        }
        if (unknownCount == 0) return parityMissCount(bits) == 0
        if (unknownCount > MAX_LINEAR_ERASURE_BITS) return false

        val unknownColumn = IntArray(CODEWORD_BITS) { -1 }
        for (column in 0 until unknownCount) {
            unknownColumn[unknownIndexes[column]] = column
        }

        val rows = LongArray(PARITY_BITS)
        var rowCount = 0
        for (check in 0 until PARITY_BITS) {
            val indexes = LDPC_CHECK_INDEXES[check]
            var mask = 0L
            var rhs = false
            for (index in indexes) {
                val column = unknownColumn[index]
                if (column >= 0) {
                    mask = mask xor (1L shl column)
                } else {
                    rhs = rhs != bits[index]
                }
            }
            if (mask != 0L) {
                rows[rowCount] = mask or (if (rhs) (1L shl unknownCount) else 0L)
                rowCount += 1
            } else if (rhs) {
                return false
            }
        }

        var rank = 0
        for (column in 0 until unknownCount) {
            var pivot = -1
            for (row in rank until rowCount) {
                if (((rows[row] ushr column) and 1L) != 0L) {
                    pivot = row
                    break
                }
            }
            if (pivot < 0) continue
            val tmp = rows[rank]
            rows[rank] = rows[pivot]
            rows[pivot] = tmp
            for (row in 0 until rowCount) {
                if (row != rank && (((rows[row] ushr column) and 1L) != 0L)) {
                    rows[row] = rows[row] xor rows[rank]
                }
            }
            rank += 1
        }

        for (row in rank until rowCount) {
            val coefficients = rows[row] and ((1L shl unknownCount) - 1L)
            val rhs = ((rows[row] ushr unknownCount) and 1L) != 0L
            if (coefficients == 0L && rhs) return false
        }
        if (rank < unknownCount) return false

        for (row in 0 until rank) {
            val coefficients = rows[row] and ((1L shl unknownCount) - 1L)
            val column = java.lang.Long.numberOfTrailingZeros(coefficients)
            val value = ((rows[row] ushr unknownCount) and 1L) != 0L
            val index = unknownIndexes[column]
            bits[index] = value
            known[index] = true
            reliability[index] = SYSTEM_SOLVED_ERASURE_RELIABILITY
        }
        return parityMissCount(bits) == 0
    }

    private fun decodeWeightedCodeword(bits: BooleanArray, reliability: FloatArray): BooleanArray? {
        val original = bits.copyOf()
        var currentCost = codewordCost(bits, original, reliability)

        repeat(LDPC_MAX_ITERATIONS) {
            if (parityMissCount(bits) == 0) return bits

            var bestIndex = -1
            var bestCost = currentCost
            for (index in 0 until CODEWORD_BITS) {
                bits[index] = !bits[index]
                val candidateCost = codewordCost(bits, original, reliability)
                bits[index] = !bits[index]
                if (candidateCost + MIN_COST_IMPROVEMENT < bestCost) {
                    bestCost = candidateCost
                    bestIndex = index
                }
            }

            if (bestIndex < 0) return null
            bits[bestIndex] = !bits[bestIndex]
            currentCost = bestCost
        }

        return if (parityMissCount(bits) == 0) bits else null
    }

    private fun codewordCost(bits: BooleanArray, original: BooleanArray, reliability: FloatArray): Float {
        var cost = parityMissCount(bits) * PARITY_MISS_COST
        for (index in 0 until CODEWORD_BITS) {
            if (bits[index] != original[index]) {
                cost += BIT_FLIP_BASE_COST + reliability[index].coerceIn(0f, 1f) * BIT_FLIP_RELIABILITY_COST
            }
        }
        return cost
    }

    private fun decodedInputCost(decoded: BooleanArray, codeword: String, reliability: FloatArray): DecodeInputCost {
        var knownFlips = 0
        var weightedCost = 0f
        for (index in 0 until CODEWORD_BITS) {
            val expected = when (codeword[index]) {
                '0' -> false
                '1' -> true
                else -> continue
            }
            if (decoded[index] != expected) {
                knownFlips += 1
                weightedCost += BIT_FLIP_BASE_COST + reliability[index].coerceIn(0f, 1f) * BIT_FLIP_RELIABILITY_COST
            }
        }
        return DecodeInputCost(knownFlips = knownFlips, weightedCost = weightedCost)
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

    private data class DecodeInputCost(
        val knownFlips: Int,
        val weightedCost: Float,
    )

    private companion object {
        const val LED_COUNT = 5
        const val MAX_PACKET_EVENTS = 3
        const val PREAMBLE_PROGRESS_PACKETS = 3
        const val MESSAGE_BITS = 36
        const val PARITY_BITS = 24
        const val CODEWORD_BITS = MESSAGE_BITS + PARITY_BITS
        const val MESSAGE_PACKET_COUNT = CODEWORD_BITS / LED_COUNT
        const val TOTAL_PROGRESS_PACKETS = PREAMBLE_PROGRESS_PACKETS + MESSAGE_PACKET_COUNT
        const val PARITY_DATA_DEGREE = 8
        const val CHECK_DEGREE = PARITY_DATA_DEGREE + 1
        const val BP_MAX_ITERATIONS = 32
        const val BP_NORMALIZATION = 0.82f
        const val BP_BASE_LLR = 0.16f
        const val BP_RELIABILITY_LLR = 2.80f
        const val BP_MAX_LLR = 6.0f
        const val MAX_DECODE_ERASURE_BITS = 20
        const val MAX_DECODE_ERASURE_PACKETS = 3
        const val BP_MAX_ACCEPT_KNOWN_FLIPS = 1
        const val BP_MAX_ACCEPT_FLIP_COST = 1.15f
        const val FALLBACK_MAX_ERASURE_BITS = 6
        const val FALLBACK_MAX_ACCEPT_KNOWN_FLIPS = 0
        const val FALLBACK_MAX_ACCEPT_FLIP_COST = 0.0f
        const val LDPC_MAX_ITERATIONS = 18
        const val PARITY_MISS_COST = 1.0f
        const val BIT_FLIP_BASE_COST = 0.06f
        const val BIT_FLIP_RELIABILITY_COST = 1.10f
        const val MIN_COST_IMPROVEMENT = 0.01f
        const val DEFAULT_HARD_BIT_RELIABILITY = 0.75f
        const val ERASURE_SOLVE_RELIABILITY_SCALE = 0.62f
        const val MAX_SOLVED_ERASURE_RELIABILITY = 0.45f
        const val MAX_LINEAR_ERASURE_BITS = 24
        const val SYSTEM_SOLVED_ERASURE_RELIABILITY = 0.35f
        const val TIMING_WINDOW_SIZE = 90
        const val NOTICE_VISIBLE_MS = 3_200L
        const val NOTICE_EXIT_MS = 420L
        const val PROGRESS_FAILURE_VISIBLE_MS = 760L
        const val DEBUG_FAILURE_NOTICE_COOLDOWN_NS = 1_500_000_000L
        const val DEBUG_TAG = "ReaderDecode"
        const val ERASURE_PACKET = "?????"

        val LDPC_GROUPS: Array<IntArray> = arrayOf(
            intArrayOf(15, 34, 8, 23, 30, 20, 18, 2),
            intArrayOf(0, 5, 26, 22, 32, 19, 7, 9),
            intArrayOf(17, 35, 33, 24, 28, 11, 10, 25),
            intArrayOf(27, 9, 14, 12, 6, 16, 13, 4),
            intArrayOf(7, 16, 3, 1, 20, 31, 24, 15),
            intArrayOf(4, 27, 21, 18, 0, 1, 29, 28),
            intArrayOf(34, 26, 12, 2, 16, 10, 33, 29),
            intArrayOf(11, 19, 13, 8, 25, 21, 14, 1),
            intArrayOf(30, 0, 23, 31, 24, 6, 32, 35),
            intArrayOf(13, 17, 28, 12, 31, 3, 22, 23),
            intArrayOf(21, 10, 9, 20, 5, 25, 29, 3),
            intArrayOf(17, 32, 20, 14, 18, 6, 33, 2),
            intArrayOf(18, 10, 22, 6, 11, 4, 26, 3),
            intArrayOf(15, 12, 5, 4, 19, 24, 35, 34),
            intArrayOf(3, 8, 35, 26, 7, 27, 25, 23),
            intArrayOf(5, 25, 31, 32, 27, 11, 1, 30),
            intArrayOf(29, 22, 0, 17, 33, 8, 30, 14),
            intArrayOf(28, 15, 7, 31, 2, 8, 11, 9),
            intArrayOf(27, 19, 34, 17, 20, 21, 0, 6),
            intArrayOf(5, 7, 30, 10, 13, 6, 1, 23),
            intArrayOf(5, 28, 16, 12, 22, 2, 21, 32),
            intArrayOf(33, 13, 26, 4, 25, 15, 24, 29),
            intArrayOf(10, 19, 32, 18, 3, 15, 35, 16),
            intArrayOf(31, 13, 21, 17, 35, 34, 9, 14),
        )

        val LDPC_CHECK_INDEXES: Array<IntArray> = Array(PARITY_BITS) { check ->
            val group = LDPC_GROUPS[check]
            IntArray(CHECK_DEGREE) { edge ->
                if (edge < PARITY_DATA_DEGREE) group[edge] else MESSAGE_BITS + check
            }
        }
    }
}
