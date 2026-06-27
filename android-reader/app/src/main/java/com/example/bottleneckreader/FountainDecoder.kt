package com.example.bottleneckreader

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.tanh

/**
 * Incremental soft BP decoder for the rateless LDGM layer plus the finite parity precode.
 *
 * LLR convention: positive means bit 0 is more likely, negative means bit 1 is more likely.
 * The graph is bounded: new LDGM observations are accepted until MAX_MEASUREMENTS, and each
 * packet runs a fixed number of BP sweeps over that bounded graph.
 */
class FountainDecoder {
    data class PacketDebug(
        val packetIndex: Int,
        val rawBits: String?,
        val measurementLlrs: FloatArray,
        val addedFactors: Int,
        val skippedFactors: Int,
        val degreeHistogram: IntArray,
    )

    data class Snapshot(
        val bits: BooleanArray,
        val certainties: FloatArray,
        val progress: Float,
        val expectedErrors: Float,
        val complete: Boolean,
        val measurements: Int,
        val minAbsLlr: Float,
        val avgAbsLlr: Float,
        val maxAbsLlr: Float,
        val channelAgreement: Float,
        val channelMatchedWeight: Float,
        val channelTotalWeight: Float,
        val channelMismatchedFactors: Int,
        val parityViolations: Int,
        val hardBits: String,
        val packetDebug: PacketDebug?,
    )

    private data class Edge(
        val factorIndex: Int,
        val variableIndex: Int,
        var variableToFactor: Float = 0f,
        var factorToVariable: Float = 0f,
    )

    private data class Factor(
        val observedLlr: Float,
        val edgeIndexes: IntArray,
    )

    private class SeededRng(var state: Long) {
        fun next(): Long {
            state = ((state * 1103515245L + 12345L) and 0xFFFFFFFFL)
            return state
        }

        fun nextBit(): Int {
            return ((next() ushr 16) and 1L).toInt()
        }

        fun nextNeighbors(maxIndex: Int): IntArray {
            val r = (next() and 0xFFFFFFFFL).toFloat() / RNG_FLOAT_SCALE
            val degree = when {
                r < 0.20f -> 1
                r < 0.50f -> 2
                r < 0.75f -> 3
                r < 0.90f -> 4
                r < 0.97f -> 5
                else -> 6
            }
            val unique = IntArray(degree)
            var count = 0
            while (count < degree) {
                val candidate = (next() % maxIndex.toLong()).toInt()
                var exists = false
                for (index in 0 until count) {
                    if (unique[index] == candidate) {
                        exists = true
                        break
                    }
                }
                if (!exists) {
                    unique[count] = candidate
                    count += 1
                }
            }
            return unique
        }
    }

    private val factors = ArrayList<Factor>(PARITY_BITS + MAX_MEASUREMENTS)
    private val edges = ArrayList<Edge>((PARITY_BITS + MAX_MEASUREMENTS) * 6)
    private val variableEdges = Array(CODEWORD_BITS) { ArrayList<Int>(32) }
    private val posterior = FloatArray(CODEWORD_BITS)
    private val ldgmRng = SeededRng(LDGM_SEED)
    private var measurementCount = 0
    private var lastPacketDebug: PacketDebug? = null

    init {
        reset()
    }

    @Synchronized
    fun addPacket(packetIndex: Int, bits: String?, bitLlrs: FloatArray?): Snapshot {
        val llrs = if (bitLlrs != null && bitLlrs.size == PACKET_BITS) {
            FloatArray(PACKET_BITS) { index -> bitLlrs[index].coerceIn(-LLR_CLAMP, LLR_CLAMP) }
        } else {
            null
        }
        val degreeHistogram = if (Diagnostics.enabled) IntArray(MAX_DEBUG_DEGREE + 1) else EMPTY_DEBUG_HISTOGRAM
        var addedFactors = 0
        var skippedFactors = 0
        for (bitIndex in 0 until PACKET_BITS) {
            val variables = ldgmRng.nextNeighbors(CODEWORD_BITS)
            if (Diagnostics.enabled) {
                degreeHistogram[variables.size.coerceIn(0, MAX_DEBUG_DEGREE)] += 1
            }
            if (llrs != null && measurementCount < MAX_MEASUREMENTS) {
                addFactor(
                    observedLlr = llrs[bitIndex],
                    variables = variables,
                )
                measurementCount += 1
                if (Diagnostics.enabled) addedFactors += 1
            } else {
                if (Diagnostics.enabled) skippedFactors += 1
            }
        }
        lastPacketDebug = if (Diagnostics.enabled) {
            PacketDebug(
                packetIndex = packetIndex,
                rawBits = bits,
                measurementLlrs = llrs ?: FloatArray(PACKET_BITS),
                addedFactors = addedFactors,
                skippedFactors = skippedFactors,
                degreeHistogram = degreeHistogram,
            )
        } else {
            null
        }
        runIterations(BP_ITERATIONS_PER_PACKET)
        return snapshot()
    }

    @Synchronized
    fun snapshot(): Snapshot {
        updatePosterior()
        val bits = BooleanArray(MESSAGE_BITS)
        val hardCodeword = BooleanArray(CODEWORD_BITS)
        val certainties = FloatArray(MESSAGE_BITS)
        var expectedErrors = 0f
        var minAbsLlr = Float.POSITIVE_INFINITY
        var maxAbsLlr = 0f
        var sumAbsLlr = 0f
        val hardBits = if (Diagnostics.enabled) StringBuilder(MESSAGE_BITS) else null
        for (index in 0 until CODEWORD_BITS) {
            val llr = posterior[index]
            hardCodeword[index] = llr < 0f
            if (index < MESSAGE_BITS) {
                bits[index] = hardCodeword[index]
                hardBits?.append(if (bits[index]) '1' else '0')
            }
            val absLlr = abs(llr)
            if (index < MESSAGE_BITS) {
                if (absLlr < minAbsLlr) minAbsLlr = absLlr
                if (absLlr > maxAbsLlr) maxAbsLlr = absLlr
                sumAbsLlr += absLlr
                val errorProbability = 1f / (1f + exp(absLlr.coerceIn(-MAX_EXP, MAX_EXP)))
                expectedErrors += errorProbability
                certainties[index] = (1f - 2f * errorProbability).coerceIn(0f, 1f)
            }
        }
        val consistency = hardConsistency(hardCodeword)
        val progress = ((MESSAGE_BITS * 0.5f - expectedErrors) / (MESSAGE_BITS * 0.5f - TARGET_EXPECTED_ERRORS))
            .coerceIn(0f, 1f)
        return Snapshot(
            bits = bits,
            certainties = certainties,
            progress = progress,
            expectedErrors = expectedErrors,
            complete = measurementCount >= MIN_COMPLETE_MEASUREMENTS &&
                expectedErrors <= TARGET_EXPECTED_ERRORS &&
                minAbsLlr >= MIN_COMPLETE_LLR &&
                consistency.parityViolations == 0 &&
                consistency.channelAgreement >= MIN_COMPLETE_CHANNEL_AGREEMENT,
            measurements = measurementCount,
            minAbsLlr = if (minAbsLlr.isFinite()) minAbsLlr else 0f,
            avgAbsLlr = sumAbsLlr / MESSAGE_BITS,
            maxAbsLlr = maxAbsLlr,
            channelAgreement = consistency.channelAgreement,
            channelMatchedWeight = consistency.channelMatchedWeight,
            channelTotalWeight = consistency.channelTotalWeight,
            channelMismatchedFactors = consistency.channelMismatchedFactors,
            parityViolations = consistency.parityViolations,
            hardBits = hardBits?.toString() ?: "",
            packetDebug = lastPacketDebug,
        )
    }

    @Synchronized
    fun reset() {
        factors.clear()
        edges.clear()
        for (list in variableEdges) list.clear()
        java.util.Arrays.fill(posterior, 0f)
        ldgmRng.state = LDGM_SEED
        measurementCount = 0
        lastPacketDebug = null
        for (check in 0 until PARITY_BITS) {
            addFactor(
                observedLlr = HARD_ZERO_LLR,
                variables = LDPC_GROUPS[check] + intArrayOf(MESSAGE_BITS + check),
            )
        }
    }

    private fun addFactor(observedLlr: Float, variables: IntArray) {
        val factorIndex = factors.size
        val edgeIndexes = IntArray(variables.size)
        for (edgeOffset in variables.indices) {
            val edgeIndex = edges.size
            val variableIndex = variables[edgeOffset]
            edges += Edge(factorIndex = factorIndex, variableIndex = variableIndex)
            variableEdges[variableIndex] += edgeIndex
            edgeIndexes[edgeOffset] = edgeIndex
        }
        factors += Factor(observedLlr = observedLlr.coerceIn(-LLR_CLAMP, LLR_CLAMP), edgeIndexes = edgeIndexes)
    }

    private fun runIterations(iterations: Int) {
        repeat(iterations) {
            updateFactorMessages()
            updateVariableMessages()
        }
    }

    private fun updateFactorMessages() {
        for (factor in factors) {
            val factorTanh = tanhHalf(factor.observedLlr)
            for (targetEdgeIndex in factor.edgeIndexes) {
                var product = factorTanh
                for (edgeIndex in factor.edgeIndexes) {
                    if (edgeIndex == targetEdgeIndex) continue
                    product *= tanhHalf(edges[edgeIndex].variableToFactor)
                }
                edges[targetEdgeIndex].factorToVariable = (2f * atanhClamped(product))
                    .coerceIn(-LLR_CLAMP, LLR_CLAMP)
            }
        }
    }

    private fun updateVariableMessages() {
        for (variableIndex in 0 until CODEWORD_BITS) {
            val edgeList = variableEdges[variableIndex]
            var sum = 0f
            for (edgeIndex in edgeList) {
                sum += edges[edgeIndex].factorToVariable
            }
            for (edgeIndex in edgeList) {
                val damped = (sum - edges[edgeIndex].factorToVariable).coerceIn(-LLR_CLAMP, LLR_CLAMP)
                edges[edgeIndex].variableToFactor =
                    edges[edgeIndex].variableToFactor * MESSAGE_DAMPING + damped * (1f - MESSAGE_DAMPING)
            }
        }
    }

    private fun updatePosterior() {
        for (variableIndex in 0 until CODEWORD_BITS) {
            var sum = 0f
            for (edgeIndex in variableEdges[variableIndex]) {
                sum += edges[edgeIndex].factorToVariable
            }
            posterior[variableIndex] = sum.coerceIn(-LLR_CLAMP, LLR_CLAMP)
        }
    }

    private data class HardConsistency(
        val channelAgreement: Float,
        val channelMatchedWeight: Float,
        val channelTotalWeight: Float,
        val channelMismatchedFactors: Int,
        val parityViolations: Int,
    )

    private fun hardConsistency(hardCodeword: BooleanArray): HardConsistency {
        var parityViolations = 0
        var channelWeight = 0f
        var matchedWeight = 0f
        var mismatchedFactors = 0
        for (factorIndex in factors.indices) {
            val factor = factors[factorIndex]
            var parity = false
            for (edgeIndex in factor.edgeIndexes) {
                parity = parity xor hardCodeword[edges[edgeIndex].variableIndex]
            }
            if (factorIndex < PARITY_BITS) {
                if (parity) parityViolations += 1
            } else {
                val weight = abs(factor.observedLlr)
                channelWeight += weight
                val expectedOne = factor.observedLlr < 0f
                if (parity == expectedOne) {
                    matchedWeight += weight
                } else {
                    mismatchedFactors += 1
                }
            }
        }
        return HardConsistency(
            channelAgreement = if (channelWeight > 0f) matchedWeight / channelWeight else 1f,
            channelMatchedWeight = matchedWeight,
            channelTotalWeight = channelWeight,
            channelMismatchedFactors = mismatchedFactors,
            parityViolations = parityViolations,
        )
    }

    private fun tanhHalf(llr: Float): Float {
        return tanh((llr * 0.5f).coerceIn(-10f, 10f))
    }

    private fun atanhClamped(value: Float): Float {
        val x = value.coerceIn(-0.9999f, 0.9999f)
        return (0.5f * kotlin.math.ln(((1f + x) / (1f - x)).toDouble())).toFloat()
    }

    companion object {
        const val MESSAGE_BITS = 36
        const val PARITY_BITS = 24
        const val CODEWORD_BITS = MESSAGE_BITS + PARITY_BITS
        const val PACKET_BITS = 5

        private const val LDGM_SEED = 0x12345678L
        private const val MAX_MEASUREMENTS = 360
        private const val MIN_COMPLETE_MEASUREMENTS = 90
        private const val BP_ITERATIONS_PER_PACKET = 8
        private const val LLR_CLAMP = 7.0f
        private const val HARD_ZERO_LLR = 7.0f
        private const val MESSAGE_DAMPING = 0.55f
        private const val TARGET_EXPECTED_ERRORS = 0.28f
        private const val MIN_COMPLETE_LLR = 1.45f
        private const val MIN_COMPLETE_CHANNEL_AGREEMENT = 0.82f
        private const val MAX_EXP = 12f
        private const val MAX_DEBUG_DEGREE = 6
        private const val RNG_FLOAT_SCALE = 4_294_967_296.0f
        private val EMPTY_DEBUG_HISTOGRAM = IntArray(0)

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
    }
}
