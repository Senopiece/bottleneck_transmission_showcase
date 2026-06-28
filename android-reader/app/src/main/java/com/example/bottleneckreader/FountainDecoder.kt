package com.example.bottleneckreader

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.tanh

/**
 * Incremental soft BP decoder for the rateless LDGM layer plus the finite parity precode.
 *
 * LLR convention: positive means bit 0 is more likely, negative means bit 1 is more likely.
 * The graph is bounded: new LDGM observations replace the oldest LDGM factors after
 * MAX_MEASUREMENTS. Belief propagation is advanced separately by pump(), so compute is decoupled
 * from packet arrival rate.
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
        val readyToFinalize: Boolean,
        val hardBits: String,
        val packetDebug: PacketDebug?,
    )

    private data class Edge(
        var factorIndex: Int,
        var variableIndex: Int,
        var variableToFactor: Float = 0f,
        var factorToVariable: Float = 0f,
    )

    private data class Factor(
        var observedLlr: Float,
        var edgeIndexes: IntArray,
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
            val r = next() and UINT_MASK
            val degree = when {
                r < DEGREE_1_CUTOFF -> 1
                r < DEGREE_2_CUTOFF -> 2
                r < DEGREE_3_CUTOFF -> 3
                r < DEGREE_4_CUTOFF -> 4
                r < DEGREE_5_CUTOFF -> 5
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
    private val residualQueue = ResidualQueue(PARITY_BITS + MAX_MEASUREMENTS)
    private var measurementCount = 0
    private var nextMeasurementSlot = 0
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
            val variables = measurementNeighbors(packetIndex * PACKET_BITS + bitIndex)
            if (Diagnostics.enabled) {
                degreeHistogram[variables.size.coerceIn(0, MAX_DEBUG_DEGREE)] += 1
            }
            if (llrs != null) {
                addMeasurementFactor(
                    observedLlr = llrs[bitIndex],
                    variables = variables,
                )
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
        return snapshot()
    }

    @Synchronized
    fun pump(iterations: Int): Snapshot {
        if (measurementCount > 0 && iterations > 0) {
            runIterations(iterations)
        }
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
            val errorProbability = 1f / (1f + exp(absLlr.coerceIn(-MAX_EXP, MAX_EXP)))
            expectedErrors += errorProbability
            if (index < MESSAGE_BITS) {
                if (absLlr < minAbsLlr) minAbsLlr = absLlr
                if (absLlr > maxAbsLlr) maxAbsLlr = absLlr
                sumAbsLlr += absLlr
                certainties[index] = (1f - 2f * errorProbability).coerceIn(0f, 1f)
            }
        }
        val consistency = hardConsistency(hardCodeword)
        val progress = ((CODEWORD_BITS * 0.5f - expectedErrors) / (CODEWORD_BITS * 0.5f - TARGET_EXPECTED_ERRORS))
            .coerceIn(0f, 1f)
        val readyToFinalize = expectedErrors <= TARGET_EXPECTED_ERRORS
        return Snapshot(
            bits = bits,
            certainties = certainties,
            progress = progress,
            expectedErrors = expectedErrors,
            complete = readyToFinalize && consistency.parityViolations == 0,
            measurements = measurementCount,
            minAbsLlr = if (minAbsLlr.isFinite()) minAbsLlr else 0f,
            avgAbsLlr = sumAbsLlr / MESSAGE_BITS,
            maxAbsLlr = maxAbsLlr,
            channelAgreement = consistency.channelAgreement,
            channelMatchedWeight = consistency.channelMatchedWeight,
            channelTotalWeight = consistency.channelTotalWeight,
            channelMismatchedFactors = consistency.channelMismatchedFactors,
            parityViolations = consistency.parityViolations,
            readyToFinalize = readyToFinalize,
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
        residualQueue.clear()
        measurementCount = 0
        nextMeasurementSlot = 0
        lastPacketDebug = null
        for (check in 0 until PARITY_BITS) {
            addParityFactor(
                observedLlr = HARD_ZERO_LLR,
                variables = LDPC_GROUPS[check] + intArrayOf(MESSAGE_BITS + check),
            )
        }
        scheduleAllFactors()
    }

    private fun addParityFactor(observedLlr: Float, variables: IntArray) {
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

    private fun addMeasurementFactor(observedLlr: Float, variables: IntArray) {
        val slot = nextMeasurementSlot
        nextMeasurementSlot = (nextMeasurementSlot + 1) % MAX_MEASUREMENTS
        val factorIndex = PARITY_BITS + slot
        replaceMeasurementFactor(
            factorIndex = factorIndex,
            observedLlr = observedLlr,
            variables = variables,
        )
        if (measurementCount < MAX_MEASUREMENTS) {
            measurementCount += 1
        }
    }

    private fun replaceMeasurementFactor(factorIndex: Int, observedLlr: Float, variables: IntArray) {
        val oldEdgeIndexes = if (factorIndex < factors.size) {
            val oldFactor = factors[factorIndex]
            detachFactor(factorIndex, oldFactor)
            oldFactor.edgeIndexes
        } else {
            IntArray(0)
        }
        val edgeIndexes = IntArray(variables.size)
        val touchedVariables = IntArray(variables.size)
        var touchedCount = 0
        for (edgeOffset in variables.indices) {
            val edgeIndex = if (edgeOffset < oldEdgeIndexes.size) {
                oldEdgeIndexes[edgeOffset]
            } else {
                val newEdgeIndex = edges.size
                edges += Edge(factorIndex = factorIndex, variableIndex = variables[edgeOffset])
                newEdgeIndex
            }
            val variableIndex = variables[edgeOffset]
            val edge = edges[edgeIndex]
            edge.factorIndex = factorIndex
            edge.variableIndex = variableIndex
            edge.variableToFactor = 0f
            edge.factorToVariable = 0f
            variableEdges[variableIndex] += edgeIndex
            edgeIndexes[edgeOffset] = edgeIndex
            touchedVariables[touchedCount] = variableIndex
            touchedCount += 1
        }
        val factor = Factor(
            observedLlr = observedLlr.coerceIn(-LLR_CLAMP, LLR_CLAMP),
            edgeIndexes = edgeIndexes,
        )
        if (factorIndex < factors.size) {
            factors[factorIndex] = factor
        } else {
            while (factors.size < factorIndex) {
                error("Unexpected gap while adding measurement factor")
            }
            factors += factor
        }
        for (index in 0 until touchedCount) {
            updateVariableMessages(touchedVariables[index])
            scheduleVariableFactors(touchedVariables[index])
        }
        scheduleFactor(factorIndex)
    }

    private fun detachFactor(factorIndex: Int, factor: Factor) {
        residualQueue.remove(factorIndex)
        for (edgeIndex in factor.edgeIndexes) {
            val edge = edges[edgeIndex]
            removeEdgeFromVariable(edge.variableIndex, edgeIndex)
            edge.variableToFactor = 0f
            edge.factorToVariable = 0f
        }
    }

    private fun removeEdgeFromVariable(variableIndex: Int, edgeIndex: Int) {
        val list = variableEdges[variableIndex]
        for (index in 0 until list.size) {
            if (list[index] == edgeIndex) {
                list.removeAt(index)
                return
            }
        }
    }

    private fun measurementNeighbors(measurementIndex: Int): IntArray {
        return SeededRng(mixMeasurementSeed(measurementIndex)).nextNeighbors(CODEWORD_BITS)
    }

    private fun mixMeasurementSeed(measurementIndex: Int): Long {
        var x = (LDGM_SEED + measurementIndex.toLong() * MIX_GOLDEN_RATIO) and UINT_MASK
        x = ((x xor (x ushr 16)) * MIX_MURMUR_1) and UINT_MASK
        x = ((x xor (x ushr 13)) * MIX_MURMUR_2) and UINT_MASK
        return (x xor (x ushr 16)) and UINT_MASK
    }

    private fun runIterations(iterations: Int) {
        runResidualUpdates(iterations * RESIDUAL_FACTOR_UPDATES_PER_ITERATION)
    }

    private fun runResidualUpdates(updates: Int) {
        for (update in 0 until updates) {
            val bestFactorIndex = residualQueue.popMax(MIN_RESIDUAL_DELTA)
            if (bestFactorIndex < 0) return
            updateResidualFactor(bestFactorIndex)
        }
    }

    private fun scheduleAllFactors() {
        for (factorIndex in factors.indices) {
            scheduleFactor(factorIndex)
        }
    }

    private fun scheduleFactor(factorIndex: Int) {
        if (factorIndex !in factors.indices) return
        residualQueue.update(factorIndex, factorResidual(factors[factorIndex]), MIN_RESIDUAL_DELTA)
    }

    private fun scheduleVariableFactors(variableIndex: Int) {
        for (edgeIndex in variableEdges[variableIndex]) {
            scheduleFactor(edges[edgeIndex].factorIndex)
        }
    }

    private fun factorResidual(factor: Factor): Float {
        var maxDelta = 0f
        for (targetEdgeIndex in factor.edgeIndexes) {
            val nextMessage = factorMessage(factor, targetEdgeIndex)
            val delta = abs(nextMessage - edges[targetEdgeIndex].factorToVariable)
            if (delta > maxDelta) maxDelta = delta
        }
        return maxDelta
    }

    private fun updateResidualFactor(factorIndex: Int) {
        val factor = factors[factorIndex]
        val touchedVariables = IntArray(factor.edgeIndexes.size)
        var touchedCount = 0
        for (targetEdgeIndex in factor.edgeIndexes) {
            val edge = edges[targetEdgeIndex]
            val nextMessage = factorMessage(factor, targetEdgeIndex)
            edge.factorToVariable = (edge.factorToVariable * FACTOR_MESSAGE_DAMPING +
                nextMessage * (1f - FACTOR_MESSAGE_DAMPING))
                .coerceIn(-LLR_CLAMP, LLR_CLAMP)
            var exists = false
            for (index in 0 until touchedCount) {
                if (touchedVariables[index] == edge.variableIndex) {
                    exists = true
                    break
                }
            }
            if (!exists) {
                touchedVariables[touchedCount] = edge.variableIndex
                touchedCount += 1
            }
        }
        for (index in 0 until touchedCount) {
            val variableIndex = touchedVariables[index]
            updateVariableMessages(variableIndex)
            scheduleVariableFactors(variableIndex)
        }
        scheduleFactor(factorIndex)
    }

    private fun factorMessage(factor: Factor, targetEdgeIndex: Int): Float {
        var product = tanhHalf(factor.observedLlr)
        for (edgeIndex in factor.edgeIndexes) {
            if (edgeIndex == targetEdgeIndex) continue
            product *= tanhHalf(edges[edgeIndex].variableToFactor)
        }
        return (2f * atanhClamped(product)).coerceIn(-LLR_CLAMP, LLR_CLAMP)
    }

    private fun updateVariableMessages(variableIndex: Int) {
        val edgeList = variableEdges[variableIndex]
        var sum = 0f
        for (edgeIndex in edgeList) {
            sum += edges[edgeIndex].factorToVariable
        }
        for (edgeIndex in edgeList) {
            val nextMessage = (sum - edges[edgeIndex].factorToVariable).coerceIn(-LLR_CLAMP, LLR_CLAMP)
            edges[edgeIndex].variableToFactor =
                edges[edgeIndex].variableToFactor * MESSAGE_DAMPING + nextMessage * (1f - MESSAGE_DAMPING)
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

    private class ResidualQueue(capacity: Int) {
        private val heap = IntArray(capacity)
        private val positions = IntArray(capacity) { -1 }
        private val priorities = FloatArray(capacity)
        private var size = 0

        fun clear() {
            java.util.Arrays.fill(positions, -1)
            java.util.Arrays.fill(priorities, 0f)
            size = 0
        }

        fun update(key: Int, priority: Float, minPriority: Float) {
            if (key !in positions.indices) return
            if (priority < minPriority) {
                remove(key)
                return
            }
            val position = positions[key]
            priorities[key] = priority
            if (position >= 0) {
                siftUp(position)
                val nextPosition = positions[key]
                if (nextPosition >= 0) siftDown(nextPosition)
            } else {
                if (size >= heap.size) return
                heap[size] = key
                positions[key] = size
                siftUp(size)
                size += 1
            }
        }

        fun remove(key: Int) {
            if (key !in positions.indices) return
            val position = positions[key]
            if (position < 0) return
            val lastIndex = size - 1
            swap(position, lastIndex)
            size -= 1
            positions[key] = -1
            priorities[key] = 0f
            if (position < size) {
                val movedKey = heap[position]
                siftUp(position)
                val movedPosition = positions[movedKey]
                if (movedPosition >= 0) siftDown(movedPosition)
            }
        }

        fun popMax(minPriority: Float): Int {
            while (size > 0) {
                val key = heap[0]
                val priority = priorities[key]
                remove(key)
                if (priority >= minPriority) return key
            }
            return -1
        }

        private fun siftUp(start: Int) {
            var index = start
            while (index > 0) {
                val parent = (index - 1) / 2
                if (priorities[heap[parent]] >= priorities[heap[index]]) break
                swap(parent, index)
                index = parent
            }
        }

        private fun siftDown(start: Int) {
            var index = start
            while (true) {
                val left = index * 2 + 1
                if (left >= size) return
                val right = left + 1
                var best = left
                if (right < size && priorities[heap[right]] > priorities[heap[left]]) {
                    best = right
                }
                if (priorities[heap[index]] >= priorities[heap[best]]) return
                swap(index, best)
                index = best
            }
        }

        private fun swap(a: Int, b: Int) {
            if (a == b) return
            val keyA = heap[a]
            val keyB = heap[b]
            heap[a] = keyB
            heap[b] = keyA
            positions[keyA] = b
            positions[keyB] = a
        }
    }

    companion object {
        const val MESSAGE_BITS = 36
        const val PARITY_BITS = 24
        const val CODEWORD_BITS = MESSAGE_BITS + PARITY_BITS
        const val PACKET_BITS = 5

        private const val LDGM_SEED = 0x12345678L
        private const val UINT_MASK = 0xFFFFFFFFL
        private const val MIX_GOLDEN_RATIO = 0x9E3779B9L
        private const val MIX_MURMUR_1 = 0x85EBCA6BL
        private const val MIX_MURMUR_2 = 0xC2B2AE35L
        private const val DEGREE_1_CUTOFF = 858_993_459L
        private const val DEGREE_2_CUTOFF = 2_147_483_648L
        private const val DEGREE_3_CUTOFF = 3_221_225_472L
        private const val DEGREE_4_CUTOFF = 3_865_470_566L
        private const val DEGREE_5_CUTOFF = 4_166_118_277L
        const val MAX_MEASUREMENTS = 360
        private const val LLR_CLAMP = 7.0f
        private const val HARD_ZERO_LLR = 7.0f
        private const val MESSAGE_DAMPING = 0.55f
        private const val FACTOR_MESSAGE_DAMPING = 0.35f
        private const val RESIDUAL_FACTOR_UPDATES_PER_ITERATION = 24
        private const val MIN_RESIDUAL_DELTA = 0.002f
        private const val TARGET_EXPECTED_ERRORS = 0.28f
        private const val MAX_EXP = 12f
        private const val MAX_DEBUG_DEGREE = 6
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
