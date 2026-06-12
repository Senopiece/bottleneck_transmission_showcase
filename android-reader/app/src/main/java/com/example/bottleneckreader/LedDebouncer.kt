package com.example.bottleneckreader

class LedDebouncer {
    data class Result(val bits: String?)

    private var externalState: String? = null
    private var pendingPacket: String? = null
    private var pendingFrames = 0
    private var bitStates: BooleanArray? = null

    fun accept(scores: List<Float>?): Result? {
        if (scores == null || scores.size != BIT_COUNT) {
            pendingPacket = null
            pendingFrames = 0
            bitStates = null
            if (externalState != null) {
                externalState = null
                return Result(null)
            }
            return null
        }

        val packet = scoresToPacket(scores)
        if (packet == externalState) {
            pendingPacket = null
            pendingFrames = 0
            return null
        }

        if (packet != pendingPacket) {
            pendingPacket = packet
            pendingFrames = 1
            return null
        }

        pendingFrames += 1
        if (pendingFrames >= REQUIRED_STABLE_FRAMES) {
            externalState = packet
            pendingPacket = null
            pendingFrames = 0
            return Result(packet)
        }

        return null
    }

    fun reset(): Result? {
        pendingPacket = null
        pendingFrames = 0
        bitStates = null
        if (externalState != null) {
            externalState = null
            return Result(null)
        }
        return null
    }

    private fun scoresToPacket(scores: List<Float>): String {
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

    private companion object {
        const val BIT_COUNT = 5
        const val REQUIRED_STABLE_FRAMES = 2
        const val ON_THRESHOLD = 0.72f
        const val OFF_THRESHOLD = 0.48f
    }
}
