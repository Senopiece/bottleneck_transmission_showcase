package com.example.bottleneckreader

import android.util.Log
import java.util.Locale

class ScoreLogger {
    private var headerLogged = false

    fun log(
        timestampNs: Long,
        scores: FloatArray?,
        event: PacketClockDecoder.Result?,
        state: String,
    ) {
        if (!headerLogged) {
            Log.d(SCORE_CSV_TAG, "timestampNs,detected,s0,s1,s2,s3,s4,event")
            headerLogged = true
        }
        val eventValue = when {
            event == null -> state
            event.failure != null -> "fail ${event.failure.name}"
            event.bits == null -> "erasure ${event.packetKind ?: "unknown"} $state"
            else -> event.bits + " ${event.packetKind ?: "unknown"} $state"
        }
        val debugValue = event?.debugInfo?.let { info ->
            listOf(
                "periodMs=${String.format(Locale.US, "%.2f", info.periodNs / 1_000_000f)}",
                "measuredMs=${String.format(Locale.US, "%.2f", info.measuredPeriodNs / 1_000_000f)}",
                "late=${info.periodsElapsed}",
                "samples=${info.sampleCount}",
                "weight=${String.format(Locale.US, "%.3f", info.sampleWeightSum)}",
                "avg=${formatArray(info.averages)}",
                "peak=${formatArray(info.peaks)}",
                "rel=${formatArray(info.reliabilities)}",
            ).joinToString(separator = " ")
        }
        val scoreColumns = if (scores == null) {
            ",,,,"
        } else {
            scores.joinToString(separator = ",") { String.format(Locale.US, "%.4f", it) }
        }
        Log.d(
            SCORE_CSV_TAG,
            "$timestampNs,${if (scores == null) 0 else 1},$scoreColumns,$eventValue${debugValue?.let { " $it" } ?: ""}",
        )
    }

    private fun formatArray(values: FloatArray): String {
        if (values.isEmpty()) return "-"
        return values.joinToString(separator = "|") { String.format(Locale.US, "%.3f", it) }
    }

    private companion object {
        const val SCORE_CSV_TAG = "LedScoresCsv"
    }
}
