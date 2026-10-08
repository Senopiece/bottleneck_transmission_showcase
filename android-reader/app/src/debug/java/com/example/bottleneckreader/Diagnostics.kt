package com.example.bottleneckreader

import android.util.Log

object Diagnostics {
    const val enabled = true

    fun logVisionTiming(message: String) {
        Log.d("VisionTiming", message)
    }
}
