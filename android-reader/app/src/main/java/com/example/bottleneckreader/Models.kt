package com.example.bottleneckreader

import androidx.compose.ui.geometry.Offset

data class ImagePoint(
    val x: Float,
    val y: Float,
)

data class LedSlot(
    val imagePoint: ImagePoint,
    val isFirst: Boolean,
    val imageRadius: Float,
)

enum class MarkerKind {
    StartSquare,
    EndTriangle,
}

data class MarkerSlot(
    val imagePoint: ImagePoint,
    val imageAlongPoint: ImagePoint,
    val kind: MarkerKind,
    val imageSize: Float,
)

data class DetectionFrame(
    val timestampNs: Long,
    val imageWidth: Int,
    val imageHeight: Int,
    val cropLeft: Int,
    val cropTop: Int,
    val cropWidth: Int,
    val cropHeight: Int,
    val rotationDegrees: Int,
    val ledScores: FloatArray = FloatArray(0),
    val patternConfidence: Float = 1f,
    val slots: List<LedSlot>,
    val markers: List<MarkerSlot> = emptyList(),
    val isAcquireMode: Boolean = false,
    val debugLines: List<String> = emptyList(),
)

data class OverlayFrame(
    val slots: List<OverlaySlot>,
    val markers: List<OverlayMarker>,
)

data class OverlaySlot(
    val point: Offset,
    val isFirst: Boolean,
    val radiusPx: Float,
)

data class OverlayMarker(
    val point: Offset,
    val alongPoint: Offset,
    val kind: MarkerKind,
    val sizePx: Float,
)

data class PacketEvent(
    val id: Long,
    val timestampNs: Long,
    val bits: String?,
)

data class DecodeProgress(
    val confidenceProgress: Float = 0f,
    val visible: Boolean = false,
    val failed: Boolean = false,
    val failureId: Long = 0L,
    val phase: DecodePhase = DecodePhase.Idle,
)

enum class DecodePhase {
    Idle,
    Preamble,
    Decoding,
}

data class DecodedMessage(
    val id: Long,
    val bits: BooleanArray,
)

data class DecoderTimingWindow(
    val samplesMs: List<Float> = emptyList(),
    val avgMs: Float = 0f,
    val minMs: Float = 0f,
    val maxMs: Float = 0f,
)

data class CameraProblem(
    val title: String,
    val message: String,
)

sealed interface ReaderEvent {
    data class Detection(val frame: DetectionFrame) : ReaderEvent
    data class DecoderTiming(val elapsedMs: Float) : ReaderEvent
    data class CameraIssue(val problem: CameraProblem) : ReaderEvent
}
