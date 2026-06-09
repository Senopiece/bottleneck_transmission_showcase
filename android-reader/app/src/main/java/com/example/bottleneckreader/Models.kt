package com.example.bottleneckreader

import androidx.compose.ui.geometry.Offset

data class ImagePoint(
    val x: Float,
    val y: Float,
)

data class LedSlot(
    val imagePoint: ImagePoint,
    val bitIndex: Int,
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
    val bits: String?,
    val slots: List<LedSlot>,
    val markers: List<MarkerSlot> = emptyList(),
    val debugLines: List<String> = emptyList(),
)

data class OverlayFrame(
    val bits: String?,
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

data class ReaderNotice(
    val id: Long,
    val message: String,
    val exiting: Boolean = false,
)

data class CameraProblem(
    val title: String,
    val message: String,
)

sealed interface ReaderEvent {
    data class Detection(val frame: DetectionFrame) : ReaderEvent
    data class Notice(val message: String) : ReaderEvent
    data class CameraIssue(val problem: CameraProblem) : ReaderEvent
    data object SlowDecoderTerminated : ReaderEvent
}
