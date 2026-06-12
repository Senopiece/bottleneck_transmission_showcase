package com.example.bottleneckreader

import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.util.Locale
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

class LedFrameDecoder {
    private data class RotatedRoi(
        val left: Float,
        val top: Float,
        val width: Float,
        val height: Float,
    )

    private data class Theta(
        val cx: Float,
        val cy: Float,
        val angle: Float,
        val logDistance: Float,
    ) {
        val distance: Float get() = exp(logDistance)
    }

    private data class PatternModel(
        val start: ImagePoint,
        val end: ImagePoint,
        val slots: Array<ImagePoint>,
        val markerSizePx: Float,
        val squareSizePx: Float,
        val triangleSizePx: Float,
        val ledRadiusPx: Float,
        val ux: Float,
        val uy: Float,
        val vx: Float,
        val vy: Float,
        val distancePx: Float,
    )

    private data class Fit(
        val theta: Theta,
        val breakdown: ScoreBreakdown,
    ) {
        val score: Float get() = breakdown.score
    }

    private data class ScoreBreakdown(
        val score: Float,
        val square: Float,
        val triangle: Float,
    )

    private val constants = GeometryConstants()

    var lastDebugLines: List<String> = emptyList()
        private set

    private var previousTheta: Theta? = null
    private var previousScore = 0f
    private var missedFrames = 0
    private var debugModeLine = "mode: none"
    private var debugBitsLine = "ledScore: none"
    private var debugBestScore = BAD_SCORE
    private var debugBestSquare = 0f
    private var debugBestTriangle = 0f
    private var debugBestDistanceInRoi = 0f

    fun decode(image: ImageProxy): DetectionFrame? {
        val reader = YuvReader(image)
        val roi = reader.rotatedRoi()
        beginDebugFrame()

        val seed = initialTheta(reader, roi)
        val tracking = previousTheta != null && missedFrames <= TRACKING_CONTINUITY_MISSES
        debugModeLine = if (tracking) {
            "mode: tracking prevScore=${fmt(previousScore)} missed=$missedFrames"
        } else {
            "mode: acquire centered"
        }

        var fit = refine(reader, roi, seed, if (tracking) TRACKING_STEPS else ACQUIRE_STEPS)
        if (tracking && !isAccepted(fit)) {
            debugModeLine = "mode: tracking fallback prevScore=${fmt(previousScore)} missed=$missedFrames"
            fit = refine(reader, roi, centeredTheta(reader, roi), ACQUIRE_STEPS)
        }
        if (!isAccepted(fit)) {
            missedFrames++
            if (missedFrames >= RESET_AFTER_MISSES) {
                previousTheta = null
                previousScore = 0f
            }
            finishDebugFrame("MISS", fit)
            return null
        }

        previousTheta = fit.theta
        previousScore = fit.score
        missedFrames = 0

        val frame = decodeWithModel(reader, modelForTheta(fit.theta))
        val debugLines = finishDebugFrame("HIT", fit)
        return frame.copy(debugLines = debugLines)
    }

    fun resetTracking() {
        previousTheta = null
        previousScore = 0f
        missedFrames = 0
        lastDebugLines = emptyList()
    }

    private fun beginDebugFrame() {
        debugModeLine = "mode: none"
        debugBitsLine = "ledScore: none"
        debugBestScore = BAD_SCORE
        debugBestSquare = 0f
        debugBestTriangle = 0f
        debugBestDistanceInRoi = 0f
    }

    private fun finishDebugFrame(status: String, fit: Fit?): List<String> {
        val lines = ArrayList<String>(7)
        lines.add(status)
        lines.add(debugModeLine)
        if (fit != null) {
            lines.add("fit score=${fmt(fit.score)} d=${fmt(fit.theta.distance)} angle=${fmt(fit.theta.angle)}")
        } else {
            lines.add("fit: none prevScore=${fmt(previousScore)} missed=$missedFrames")
        }
        lines.add(debugBestLine())
        lines.add(debugBitsLine)
        lastDebugLines = lines
        return lines
    }

    private fun fmt(value: Float): String = String.format(Locale.US, "%.2f", value)

    private fun debugBestLine(): String {
        if (debugBestScore == BAD_SCORE) return "best: none"
        return "best score=${fmt(debugBestScore)} sq=${fmt(debugBestSquare)} tri=${fmt(debugBestTriangle)}" +
            " d=${fmt(debugBestDistanceInRoi)}"
    }

    private fun initialTheta(reader: YuvReader, roi: RotatedRoi): Theta {
        val previous = previousTheta
        if (previous != null && missedFrames <= TRACKING_CONTINUITY_MISSES) return previous
        return centeredTheta(reader, roi)
    }

    private fun centeredTheta(reader: YuvReader, roi: RotatedRoi): Theta {
        val rotatedCenterX = roi.left + roi.width * 0.5f
        val rotatedCenterY = roi.top + roi.height * 0.5f
        val center = reader.rotatedToImage(rotatedCenterX, rotatedCenterY)
        val p1 = reader.rotatedToImage(rotatedCenterX + 1f, rotatedCenterY)
        val angle = atan2(p1.y - center.y, p1.x - center.x)
        val distance = (roi.width * INITIAL_PATTERN_DISTANCE_FRACTION)
            .coerceAtLeast(MIN_PATTERN_DISTANCE_PX)
        return Theta(
            cx = center.x,
            cy = center.y,
            angle = angle,
            logDistance = ln(distance),
        )
    }

    private fun refine(reader: YuvReader, roi: RotatedRoi, seed: Theta, steps: Array<Step>): Fit {
        var theta = normalizeTheta(seed, roi)
        var breakdown = scoreBreakdown(reader, roi, theta)
        var score = breakdown.score
        var bestTheta = theta
        var bestBreakdown = breakdown
        var bestScore = score

        for (step in steps) {
            var localBestTheta = theta
            var localBestBreakdown = breakdown
            var localBestScore = score

            fun tryCandidate(candidate: Theta) {
                val normalized = normalizeTheta(candidate, roi)
                val candidateBreakdown = scoreBreakdown(reader, roi, normalized)
                val candidateScore = candidateBreakdown.score
                if (candidateScore > localBestScore) {
                    localBestScore = candidateScore
                    localBestBreakdown = candidateBreakdown
                    localBestTheta = normalized
                }
            }

            tryCandidate(theta.copy(cx = theta.cx + step.translationPx))
            tryCandidate(theta.copy(cx = theta.cx - step.translationPx))
            tryCandidate(theta.copy(cy = theta.cy + step.translationPx))
            tryCandidate(theta.copy(cy = theta.cy - step.translationPx))
            tryCandidate(theta.copy(angle = theta.angle + step.angleRad))
            tryCandidate(theta.copy(angle = theta.angle - step.angleRad))
            tryCandidate(theta.copy(logDistance = theta.logDistance + step.logDistance))
            tryCandidate(theta.copy(logDistance = theta.logDistance - step.logDistance))

            val improved = localBestScore > score
            theta = localBestTheta
            breakdown = localBestBreakdown
            score = localBestScore
            if (score > bestScore) {
                bestScore = score
                bestTheta = theta
                bestBreakdown = breakdown
            }
            if (!improved && isAccepted(bestBreakdown, bestScore)) break
        }

        return Fit(bestTheta, bestBreakdown)
    }

    private fun normalizeTheta(theta: Theta, roi: RotatedRoi): Theta {
        val minDistance = max(MIN_PATTERN_DISTANCE_PX, roi.width * 0.58f)
        val maxDistance = roi.width * 1.02f
        return theta.copy(
            angle = normalizeAngle(theta.angle),
            logDistance = ln(theta.distance.coerceIn(minDistance, maxDistance)),
        )
    }

    private fun isAccepted(fit: Fit): Boolean {
        return isAccepted(fit.breakdown, fit.score)
    }

    private fun isAccepted(parts: ScoreBreakdown, score: Float): Boolean {
        if (score < MIN_ACCEPT_SCORE) return false
        return parts.square >= MIN_ACCEPT_SQUARE &&
            parts.triangle >= MIN_ACCEPT_TRIANGLE
    }

    private fun scoreBreakdown(reader: YuvReader, roi: RotatedRoi, theta: Theta): ScoreBreakdown {
        val model = modelForTheta(theta)
        if (!modelInsideRoi(reader, roi, model)) {
            return ScoreBreakdown(BAD_SCORE, 0f, 0f)
        }

        val square = squareTemplateScore(reader, model)
        val triangle = triangleTemplateScore(reader, model)
        val markerDistanceInRoi = model.distancePx / roi.width

        val score = square * 2.95f +
            triangle * 4.35f

        if (score > debugBestScore) {
            debugBestScore = score
            debugBestSquare = square
            debugBestTriangle = triangle
            debugBestDistanceInRoi = markerDistanceInRoi
        }
        return ScoreBreakdown(
            score = score,
            square = square,
            triangle = triangle,
        )
    }

    private fun modelForTheta(theta: Theta): PatternModel {
        val angle = normalizeAngle(theta.angle)
        val ux = cos(angle)
        val uy = sin(angle)
        val vx = -uy
        val vy = ux
        val distance = theta.distance
        val markerSize = distance / constants.markerDistanceToSizeRatio()
        val start = ImagePoint(theta.cx - ux * distance * 0.5f, theta.cy - uy * distance * 0.5f)
        val end = ImagePoint(theta.cx + ux * distance * 0.5f, theta.cy + uy * distance * 0.5f)
        val slotFractions = constants.slotFractions
        val slots = Array(slotFractions.size) { index ->
            val fraction = slotFractions[index]
            ImagePoint(start.x + ux * distance * fraction, start.y + uy * distance * fraction)
        }
        return PatternModel(
            start = start,
            end = end,
            slots = slots,
            markerSizePx = markerSize,
            squareSizePx = markerSize * constants.squareSizeToMarkerSizeRatio(),
            triangleSizePx = markerSize * constants.triangleSizeToMarkerSizeRatio(),
            ledRadiusPx = markerSize * constants.ledRadiusToMarkerSizeRatio(),
            ux = ux,
            uy = uy,
            vx = vx,
            vy = vy,
            distancePx = distance,
        )
    }

    private fun modelInsideRoi(reader: YuvReader, roi: RotatedRoi, model: PatternModel): Boolean {
        val margin = max(2f, model.markerSizePx * 0.58f)
        for (slot in model.slots) {
            if (!pointInsideRoi(reader, roi, slot, margin)) return false
        }
        return pointInsideRoi(reader, roi, model.start, margin) &&
            pointInsideRoi(reader, roi, model.end, margin)
    }

    private fun pointInsideRoi(reader: YuvReader, roi: RotatedRoi, point: ImagePoint, margin: Float): Boolean {
        if (point.x < margin || point.x >= reader.width - margin || point.y < margin || point.y >= reader.height - margin) {
            return false
        }
        val rotated = reader.imageToRotated(point)
        return rotated.x >= roi.left + margin &&
            rotated.x <= roi.left + roi.width - margin &&
            rotated.y >= roi.top + margin &&
            rotated.y <= roi.top + roi.height - margin
    }

    private fun squareTemplateScore(reader: YuvReader, model: PatternModel): Float {
        val size = model.squareSizePx
        val radius = sampleRadius(size)
        val insideMean = markerMean(reader, model, model.start, squareInsideSamples, size, radius)
        val outsideMean = markerMean(reader, model, model.start, squareOutsideSamples, size, radius)
        val cornerMin = markerMin(reader, model, model.start, squareCornerSamples, size, radius)
        val edgeContrast = (insideMean - outsideMean).coerceIn(0f, 1f)
        val fill = min(insideMean, cornerMin + 0.12f)
        val compact = (markerValue(reader, model.start, radius) - cornerMin - 0.18f).coerceIn(0f, 1f)
        val score = (fill * 0.42f + edgeContrast * 0.42f + cornerMin * 0.28f - compact * 0.55f)
            .coerceIn(0f, 1f)
        return score
    }

    private fun triangleTemplateScore(reader: YuvReader, model: PatternModel): Float {
        val size = model.triangleSizePx
        val radius = sampleRadius(size)
        val insideMean = markerMean(reader, model, model.end, triangleInsideSamples, size, radius)
        val baseMean = markerMean(reader, model, model.end, triangleBaseSamples, size, radius)
        val outsideMean = markerMean(reader, model, model.end, triangleOutsideSamples, size, radius)
        val upperMean = markerMean(reader, model, model.end, triangleUpperEmptySamples, size, radius)
        val contrast = (insideMean - outsideMean).coerceIn(0f, 1f)
        val taper = (baseMean - upperMean).coerceIn(0f, 1f)
        val fill = min(insideMean, baseMean + 0.10f)
        val compact = (insideMean - baseMean - 0.20f).coerceIn(0f, 1f)
        val score = (fill * 0.36f + contrast * 0.34f + taper * 0.42f - compact * 0.62f)
            .coerceIn(0f, 1f)
        return score
    }

    private fun decodeWithModel(reader: YuvReader, model: PatternModel): DetectionFrame {
        val scores = readLedScores(reader, model)
        val overlayRadius = (model.ledRadiusPx * 1.25f).coerceIn(3.5f, 13f)
        return DetectionFrame(
            timestampNs = reader.timestampNs,
            imageWidth = reader.width,
            imageHeight = reader.height,
            cropLeft = reader.cropLeft,
            cropTop = reader.cropTop,
            cropWidth = reader.cropWidth,
            cropHeight = reader.cropHeight,
            rotationDegrees = reader.rotationDegrees,
            bits = null,
            ledScores = scores.toList(),
            slots = model.slots.mapIndexed { index, point ->
                LedSlot(
                    imagePoint = point,
                    bitIndex = 4 - index,
                    isFirst = index == 0,
                    imageRadius = overlayRadius,
                )
            },
            markers = listOf(
                MarkerSlot(
                    imagePoint = model.start,
                    imageAlongPoint = model.end,
                    kind = MarkerKind.StartSquare,
                    imageSize = model.squareSizePx,
                ),
                MarkerSlot(
                    imagePoint = model.end,
                    imageAlongPoint = ImagePoint(
                        x = model.end.x + model.ux * model.triangleSizePx,
                        y = model.end.y + model.uy * model.triangleSizePx,
                    ),
                    kind = MarkerKind.EndTriangle,
                    imageSize = model.triangleSizePx,
                ),
            ),
        )
    }

    private fun readLedScores(reader: YuvReader, model: PatternModel): FloatArray {
        val scores = FloatArray(model.slots.size)
        for (index in model.slots.indices) {
            scores[index] = ledOnScore(reader, model.slots[index], model.ledRadiusPx, model)
        }
        debugBitsLine = buildString {
            append("ledScore")
            scores.forEach { append(' ').append(fmt(it)) }
        }
        return scores
    }

    private fun ledOnScore(reader: YuvReader, center: ImagePoint, radiusPx: Float, model: PatternModel): Float {
        val r = radiusPx.roundToInt().coerceIn(2, 9)
        val tightR = max(1, (r * 0.72f).roundToInt())
        val side = radiusPx * 2.65f
        val centerLuma = lumaMean(reader, center, tightR)
        val centerPeak = lumaMax(reader, center, r)
        val bgLuma = (
            lumaMean(reader, center.local(model, 0f, side, 1f), r) +
                lumaMean(reader, center.local(model, 0f, -side, 1f), r) +
                lumaMean(reader, center.local(model, side, 0f, 1f), r) +
                lumaMean(reader, center.local(model, -side, 0f, 1f), r)
            ) * 0.25f
        val blue = blueObjectValue(reader, center, r)
        val lumaContrast = ((centerLuma - bgLuma - 10f) / 78f).coerceIn(-0.35f, 1.35f)
        val peakContrast = ((centerPeak - bgLuma - 18f) / 95f).coerceIn(-0.35f, 1.35f)
        val highlight = ((centerPeak - 150f) / 85f).coerceIn(0f, 1.15f)
        return (lumaContrast * 0.42f + peakContrast * 0.38f + highlight * 0.20f + blue * 0.06f)
            .coerceIn(-0.5f, 1.6f)
    }

    private fun ledObjectValue(reader: YuvReader, center: ImagePoint, radiusPx: Float): Float {
        val r = radiusPx.roundToInt().coerceIn(2, 9)
        val blue = blueObjectValue(reader, center, r)
        val bright = ((lumaMean(reader, center, r) - 70f) / 130f).coerceIn(0f, 1f)
        return (blue * 0.62f + bright * 0.38f).coerceIn(0f, 1f)
    }

    private fun markerValue(reader: YuvReader, center: ImagePoint, radiusPx: Int): Float {
        var score = 0f
        var weightSum = 0f
        val cx = center.x.roundToInt()
        val cy = center.y.roundToInt()
        for (i in SAMPLE_X.indices) {
            val x = cx + (SAMPLE_X[i] * radiusPx).roundToInt()
            val y = cy + (SAMPLE_Y[i] * radiusPx).roundToInt()
            if (x !in 0 until reader.width || y !in 0 until reader.height) continue
            val yy = reader.y(x, y)
            val neutral = (1f - (abs(reader.u(x, y) - 128) + abs(reader.v(x, y) - 128)) / 128f)
                .coerceIn(0f, 1f)
            val brightness = ((yy - 78f) / 148f).coerceIn(0f, 1f)
            val bluePenalty = (max(0f, reader.blueDominance(x, y) - 38f) / 95f).coerceIn(0f, 1f)
            val w = SAMPLE_W[i]
            score += brightness * (0.50f + neutral * 0.50f) * (1f - bluePenalty * 0.72f) * w
            weightSum += w
        }
        return if (weightSum == 0f) 0f else score / weightSum
    }

    private fun blueObjectValue(reader: YuvReader, center: ImagePoint, radiusPx: Int): Float {
        var score = 0f
        var weightSum = 0f
        val cx = center.x.roundToInt()
        val cy = center.y.roundToInt()
        for (i in SAMPLE_X.indices) {
            val x = cx + (SAMPLE_X[i] * radiusPx).roundToInt()
            val y = cy + (SAMPLE_Y[i] * radiusPx).roundToInt()
            if (x !in 0 until reader.width || y !in 0 until reader.height) continue
            val blue = ((reader.blueDominance(x, y) - 12f) / 92f).coerceIn(0f, 1f)
            val visible = ((abs(reader.y(x, y) - 72f) + 24f) / 160f).coerceIn(0.15f, 1f)
            val w = SAMPLE_W[i]
            score += blue * visible * w
            weightSum += w
        }
        return if (weightSum == 0f) 0f else score / weightSum
    }

    private fun lumaMean(reader: YuvReader, center: ImagePoint, radiusPx: Int): Float {
        var sum = 0f
        var weightSum = 0f
        val cx = center.x.roundToInt()
        val cy = center.y.roundToInt()
        for (i in SAMPLE_X.indices) {
            val x = cx + (SAMPLE_X[i] * radiusPx).roundToInt()
            val y = cy + (SAMPLE_Y[i] * radiusPx).roundToInt()
            if (x !in 0 until reader.width || y !in 0 until reader.height) continue
            val w = SAMPLE_W[i]
            sum += reader.y(x, y) * w
            weightSum += w
        }
        return if (weightSum == 0f) 0f else sum / weightSum
    }

    private fun lumaMax(reader: YuvReader, center: ImagePoint, radiusPx: Int): Float {
        var best = 0
        val cx = center.x.roundToInt()
        val cy = center.y.roundToInt()
        for (i in SAMPLE_X.indices) {
            val x = cx + (SAMPLE_X[i] * radiusPx).roundToInt()
            val y = cy + (SAMPLE_Y[i] * radiusPx).roundToInt()
            if (x !in 0 until reader.width || y !in 0 until reader.height) continue
            best = max(best, reader.y(x, y))
        }
        return best.toFloat()
    }

    private fun ImagePoint.local(model: PatternModel, along: Float, normal: Float, scale: Float): ImagePoint {
        return ImagePoint(
            x = x + model.ux * along * scale + model.vx * normal * scale,
            y = y + model.uy * along * scale + model.vy * normal * scale,
        )
    }

    private fun sampleRadius(sizePx: Float): Int {
        return (sizePx * 0.085f).roundToInt().coerceIn(1, 4)
    }

    private fun markerMean(
        reader: YuvReader,
        model: PatternModel,
        origin: ImagePoint,
        samples: List<LocalPoint>,
        size: Float,
        radius: Int,
    ): Float {
        if (samples.isEmpty()) return 0f
        var sum = 0f
        for (sample in samples) {
            sum += markerValue(reader, origin.local(model, sample.x, sample.y, size), radius)
        }
        return sum / samples.size
    }

    private fun markerMin(
        reader: YuvReader,
        model: PatternModel,
        origin: ImagePoint,
        samples: List<LocalPoint>,
        size: Float,
        radius: Int,
    ): Float {
        if (samples.isEmpty()) return 0f
        var best = Float.POSITIVE_INFINITY
        for (sample in samples) {
            best = min(best, markerValue(reader, origin.local(model, sample.x, sample.y, size), radius))
        }
        return best
    }

    private fun normalizeAngle(angle: Float): Float {
        var a = angle
        val twoPi = (2.0 * PI).toFloat()
        while (a <= -PI.toFloat()) a += twoPi
        while (a > PI.toFloat()) a -= twoPi
        return a
    }

    private data class LocalPoint(val x: Float, val y: Float)

    private data class Step(val translationPx: Float, val angleRad: Float, val logDistance: Float)

    private class YuvReader(image: ImageProxy) {
        val width: Int = image.width
        val height: Int = image.height
        val timestampNs: Long = image.imageInfo.timestamp
        val rotationDegrees: Int = image.imageInfo.rotationDegrees
        val cropLeft: Int = image.cropRect.left
        val cropTop: Int = image.cropRect.top
        val cropWidth: Int = image.cropRect.width()
        val cropHeight: Int = image.cropRect.height()

        private val yBuffer: ByteBuffer = image.planes[0].buffer
        private val uBuffer: ByteBuffer = image.planes[1].buffer
        private val vBuffer: ByteBuffer = image.planes[2].buffer
        private val yRowStride: Int = image.planes[0].rowStride
        private val uRowStride: Int = image.planes[1].rowStride
        private val vRowStride: Int = image.planes[2].rowStride
        private val uPixelStride: Int = image.planes[1].pixelStride
        private val vPixelStride: Int = image.planes[2].pixelStride

        fun rotatedRoi(): RotatedRoi {
            val rotatedWidth = when (rotationDegrees) {
                90, 270 -> cropHeight.toFloat()
                else -> cropWidth.toFloat()
            }
            val rotatedHeight = when (rotationDegrees) {
                90, 270 -> cropWidth.toFloat()
                else -> cropHeight.toFloat()
            }
            val roiWidth = rotatedWidth * ReaderRoi.WIDTH_FRACTION
            val roiHeight = (roiWidth * ReaderRoi.ROI_ASPECT_RATIO).coerceAtMost(rotatedHeight)
            return RotatedRoi(
                left = (rotatedWidth - roiWidth) * 0.5f,
                top = (rotatedHeight - roiHeight) * 0.5f,
                width = roiWidth,
                height = roiHeight,
            )
        }

        fun imageToRotated(point: ImagePoint): ImagePoint {
            val x = point.x - cropLeft
            val y = point.y - cropTop
            return when (rotationDegrees) {
                90 -> ImagePoint(cropHeight - y, x)
                180 -> ImagePoint(cropWidth - x, cropHeight - y)
                270 -> ImagePoint(y, cropWidth - x)
                else -> ImagePoint(x, y)
            }
        }

        fun rotatedToImage(x: Float, y: Float): ImagePoint {
            val point = when (rotationDegrees) {
                90 -> ImagePoint(y, cropHeight - x)
                180 -> ImagePoint(cropWidth - x, cropHeight - y)
                270 -> ImagePoint(cropWidth - y, x)
                else -> ImagePoint(x, y)
            }
            return ImagePoint(
                x = point.x + cropLeft,
                y = point.y + cropTop,
            )
        }

        fun y(x: Int, y: Int): Int {
            return yBuffer.get(y * yRowStride + x).toInt() and 0xff
        }

        fun u(x: Int, y: Int): Int {
            return uBuffer.get((y / 2) * uRowStride + (x / 2) * uPixelStride).toInt() and 0xff
        }

        fun v(x: Int, y: Int): Int {
            return vBuffer.get((y / 2) * vRowStride + (x / 2) * vPixelStride).toInt() and 0xff
        }

        fun blueDominance(x: Int, y: Int): Float {
            val yy = this.y(x, y)
            return (u(x, y) - v(x, y)) + (yy - 45) * 0.12f
        }
    }

    private class GeometryConstants {
        private val ledMm = 3f
        private val gapMm = 2.5f
        private val markerMm = 4f
        private val squareMm = ledMm + 0.45f
        private val triangleMm = ledMm + 2f
        private val markerGapMm = 4f
        private val stepMm = ledMm + gapMm
        private val markerDistanceMm = markerMm + markerGapMm * 2 + ledMm + stepMm * 4
        private val firstLedOffsetMm = markerMm / 2f + markerGapMm + ledMm / 2f

        val slotFractions: FloatArray = FloatArray(5) { index -> (firstLedOffsetMm + index * stepMm) / markerDistanceMm }

        fun markerDistanceToSizeRatio(): Float = markerDistanceMm / markerMm

        fun ledRadiusToMarkerSizeRatio(): Float = (ledMm * 0.5f) / markerMm

        fun squareSizeToMarkerSizeRatio(): Float = squareMm / markerMm

        fun triangleSizeToMarkerSizeRatio(): Float = triangleMm / markerMm
    }

    private companion object {
        const val RESET_AFTER_MISSES = 4
        const val TRACKING_CONTINUITY_MISSES = 1
        const val MIN_ACCEPT_SCORE = 6.0f
        const val INITIAL_PATTERN_DISTANCE_FRACTION = 0.82f
        const val MIN_PATTERN_DISTANCE_PX = 32f
        const val MIN_ACCEPT_SQUARE = 0.62f
        const val MIN_ACCEPT_TRIANGLE = 0.38f
        const val BAD_SCORE = -1_000_000f

        val ACQUIRE_STEPS = arrayOf(
            Step(14f, (5.2f * PI / 180.0).toFloat(), 0.060f),
            Step(8f, (3.0f * PI / 180.0).toFloat(), 0.034f),
            Step(4.5f, (1.7f * PI / 180.0).toFloat(), 0.020f),
            Step(2.4f, (0.9f * PI / 180.0).toFloat(), 0.012f),
            Step(1.2f, (0.45f * PI / 180.0).toFloat(), 0.006f),
            Step(0.65f, (0.24f * PI / 180.0).toFloat(), 0.003f),
        )
        val TRACKING_STEPS = arrayOf(
            Step(5.5f, (2.0f * PI / 180.0).toFloat(), 0.024f),
            Step(3.0f, (1.1f * PI / 180.0).toFloat(), 0.014f),
            Step(1.6f, (0.55f * PI / 180.0).toFloat(), 0.007f),
            Step(0.8f, (0.28f * PI / 180.0).toFloat(), 0.0035f),
        )

        val squareInsideSamples = listOf(
            LocalPoint(0f, 0f),
            LocalPoint(-0.24f, -0.24f),
            LocalPoint(0.24f, -0.24f),
            LocalPoint(-0.24f, 0.24f),
            LocalPoint(0.24f, 0.24f),
            LocalPoint(-0.34f, 0f),
            LocalPoint(0.34f, 0f),
            LocalPoint(0f, -0.34f),
            LocalPoint(0f, 0.34f),
        )
        val squareCornerSamples = listOf(
            LocalPoint(-0.34f, -0.34f),
            LocalPoint(0.34f, -0.34f),
            LocalPoint(-0.34f, 0.34f),
            LocalPoint(0.34f, 0.34f),
        )
        val squareOutsideSamples = listOf(
            LocalPoint(-0.64f, 0f),
            LocalPoint(0.64f, 0f),
            LocalPoint(0f, -0.64f),
            LocalPoint(0f, 0.64f),
            LocalPoint(-0.58f, -0.58f),
            LocalPoint(0.58f, -0.58f),
            LocalPoint(-0.58f, 0.58f),
            LocalPoint(0.58f, 0.58f),
        )
        val triangleInsideSamples = listOf(
            LocalPoint(0f, -0.32f),
            LocalPoint(0f, -0.06f),
            LocalPoint(0f, 0.14f),
            LocalPoint(-0.16f, 0.22f),
            LocalPoint(0.16f, 0.22f),
            LocalPoint(-0.26f, 0.34f),
            LocalPoint(0.26f, 0.34f),
        )
        val triangleBaseSamples = listOf(
            LocalPoint(-0.32f, 0.34f),
            LocalPoint(0f, 0.34f),
            LocalPoint(0.32f, 0.34f),
        )
        val triangleUpperEmptySamples = listOf(
            LocalPoint(-0.34f, -0.18f),
            LocalPoint(0.34f, -0.18f),
            LocalPoint(-0.44f, 0.02f),
            LocalPoint(0.44f, 0.02f),
        )
        val triangleOutsideSamples = listOf(
            LocalPoint(-0.58f, 0.06f),
            LocalPoint(0.58f, 0.06f),
            LocalPoint(-0.34f, -0.42f),
            LocalPoint(0.34f, -0.42f),
            LocalPoint(0f, -0.66f),
            LocalPoint(0f, 0.66f),
        )
        val SAMPLE_X = floatArrayOf(0f, 1f, -1f, 0f, 0f, 0.70f, -0.70f, 0.70f, -0.70f, 0.45f, -0.45f, 0f, 0f)
        val SAMPLE_Y = floatArrayOf(0f, 0f, 0f, 1f, -1f, 0.70f, 0.70f, -0.70f, -0.70f, 0f, 0f, 0.45f, -0.45f)
        val SAMPLE_W = floatArrayOf(0.24f, 0.09f, 0.09f, 0.09f, 0.09f, 0.06f, 0.06f, 0.06f, 0.06f, 0.04f, 0.04f, 0.04f, 0.04f)
    }
}
