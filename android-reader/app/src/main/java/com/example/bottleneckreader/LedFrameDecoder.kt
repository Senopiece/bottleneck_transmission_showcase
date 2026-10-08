package com.example.bottleneckreader

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.Locale
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.sin

class LedFrameDecoder(context: Context) {
    private data class RotatedSearchArea(
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

    private enum class TrackerModelKind {
        ACQUIRE,
        PRECISE,
    }

    private val constants = GeometryConstants()
    private val neural = NeuralVisionModels(context.applicationContext, constants)

    var lastDebugLines: List<String> = emptyList()
        private set
    var isAcquireMode: Boolean = true
        private set

    private var previousTheta: Theta? = null
    private var previousPreviousTheta: Theta? = null
    private var previousTimestampNs = 0L
    private var previousPreviousTimestampNs = 0L
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
        val searchArea = reader.rotatedSearchArea()
        beginDebugFrame()
        neural.beginDebugFrame()

        val tracking = previousTheta != null && missedFrames <= TRACKING_CONTINUITY_MISSES
        isAcquireMode = !tracking
        if (Diagnostics.enabled) {
            debugModeLine = if (tracking) {
                "mode: tracking prevScore=${fmt(previousScore)} missed=$missedFrames"
            } else {
                "mode: acquire centered"
            }
        }

        var usedAcquire = !tracking
        var fit = if (tracking) refine(
            reader = reader,
            searchArea = searchArea,
            seeds = trackingSeeds(reader.timestampNs),
            steps = TRACKING_STEPS,
            modelKind = TrackerModelKind.PRECISE,
        ) else acquireAndVerify(reader, searchArea)
        if (tracking && !isAccepted(fit, tracking = true)) {
            isAcquireMode = true
            usedAcquire = true
            if (Diagnostics.enabled) {
                debugModeLine = "mode: tracking fallback prevScore=${fmt(previousScore)} missed=$missedFrames"
            }
            fit = acquireAndVerify(reader, searchArea)
        }
        if (!isAccepted(fit, tracking = !isAcquireMode)) {
            missedFrames++
            if (missedFrames >= RESET_AFTER_MISSES) {
                previousTheta = null
                previousPreviousTheta = null
                previousTimestampNs = 0L
                previousPreviousTimestampNs = 0L
                previousScore = 0f
            }
            finishDebugFrame("MISS", fit)
            return null
        }

        if (usedAcquire) {
            previousPreviousTheta = null
            previousPreviousTimestampNs = 0L
        } else {
            previousPreviousTheta = previousTheta
            previousPreviousTimestampNs = previousTimestampNs
        }
        previousTheta = fit.theta
        previousTimestampNs = reader.timestampNs
        previousScore = fit.score
        missedFrames = 0

        val model = modelForTheta(fit.theta)
        val patternConfidence = fit.score.coerceIn(0f, 1f)
        val scores = readLedScores(reader, model, patternConfidence)
        updateLedDebugLine(scores, patternConfidence)
        val debugLines = finishDebugFrame("HIT", fit)
        return frameForModel(reader, model, scores, patternConfidence, debugLines)
    }

    fun resetTracking() {
        previousTheta = null
        previousPreviousTheta = null
        previousTimestampNs = 0L
        previousPreviousTimestampNs = 0L
        previousScore = 0f
        missedFrames = 0
        lastDebugLines = emptyList()
        isAcquireMode = true
    }

    private fun beginDebugFrame() {
        if (!Diagnostics.enabled) return
        debugModeLine = "mode: none"
        debugBitsLine = "ledScore: none"
        debugBestScore = BAD_SCORE
        debugBestSquare = 0f
        debugBestTriangle = 0f
        debugBestDistanceInRoi = 0f
    }

    private fun finishDebugFrame(status: String, fit: Fit?): List<String> {
        if (!Diagnostics.enabled) {
            lastDebugLines = emptyList()
            return emptyList()
        }
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
        lines.add(neural.debugTimingLine())
        lastDebugLines = lines
        return lines
    }

    private fun fmt(value: Float): String = String.format(Locale.US, "%.2f", value)

    private fun debugBestLine(): String {
        if (debugBestScore == BAD_SCORE) return "best: none"
        return "best score=${fmt(debugBestScore)} sq=${fmt(debugBestSquare)} tri=${fmt(debugBestTriangle)}" +
            " d=${fmt(debugBestDistanceInRoi)}"
    }

    private fun acquireAndVerify(reader: YuvReader, searchArea: RotatedSearchArea): Fit {
        val coarse = refine(
            reader = reader,
            searchArea = searchArea,
            seeds = listOf(centeredTheta(reader, searchArea)),
            steps = ACQUIRE_STEPS,
            modelKind = TrackerModelKind.ACQUIRE,
        )
        if (coarse.score < MIN_COARSE_ACQUIRE_SCORE) return coarse
        return refine(
            reader = reader,
            searchArea = searchArea,
            seeds = listOf(coarse.theta),
            steps = VERIFY_STEPS,
            modelKind = TrackerModelKind.PRECISE,
        )
    }

    private fun trackingSeeds(timestampNs: Long): List<Theta> {
        val previous = requireNotNull(previousTheta)
        val older = previousPreviousTheta ?: return listOf(previous)
        val historyDt = previousTimestampNs - previousPreviousTimestampNs
        val predictionDt = timestampNs - previousTimestampNs
        if (historyDt <= 0L || predictionDt <= 0L) return listOf(previous)
        val ratio = (predictionDt.toDouble() / historyDt.toDouble()).toFloat().coerceIn(0f, 2.2f)

        var dx = (previous.cx - older.cx) * ratio
        var dy = (previous.cy - older.cy) * ratio
        val maxTranslation = previous.distance * MAX_PREDICTED_TRANSLATION_FRACTION
        val translation = sqrt(dx * dx + dy * dy)
        if (translation > maxTranslation && translation > 1e-4f) {
            val scale = maxTranslation / translation
            dx *= scale
            dy *= scale
        }
        val angleDelta = (
            normalizeAngle(previous.angle - older.angle) * ratio
            ).coerceIn(-MAX_PREDICTED_ANGLE_RAD, MAX_PREDICTED_ANGLE_RAD)
        val logDistanceDelta = (
            (previous.logDistance - older.logDistance) * ratio
            ).coerceIn(-MAX_PREDICTED_LOG_DISTANCE, MAX_PREDICTED_LOG_DISTANCE)
        val predicted = previous.copy(
            cx = previous.cx + dx,
            cy = previous.cy + dy,
            angle = previous.angle + angleDelta,
            logDistance = previous.logDistance + logDistanceDelta,
        )
        return listOf(previous, predicted)
    }

    private fun centeredTheta(reader: YuvReader, searchArea: RotatedSearchArea): Theta {
        val rotatedCenterX = searchArea.left + searchArea.width * 0.5f
        val rotatedCenterY = searchArea.top + searchArea.height * 0.5f
        val center = reader.rotatedToImage(rotatedCenterX, rotatedCenterY)
        val p1 = reader.rotatedToImage(rotatedCenterX + 1f, rotatedCenterY)
        val angle = atan2(p1.y - center.y, p1.x - center.x)
        val distance = (reader.guideWidth() * INITIAL_PATTERN_DISTANCE_FRACTION)
            .coerceAtLeast(MIN_PATTERN_DISTANCE_PX)
        return Theta(
            cx = center.x,
            cy = center.y,
            angle = angle,
            logDistance = ln(distance),
        )
    }

    private fun refine(
        reader: YuvReader,
        searchArea: RotatedSearchArea,
        seeds: List<Theta>,
        steps: Array<Step>,
        modelKind: TrackerModelKind,
    ): Fit {
        val guideWidth = reader.guideWidth()
        val minLogDistance = ln(max(MIN_PATTERN_DISTANCE_PX, guideWidth * MIN_PATTERN_DISTANCE_FRACTION))
        val maxLogDistance = ln(guideWidth * 1.02f)
        val normalizedSeeds = seeds.map { normalizeTheta(it, minLogDistance, maxLogDistance) }
        val seedBreakdowns = scoreBreakdowns(reader, searchArea, normalizedSeeds, modelKind)
        var seedIndex = 0
        for (index in 1 until normalizedSeeds.size) {
            if (seedBreakdowns[index].score > seedBreakdowns[seedIndex].score) seedIndex = index
        }
        var theta = normalizedSeeds[seedIndex]
        var breakdown = seedBreakdowns[seedIndex]
        var score = breakdown.score
        var bestTheta = theta
        var bestBreakdown = breakdown
        var bestScore = score

        for (step in steps) {
            var pass = 0
            while (pass < MAX_PASSES_PER_STEP) {
                var localBestTheta = theta
                var localBestBreakdown = breakdown
                var localBestScore = score

                val candidates = arrayOf(
                theta.copy(cx = theta.cx + step.translationPx),
                theta.copy(cx = theta.cx - step.translationPx),
                theta.copy(cy = theta.cy + step.translationPx),
                theta.copy(cy = theta.cy - step.translationPx),
                theta.copy(angle = theta.angle + step.angleRad),
                theta.copy(angle = theta.angle - step.angleRad),
                theta.copy(logDistance = theta.logDistance + step.logDistance),
                theta.copy(logDistance = theta.logDistance - step.logDistance),
                ).map { normalizeTheta(it, minLogDistance, maxLogDistance) }
                val candidateBreakdowns = scoreBreakdowns(reader, searchArea, candidates, modelKind)
                for (index in candidates.indices) {
                    val candidateBreakdown = candidateBreakdowns[index]
                    val candidateScore = candidateBreakdown.score
                    if (candidateScore > localBestScore + MIN_ASCENT_IMPROVEMENT) {
                        localBestScore = candidateScore
                        localBestBreakdown = candidateBreakdown
                        localBestTheta = candidates[index]
                    }
                }

                val improved = localBestScore > score + MIN_ASCENT_IMPROVEMENT
                theta = localBestTheta
                breakdown = localBestBreakdown
                score = localBestScore
                if (score > bestScore) {
                    bestScore = score
                    bestTheta = theta
                    bestBreakdown = breakdown
                }
                if (!improved) break
                pass++
            }
        }

        return Fit(bestTheta, bestBreakdown)
    }

    private fun normalizeTheta(theta: Theta, minLogDistance: Float, maxLogDistance: Float): Theta {
        return theta.copy(
            angle = normalizeAngle(theta.angle),
            logDistance = theta.logDistance.coerceIn(minLogDistance, maxLogDistance),
        )
    }

    private fun isAccepted(fit: Fit, tracking: Boolean): Boolean {
        return isAccepted(fit.score, tracking)
    }

    private fun isAccepted(score: Float, tracking: Boolean): Boolean {
        val minScore = acceptScoreThreshold(tracking)
        return score >= minScore
    }

    private fun scoreBreakdowns(
        reader: YuvReader,
        searchArea: RotatedSearchArea,
        thetas: List<Theta>,
        modelKind: TrackerModelKind,
    ): Array<ScoreBreakdown> {
        val results = Array(thetas.size) { ScoreBreakdown(BAD_SCORE, 0f, 0f) }
        val validIndices = ArrayList<Int>(thetas.size)
        val validModels = ArrayList<PatternModel>(thetas.size)
        for (index in thetas.indices) {
            val model = modelForTheta(thetas[index])
            if (modelInsideSearchArea(reader, searchArea, model)) {
                validIndices.add(index)
                validModels.add(model)
            }
        }
        if (validModels.isEmpty()) return results

        val likelihoods = neural.trackerLikelihoods(reader, validModels, modelKind)
        val guideWidth = reader.guideWidth()
        for (validIndex in validModels.indices) {
            val likelihood = likelihoods[validIndex]
            val model = validModels[validIndex]
            results[validIndices[validIndex]] = ScoreBreakdown(
                score = likelihood,
                square = likelihood,
                triangle = likelihood,
            )
            if (Diagnostics.enabled && likelihood > debugBestScore) {
                debugBestScore = likelihood
                debugBestSquare = likelihood
                debugBestTriangle = likelihood
                debugBestDistanceInRoi = model.distancePx / guideWidth
            }
        }
        return results
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

    private fun modelInsideSearchArea(reader: YuvReader, searchArea: RotatedSearchArea, model: PatternModel): Boolean {
        val margin = max(2f, model.markerSizePx * 0.58f)
        for (slot in model.slots) {
            if (!pointInsideSearchArea(reader, searchArea, slot, margin)) return false
        }
        return pointInsideSearchArea(reader, searchArea, model.start, margin) &&
            pointInsideSearchArea(reader, searchArea, model.end, margin)
    }

    private fun pointInsideSearchArea(reader: YuvReader, searchArea: RotatedSearchArea, point: ImagePoint, margin: Float): Boolean {
        if (point.x < margin || point.x >= reader.width - margin || point.y < margin || point.y >= reader.height - margin) {
            return false
        }
        val rotated = reader.imageToRotated(point)
        return rotated.x >= searchArea.left + margin &&
            rotated.x <= searchArea.left + searchArea.width - margin &&
            rotated.y >= searchArea.top + margin &&
            rotated.y <= searchArea.top + searchArea.height - margin
    }

    private fun frameForModel(
        reader: YuvReader,
        model: PatternModel,
        scores: FloatArray,
        patternConfidence: Float,
        debugLines: List<String>,
    ): DetectionFrame {
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
            ledScores = scores,
            patternConfidence = patternConfidence,
            slots = model.slots.mapIndexed { index, point ->
                LedSlot(
                    imagePoint = point,
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
            isAcquireMode = isAcquireMode,
            debugLines = debugLines,
        )
    }

    private fun readLedScores(reader: YuvReader, model: PatternModel, patternConfidence: Float): FloatArray {
        return neural.ledScores(reader, model, patternConfidence)
    }

    private fun acceptScoreThreshold(tracking: Boolean): Float {
        return if (tracking) MIN_TRACKING_ACCEPT_SCORE else MIN_ACQUIRE_ACCEPT_SCORE
    }

    private fun updateLedDebugLine(scores: FloatArray, patternConfidence: Float) {
        if (Diagnostics.enabled) {
            debugBitsLine = buildString {
                append("ledScore w=").append(fmt(patternConfidence))
                scores.forEach { append(' ').append(fmt(it)) }
            }
        }
    }

    private fun normalizeAngle(angle: Float): Float {
        var a = angle
        val twoPi = (2.0 * PI).toFloat()
        while (a <= -PI.toFloat()) a += twoPi
        while (a > PI.toFloat()) a -= twoPi
        return a
    }

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

        fun rotatedSearchArea(): RotatedSearchArea {
            val rotatedWidth = when (rotationDegrees) {
                90, 270 -> cropHeight.toFloat()
                else -> cropWidth.toFloat()
            }
            val rotatedHeight = when (rotationDegrees) {
                90, 270 -> cropWidth.toFloat()
                else -> cropHeight.toFloat()
            }
            return RotatedSearchArea(
                left = 0f,
                top = 0f,
                width = rotatedWidth,
                height = rotatedHeight,
            )
        }

        fun guideWidth(): Float {
            val rotatedWidth = when (rotationDegrees) {
                90, 270 -> cropHeight.toFloat()
                else -> cropWidth.toFloat()
            }
            return rotatedWidth * ReaderRoi.WIDTH_FRACTION
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

        fun yBilinear(x: Float, y: Float): Float {
            val clampedX = x.coerceIn(0f, (width - 1).toFloat())
            val clampedY = y.coerceIn(0f, (height - 1).toFloat())
            val x0 = clampedX.toInt()
            val y0 = clampedY.toInt()
            val x1 = (x0 + 1).coerceAtMost(width - 1)
            val y1 = (y0 + 1).coerceAtMost(height - 1)
            return interpolatePlane(
                buffer = yBuffer,
                rowStride = yRowStride,
                pixelStride = 1,
                x0 = x0,
                y0 = y0,
                x1 = x1,
                y1 = y1,
                fx = clampedX - x0,
                fy = clampedY - y0,
                subsample = false,
            )
        }

        fun uBilinear(x: Float, y: Float): Float {
            return chromaBilinear(uBuffer, uRowStride, uPixelStride, x, y)
        }

        fun vBilinear(x: Float, y: Float): Float {
            return chromaBilinear(vBuffer, vRowStride, vPixelStride, x, y)
        }

        private fun chromaBilinear(
            buffer: ByteBuffer,
            rowStride: Int,
            pixelStride: Int,
            x: Float,
            y: Float,
        ): Float {
            val clampedX = x.coerceIn(0f, (width - 1).toFloat())
            val clampedY = y.coerceIn(0f, (height - 1).toFloat())
            val x0 = clampedX.toInt()
            val y0 = clampedY.toInt()
            val x1 = (x0 + 1).coerceAtMost(width - 1)
            val y1 = (y0 + 1).coerceAtMost(height - 1)
            return interpolatePlane(
                buffer = buffer,
                rowStride = rowStride,
                pixelStride = pixelStride,
                x0 = x0,
                y0 = y0,
                x1 = x1,
                y1 = y1,
                fx = clampedX - x0,
                fy = clampedY - y0,
                subsample = true,
            )
        }

        private fun interpolatePlane(
            buffer: ByteBuffer,
            rowStride: Int,
            pixelStride: Int,
            x0: Int,
            y0: Int,
            x1: Int,
            y1: Int,
            fx: Float,
            fy: Float,
            subsample: Boolean,
        ): Float {
            val sx0 = if (subsample) x0 / 2 else x0
            val sx1 = if (subsample) x1 / 2 else x1
            val sy0 = if (subsample) y0 / 2 else y0
            val sy1 = if (subsample) y1 / 2 else y1
            val p00 = (buffer.get(sy0 * rowStride + sx0 * pixelStride).toInt() and 0xff).toFloat()
            val p10 = (buffer.get(sy0 * rowStride + sx1 * pixelStride).toInt() and 0xff).toFloat()
            val p01 = (buffer.get(sy1 * rowStride + sx0 * pixelStride).toInt() and 0xff).toFloat()
            val p11 = (buffer.get(sy1 * rowStride + sx1 * pixelStride).toInt() and 0xff).toFloat()
            val top = p00 * (1f - fx) + p10 * fx
            val bottom = p01 * (1f - fx) + p11 * fx
            return top * (1f - fy) + bottom * fy
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

    private class NeuralVisionModels(
        context: Context,
        private val constants: GeometryConstants,
    ) {
        private var trackerPrepNs = 0L
        private var trackerInferenceNs = 0L
        private var trackerCalls = 0
        private var trackerCandidates = 0
        private var ledPrepNs = 0L
        private var ledInferenceNs = 0L
        private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
        private val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(1)
        }
        private val acquireTracker = TrackerRuntime(
            modelBytes = context.assets.open(ACQUIRE_TRACKER_MODEL_ASSET).use { it.readBytes() },
            width = ACQUIRE_TRACKER_PATCH_W,
            height = ACQUIRE_TRACKER_PATCH_H,
        )
        private val preciseTracker = TrackerRuntime(
            modelBytes = context.assets.open(PRECISE_TRACKER_MODEL_ASSET).use { it.readBytes() },
            width = PRECISE_TRACKER_PATCH_W,
            height = PRECISE_TRACKER_PATCH_H,
        )
        private val ledSession: OrtSession = env.createSession(
            context.assets.open(LED_MODEL_ASSET).use { it.readBytes() },
            sessionOptions,
        )

        fun beginDebugFrame() {
            if (!Diagnostics.enabled) return
            trackerPrepNs = 0L
            trackerInferenceNs = 0L
            trackerCalls = 0
            trackerCandidates = 0
            ledPrepNs = 0L
            ledInferenceNs = 0L
        }

        fun debugTimingLine(): String {
            if (!Diagnostics.enabled) return ""
            return String.format(
                Locale.US,
                "vision trPrep=%.1f trOrt=%.1f calls=%d n=%d ledPrep=%.1f ledOrt=%.1f",
                trackerPrepNs / 1_000_000.0,
                trackerInferenceNs / 1_000_000.0,
                trackerCalls,
                trackerCandidates,
                ledPrepNs / 1_000_000.0,
                ledInferenceNs / 1_000_000.0,
            )
        }

        fun trackerLikelihoods(
            reader: YuvReader,
            models: List<PatternModel>,
            modelKind: TrackerModelKind,
        ): FloatArray {
            val runtime = if (modelKind == TrackerModelKind.ACQUIRE) acquireTracker else preciseTracker
            return runtime.likelihoods(reader, models, sharpLikelihood = modelKind == TrackerModelKind.PRECISE)
        }

        fun ledScores(reader: YuvReader, model: PatternModel, detectorLikelihood: Float): FloatArray {
            val prepStarted = if (Diagnostics.enabled) System.nanoTime() else 0L
            val input = ledTensor(reader, model)
            val likelihood = FloatArray(LED_COUNT) { detectorLikelihood.coerceIn(0f, 1f) }
            if (Diagnostics.enabled) ledPrepNs += System.nanoTime() - prepStarted
            val inferenceStarted = if (Diagnostics.enabled) System.nanoTime() else 0L
            OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(LED_COUNT.toLong(), 3, LED_PATCH.toLong(), LED_PATCH.toLong())).use { cropTensor ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(likelihood), longArrayOf(LED_COUNT.toLong())).use { likelihoodTensor ->
                    ledSession.run(
                        mapOf(
                            "led_crop" to cropTensor,
                            "detector_likelihood" to likelihoodTensor,
                        ),
                    ).use { result ->
                        if (Diagnostics.enabled) ledInferenceNs += System.nanoTime() - inferenceStarted
                        val logits = floats(result[0].value, LED_COUNT)
                        return FloatArray(LED_COUNT) { index ->
                            // PacketClockDecoder consumes the old score scale:
                            // OFF around 0.56, ON above 1.0. Keep the protocol layer unchanged.
                            0.54f + sigmoid(logits[index]) * 0.64f
                        }
                    }
                }
            }
        }

        private inner class TrackerRuntime(
            modelBytes: ByteArray,
            private val width: Int,
            private val height: Int,
        ) {
            private val session = env.createSession(modelBytes, sessionOptions)
            private val inputScratch = FloatArray(MAX_TRACKER_BATCH * 2 * width * height)
            private val lumaScratch = FloatArray(width * height)
            private val percentileScratch = FloatArray(width * height)
            private val localX = FloatArray(width * height) { index ->
                val x = index % width
                ((x.toFloat() / (width - 1)) - 0.5f) * MARKER_PATCH_LOCAL_W
            }
            private val localY = FloatArray(width * height) { index ->
                val y = index / width
                ((y.toFloat() / (height - 1)) - 0.5f) * MARKER_PATCH_LOCAL_H
            }

            fun likelihoods(
                reader: YuvReader,
                models: List<PatternModel>,
                sharpLikelihood: Boolean,
            ): FloatArray {
                require(models.isNotEmpty())
                require(models.size <= MAX_TRACKER_BATCH)
                val prepStarted = if (Diagnostics.enabled) System.nanoTime() else 0L
                val input = markerTensor(reader, models)
                if (Diagnostics.enabled) {
                    trackerPrepNs += System.nanoTime() - prepStarted
                    trackerCalls++
                    trackerCandidates += models.size
                }
                val inputSize = models.size * 2 * height * width
                val inferenceStarted = if (Diagnostics.enabled) System.nanoTime() else 0L
                OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(input, 0, inputSize),
                    longArrayOf(models.size.toLong(), 2, height.toLong(), width.toLong()),
                ).use { tensor ->
                    session.run(mapOf("patch" to tensor)).use { result ->
                        if (Diagnostics.enabled) trackerInferenceNs += System.nanoTime() - inferenceStarted
                        val logits = trackerLogits(result[0].value, models.size, sharpLikelihood)
                        return FloatArray(models.size) { index -> sigmoid(logits[index]) }
                    }
                }
            }

            private fun markerTensor(reader: YuvReader, models: List<PatternModel>): FloatArray {
                val count = width * height
                val perModel = 2 * count
                for (modelIndex in models.indices) {
                    val model = models[modelIndex]
                    for (index in 0 until count) {
                        lumaScratch[index] = sampleLuma(reader, model, localX[index], localY[index]) / 255f
                    }
                    val base = modelIndex * perModel
                    normalizedLumaInto(lumaScratch, inputScratch, base)
                    edgeChannelInto(
                        luma = lumaScratch,
                        width = width,
                        height = height,
                        out = inputScratch,
                        outOffset = base + count,
                        percentileScratch = percentileScratch,
                    )
                }
                return inputScratch
            }
        }

        private fun ledTensor(reader: YuvReader, model: PatternModel): FloatArray {
            val perCrop = LED_PATCH * LED_PATCH
            val out = FloatArray(LED_COUNT * 3 * perCrop)
            val sideX = LED_CROP_SOURCE_SIDE / LED_MARKER_PATCH_W * MARKER_PATCH_LOCAL_W
            val sideY = LED_CROP_SOURCE_SIDE / LED_MARKER_PATCH_H * MARKER_PATCH_LOCAL_H
            for (led in 0 until LED_COUNT) {
                val luma = FloatArray(perCrop)
                val blue = FloatArray(perCrop)
                val centerX = constants.slotFractions[led] - 0.5f
                for (y in 0 until LED_PATCH) {
                    val dy = ((y.toFloat() / (LED_PATCH - 1)) - 0.5f) * sideY
                    for (x in 0 until LED_PATCH) {
                        val dx = ((x.toFloat() / (LED_PATCH - 1)) - 0.5f) * sideX
                        val index = y * LED_PATCH + x
                        val pixel = sampleRgb(reader, model, centerX + dx, dy)
                        luma[index] = pixel.luma / 255f
                        blue[index] = (pixel.b - 0.5f * pixel.g - 0.5f * pixel.r).coerceIn(-1f, 1f)
                    }
                }
                val normalized = normalizedLuma(luma)
                val edge = edgeChannel(luma, LED_PATCH, LED_PATCH)
                val base = led * 3 * perCrop
                System.arraycopy(normalized, 0, out, base, perCrop)
                System.arraycopy(blue, 0, out, base + perCrop, perCrop)
                System.arraycopy(edge, 0, out, base + 2 * perCrop, perCrop)
            }
            return out
        }

        private fun normalizedLuma(luma: FloatArray): FloatArray {
            val out = FloatArray(luma.size)
            normalizedLumaInto(luma, out, 0)
            return out
        }

        private fun normalizedLumaInto(luma: FloatArray, out: FloatArray, outOffset: Int) {
            var sum = 0f
            for (value in luma) sum += value
            val mean = sum / luma.size
            var variance = 0f
            for (value in luma) {
                val d = value - mean
                variance += d * d
            }
            val std = sqrt(variance / luma.size).coerceAtLeast(1e-4f)
            for (index in luma.indices) {
                out[outOffset + index] = ((luma[index] - mean) / std).coerceIn(-3f, 3f) / 3f
            }
        }

        private fun edgeChannel(luma: FloatArray, width: Int, height: Int): FloatArray {
            val edge = FloatArray(luma.size)
            val percentileScratch = FloatArray(luma.size)
            edgeChannelInto(luma, width, height, edge, 0, percentileScratch)
            return edge
        }

        private fun edgeChannelInto(
            luma: FloatArray,
            width: Int,
            height: Int,
            out: FloatArray,
            outOffset: Int,
            percentileScratch: FloatArray,
        ) {
            for (y in 0 until height) {
                val ym = (y - 1).coerceAtLeast(0)
                val yp = (y + 1).coerceAtMost(height - 1)
                for (x in 0 until width) {
                    val xm = (x - 1).coerceAtLeast(0)
                    val xp = (x + 1).coerceAtMost(width - 1)
                    val gx = luma[y * width + xp] - luma[y * width + xm]
                    val gy = luma[yp * width + x] - luma[ym * width + x]
                    out[outOffset + y * width + x] = sqrt(gx * gx + gy * gy)
                }
            }
            for (index in luma.indices) percentileScratch[index] = out[outOffset + index]
            val p95 = selectKth(
                percentileScratch,
                (percentileScratch.size * 95 / 100).coerceIn(0, percentileScratch.lastIndex),
            ).coerceAtLeast(1e-4f)
            for (index in luma.indices) {
                out[outOffset + index] = (out[outOffset + index] / p95).coerceIn(0f, 1f)
            }
        }

        private fun selectKth(values: FloatArray, target: Int): Float {
            var left = 0
            var right = values.lastIndex
            while (left < right) {
                val pivot = values[(left + right) ushr 1]
                var i = left
                var j = right
                while (i <= j) {
                    while (values[i] < pivot) i++
                    while (values[j] > pivot) j--
                    if (i <= j) {
                        val tmp = values[i]
                        values[i] = values[j]
                        values[j] = tmp
                        i++
                        j--
                    }
                }
                when {
                    target <= j -> right = j
                    target >= i -> left = i
                    else -> return values[target]
                }
            }
            return values[left]
        }

        private fun sampleLuma(reader: YuvReader, model: PatternModel, localX: Float, localY: Float): Float {
            val x = model.cx(localX, localY)
            val y = model.cy(localX, localY)
            return reader.yBilinear(x, y)
        }

        private fun sampleRgb(reader: YuvReader, model: PatternModel, localX: Float, localY: Float): RgbPixel {
            val x = model.cx(localX, localY)
            val y = model.cy(localX, localY)
            val yy = reader.yBilinear(x, y)
            val uu = reader.uBilinear(x, y) - 128f
            val vv = reader.vBilinear(x, y) - 128f
            val r = ((yy + 1.402f * vv) / 255f).coerceIn(0f, 1f)
            val g = ((yy - 0.344136f * uu - 0.714136f * vv) / 255f).coerceIn(0f, 1f)
            val b = ((yy + 1.772f * uu) / 255f).coerceIn(0f, 1f)
            return RgbPixel(r = r, g = g, b = b, luma = yy)
        }

        private fun PatternModel.cx(localX: Float, localY: Float): Float {
            return start.x + ux * distancePx * (localX + 0.5f) + vx * distancePx * localY
        }

        private fun PatternModel.cy(localX: Float, localY: Float): Float {
            return start.y + uy * distancePx * (localX + 0.5f) + vy * distancePx * localY
        }

        private fun sigmoid(value: Float): Float {
            val clamped = value.coerceIn(-40f, 40f)
            return (1f / (1f + exp(-clamped)))
        }

        private fun floats(value: Any, expected: Int): FloatArray {
            return when (value) {
                is FloatArray -> value.copyOf(expected)
                is Array<*> -> {
                    if (value.isNotEmpty() && value[0] is FloatArray) {
                        (value[0] as FloatArray).copyOf(expected)
                    } else {
                        FloatArray(expected) { index -> (value[index] as Number).toFloat() }
                    }
                }
                else -> FloatArray(expected) { 0f }
            }
        }

        private fun trackerLogits(value: Any, batchSize: Int, sharpLikelihood: Boolean): FloatArray {
            val head = if (sharpLikelihood) 1 else 0
            return when (value) {
                is FloatArray -> {
                    val heads = (value.size / batchSize).coerceAtLeast(1)
                    FloatArray(batchSize) { index -> value[index * heads + head.coerceAtMost(heads - 1)] }
                }
                is Array<*> -> FloatArray(batchSize) { index ->
                    when (val row = value[index]) {
                        is FloatArray -> row[head.coerceAtMost(row.lastIndex)]
                        is Array<*> -> (row[head.coerceAtMost(row.lastIndex)] as Number).toFloat()
                        is Number -> row.toFloat()
                        else -> 0f
                    }
                }
                else -> FloatArray(batchSize)
            }
        }

        private data class RgbPixel(val r: Float, val g: Float, val b: Float, val luma: Float)

        private companion object {
            const val MAX_TRACKER_BATCH = 8
        }
    }

    private companion object {
        const val RESET_AFTER_MISSES = 4
        const val TRACKING_CONTINUITY_MISSES = RESET_AFTER_MISSES - 1
        const val MAX_PASSES_PER_STEP = 2
        const val MIN_ASCENT_IMPROVEMENT = 0.0015f
        const val MAX_PREDICTED_TRANSLATION_FRACTION = 0.12f
        const val MAX_PREDICTED_ANGLE_RAD = 0.12f
        const val MAX_PREDICTED_LOG_DISTANCE = 0.08f
        const val MIN_TRACKING_ACCEPT_SCORE = 0.56f
        const val MIN_ACQUIRE_ACCEPT_SCORE = 0.56f
        const val MIN_COARSE_ACQUIRE_SCORE = 0.30f
        const val INITIAL_PATTERN_DISTANCE_FRACTION = 0.82f
        const val MIN_PATTERN_DISTANCE_PX = 32f
        const val MIN_PATTERN_DISTANCE_FRACTION = 0.30f
        const val BAD_SCORE = -1_000_000f
        const val ACQUIRE_TRACKER_MODEL_ASSET = "tracker_acquire.onnx"
        const val PRECISE_TRACKER_MODEL_ASSET = "tracker_precise.onnx"
        const val LED_MODEL_ASSET = "led_reader.onnx"
        const val ACQUIRE_TRACKER_PATCH_W = 64
        const val ACQUIRE_TRACKER_PATCH_H = 24
        const val PRECISE_TRACKER_PATCH_W = 96
        const val PRECISE_TRACKER_PATCH_H = 36
        const val LED_PATCH = 28
        const val LED_COUNT = 5
        const val MARKER_PATCH_LOCAL_W = 1.35f
        const val MARKER_PATCH_LOCAL_H = 0.46f
        const val LED_MARKER_PATCH_W = 160f
        const val LED_MARKER_PATCH_H = 64f
        const val LED_CROP_SOURCE_SIDE = 29f

        val ACQUIRE_STEPS = arrayOf(
            Step(32f, (5.2f * PI / 180.0).toFloat(), 0.120f),
            Step(18f, (3.0f * PI / 180.0).toFloat(), 0.075f),
            Step(10f, (1.7f * PI / 180.0).toFloat(), 0.045f),
            Step(5.5f, (0.9f * PI / 180.0).toFloat(), 0.025f),
            Step(3f, (0.45f * PI / 180.0).toFloat(), 0.012f),
            Step(1.5f, (0.24f * PI / 180.0).toFloat(), 0.006f),
        )
        val TRACKING_STEPS = arrayOf(
            Step(9f, (3.0f * PI / 180.0).toFloat(), 0.035f),
            Step(4.5f, (1.5f * PI / 180.0).toFloat(), 0.018f),
            Step(2.2f, (0.7f * PI / 180.0).toFloat(), 0.008f),
            Step(1f, (0.3f * PI / 180.0).toFloat(), 0.0035f),
        )
        val VERIFY_STEPS = arrayOf(
            Step(3.0f, (1.1f * PI / 180.0).toFloat(), 0.014f),
            Step(1.5f, (0.55f * PI / 180.0).toFloat(), 0.007f),
            Step(0.7f, (0.25f * PI / 180.0).toFloat(), 0.003f),
        )
    }
}
