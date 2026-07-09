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

    private val constants = GeometryConstants()
    private val neural = NeuralVisionModels(context.applicationContext, constants)

    var lastDebugLines: List<String> = emptyList()
        private set
    var isAcquireMode: Boolean = true
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
        val searchArea = reader.rotatedSearchArea()
        beginDebugFrame()

        val seed = initialTheta(reader, searchArea)
        val tracking = previousTheta != null && missedFrames <= TRACKING_CONTINUITY_MISSES
        isAcquireMode = !tracking
        if (Diagnostics.enabled) {
            debugModeLine = if (tracking) {
                "mode: tracking prevScore=${fmt(previousScore)} missed=$missedFrames"
            } else {
                "mode: acquire centered"
            }
        }

        var fit = refine(
            reader = reader,
            searchArea = searchArea,
            seed = seed,
            steps = if (tracking) TRACKING_STEPS else ACQUIRE_STEPS,
            trackingAcceptance = tracking,
        )
        if (tracking && !isAccepted(fit, tracking = true)) {
            isAcquireMode = true
            if (Diagnostics.enabled) {
                debugModeLine = "mode: tracking fallback prevScore=${fmt(previousScore)} missed=$missedFrames"
            }
            fit = refine(
                reader = reader,
                searchArea = searchArea,
                seed = centeredTheta(reader, searchArea),
                steps = ACQUIRE_STEPS,
                trackingAcceptance = false,
            )
        }
        if (!isAccepted(fit, tracking = !isAcquireMode)) {
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

        val model = modelForTheta(fit.theta)
        val patternConfidence = fit.score.coerceIn(0f, 1f)
        val scores = readLedScores(reader, model, patternConfidence)
        updateLedDebugLine(scores, patternConfidence)
        val debugLines = finishDebugFrame("HIT", fit)
        return frameForModel(reader, model, scores, patternConfidence, debugLines)
    }

    fun resetTracking() {
        previousTheta = null
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
        lastDebugLines = lines
        return lines
    }

    private fun fmt(value: Float): String = String.format(Locale.US, "%.2f", value)

    private fun debugBestLine(): String {
        if (debugBestScore == BAD_SCORE) return "best: none"
        return "best score=${fmt(debugBestScore)} sq=${fmt(debugBestSquare)} tri=${fmt(debugBestTriangle)}" +
            " d=${fmt(debugBestDistanceInRoi)}"
    }

    private fun initialTheta(reader: YuvReader, searchArea: RotatedSearchArea): Theta {
        val previous = previousTheta
        if (previous != null && missedFrames <= TRACKING_CONTINUITY_MISSES) return previous
        return centeredTheta(reader, searchArea)
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
        seed: Theta,
        steps: Array<Step>,
        trackingAcceptance: Boolean,
    ): Fit {
        val guideWidth = reader.guideWidth()
        val minLogDistance = ln(max(MIN_PATTERN_DISTANCE_PX, guideWidth * 0.58f))
        val maxLogDistance = ln(guideWidth * 1.02f)
        var theta = normalizeTheta(seed, minLogDistance, maxLogDistance)
        var breakdown = scoreBreakdown(reader, searchArea, theta)
        var score = breakdown.score
        var bestTheta = theta
        var bestBreakdown = breakdown
        var bestScore = score

        for (step in steps) {
            var localBestTheta = theta
            var localBestBreakdown = breakdown
            var localBestScore = score

            fun tryCandidate(candidate: Theta) {
                val normalized = normalizeTheta(candidate, minLogDistance, maxLogDistance)
                val candidateBreakdown = scoreBreakdown(reader, searchArea, normalized)
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
            if (!improved && isAccepted(bestScore, tracking = trackingAcceptance)) break
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

    private fun scoreBreakdown(reader: YuvReader, searchArea: RotatedSearchArea, theta: Theta): ScoreBreakdown {
        val model = modelForTheta(theta)
        if (!modelInsideSearchArea(reader, searchArea, model)) {
            return ScoreBreakdown(BAD_SCORE, 0f, 0f)
        }

        val likelihood = neural.trackerLikelihood(reader, model)
        val square = likelihood
        val triangle = likelihood
        val markerDistanceInGuide = model.distancePx / reader.guideWidth()

        val score = likelihood

        if (Diagnostics.enabled && score > debugBestScore) {
            debugBestScore = score
            debugBestSquare = square
            debugBestTriangle = triangle
            debugBestDistanceInRoi = markerDistanceInGuide
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

        fun y(x: Int, y: Int): Int {
            return yBuffer.get(y * yRowStride + x).toInt() and 0xff
        }

        fun u(x: Int, y: Int): Int {
            return uBuffer.get((y / 2) * uRowStride + (x / 2) * uPixelStride).toInt() and 0xff
        }

        fun v(x: Int, y: Int): Int {
            return vBuffer.get((y / 2) * vRowStride + (x / 2) * vPixelStride).toInt() and 0xff
        }

        fun yBilinear(x: Float, y: Float): Float {
            return bilinear(
                x = x,
                y = y,
                maxX = width - 1,
                maxY = height - 1,
            ) { xi, yi -> this.y(xi, yi).toFloat() }
        }

        fun uBilinear(x: Float, y: Float): Float {
            return bilinear(
                x = x,
                y = y,
                maxX = width - 1,
                maxY = height - 1,
            ) { xi, yi -> this.u(xi, yi).toFloat() }
        }

        fun vBilinear(x: Float, y: Float): Float {
            return bilinear(
                x = x,
                y = y,
                maxX = width - 1,
                maxY = height - 1,
            ) { xi, yi -> this.v(xi, yi).toFloat() }
        }

        private fun bilinear(
            x: Float,
            y: Float,
            maxX: Int,
            maxY: Int,
            sample: (Int, Int) -> Float,
        ): Float {
            val clampedX = x.coerceIn(0f, maxX.toFloat())
            val clampedY = y.coerceIn(0f, maxY.toFloat())
            val x0 = clampedX.toInt().coerceIn(0, maxX)
            val y0 = clampedY.toInt().coerceIn(0, maxY)
            val x1 = (x0 + 1).coerceAtMost(maxX)
            val y1 = (y0 + 1).coerceAtMost(maxY)
            val fx = clampedX - x0
            val fy = clampedY - y0
            val top = sample(x0, y0) * (1f - fx) + sample(x1, y0) * fx
            val bottom = sample(x0, y1) * (1f - fx) + sample(x1, y1) * fx
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
        private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
        private val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            setIntraOpNumThreads(1)
        }
        private val trackerSession: OrtSession = env.createSession(
            context.assets.open(TRACKER_MODEL_ASSET).use { it.readBytes() },
            sessionOptions,
        )
        private val ledSession: OrtSession = env.createSession(
            context.assets.open(LED_MODEL_ASSET).use { it.readBytes() },
            sessionOptions,
        )

        fun trackerLikelihood(reader: YuvReader, model: PatternModel): Float {
            val input = markerTensor(reader, model)
            OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 2, TRACKER_PATCH_H.toLong(), TRACKER_PATCH_W.toLong())).use { tensor ->
                trackerSession.run(mapOf("patch" to tensor)).use { result ->
                    val logit = firstFloat(result[0].value)
                    return sigmoid(logit)
                }
            }
        }

        fun ledScores(reader: YuvReader, model: PatternModel, detectorLikelihood: Float): FloatArray {
            val input = ledTensor(reader, model)
            val likelihood = FloatArray(LED_COUNT) { detectorLikelihood.coerceIn(0f, 1f) }
            OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(LED_COUNT.toLong(), 3, LED_PATCH.toLong(), LED_PATCH.toLong())).use { cropTensor ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(likelihood), longArrayOf(LED_COUNT.toLong())).use { likelihoodTensor ->
                    ledSession.run(
                        mapOf(
                            "led_crop" to cropTensor,
                            "detector_likelihood" to likelihoodTensor,
                        ),
                    ).use { result ->
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

        private fun markerTensor(reader: YuvReader, model: PatternModel): FloatArray {
            val count = TRACKER_PATCH_W * TRACKER_PATCH_H
            val luma = FloatArray(count)
            for (y in 0 until TRACKER_PATCH_H) {
                val localY = ((y.toFloat() / (TRACKER_PATCH_H - 1)) - 0.5f) * MARKER_PATCH_LOCAL_H
                for (x in 0 until TRACKER_PATCH_W) {
                    val localX = ((x.toFloat() / (TRACKER_PATCH_W - 1)) - 0.5f) * MARKER_PATCH_LOCAL_W
                    luma[y * TRACKER_PATCH_W + x] = sampleLuma(reader, model, localX, localY) / 255f
                }
            }
            val normalized = normalizedLuma(luma)
            val edge = edgeChannel(luma, TRACKER_PATCH_W, TRACKER_PATCH_H)
            val out = FloatArray(2 * count)
            System.arraycopy(normalized, 0, out, 0, count)
            System.arraycopy(edge, 0, out, count, count)
            return out
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
            var sum = 0f
            for (value in luma) sum += value
            val mean = sum / luma.size
            var variance = 0f
            for (value in luma) {
                val d = value - mean
                variance += d * d
            }
            val std = sqrt(variance / luma.size).coerceAtLeast(1e-4f)
            return FloatArray(luma.size) { index -> ((luma[index] - mean) / std).coerceIn(-3f, 3f) / 3f }
        }

        private fun edgeChannel(luma: FloatArray, width: Int, height: Int): FloatArray {
            val edge = FloatArray(luma.size)
            for (y in 0 until height) {
                val ym = (y - 1).coerceAtLeast(0)
                val yp = (y + 1).coerceAtMost(height - 1)
                for (x in 0 until width) {
                    val xm = (x - 1).coerceAtLeast(0)
                    val xp = (x + 1).coerceAtMost(width - 1)
                    val gx = luma[y * width + xp] - luma[y * width + xm]
                    val gy = luma[yp * width + x] - luma[ym * width + x]
                    edge[y * width + x] = sqrt(gx * gx + gy * gy)
                }
            }
            val sorted = edge.copyOf()
            sorted.sort()
            val p95 = sorted[(sorted.size * 95 / 100).coerceIn(0, sorted.lastIndex)].coerceAtLeast(1e-4f)
            for (index in edge.indices) edge[index] = (edge[index] / p95).coerceIn(0f, 1f)
            return edge
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

        private fun firstFloat(value: Any): Float {
            return floats(value, 1)[0]
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

        private data class RgbPixel(val r: Float, val g: Float, val b: Float, val luma: Float)
    }

    private companion object {
        const val RESET_AFTER_MISSES = 4
        const val TRACKING_CONTINUITY_MISSES = 1
        const val MIN_TRACKING_ACCEPT_SCORE = 0.56f
        const val MIN_ACQUIRE_ACCEPT_SCORE = 0.72f
        const val INITIAL_PATTERN_DISTANCE_FRACTION = 0.82f
        const val MIN_PATTERN_DISTANCE_PX = 32f
        const val BAD_SCORE = -1_000_000f
        const val TRACKER_MODEL_ASSET = "tracker_likelihood.onnx"
        const val LED_MODEL_ASSET = "led_reader.onnx"
        const val TRACKER_PATCH_W = 96
        const val TRACKER_PATCH_H = 36
        const val LED_PATCH = 28
        const val LED_COUNT = 5
        const val MARKER_PATCH_LOCAL_W = 1.35f
        const val MARKER_PATCH_LOCAL_H = 0.46f
        const val LED_MARKER_PATCH_W = 160f
        const val LED_MARKER_PATCH_H = 64f
        const val LED_CROP_SOURCE_SIDE = 29f

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
    }
}
