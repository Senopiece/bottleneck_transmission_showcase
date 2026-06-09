package com.example.bottleneckreader

import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.util.Locale
import java.util.Random
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

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
        val scale: Float,
    )

    private data class PatternModel(
        val start: ImagePoint,
        val end: ImagePoint,
        val slots: List<ImagePoint>,
        val markerSizePx: Float,
        val squareSizePx: Float,
        val triangleSizePx: Float,
        val ledRadiusPx: Float,
        val slotStepPx: Float,
        val ux: Float,
        val uy: Float,
        val vx: Float,
        val vy: Float,
    )

    private data class Fit(
        val theta: Theta,
        val score: Float,
        val confidence: Float,
    )

    private data class NoiseLevel(
        val translationPx: Float,
        val angleRad: Float,
        val scaleRatio: Float,
        val temperature: Float,
        val alpha: Float,
        val samples: Int,
        val steps: Int,
    )

    private data class LedBitMeasurement(
        val isOn: Boolean,
        val confidence: Float,
    )

    private val constants = GeometryConstants()
    private val rng = Random(0x51a7eEDL)

    var lastDebugLines: List<String> = emptyList()
        private set

    private var previousTheta: Theta? = null
    private var previousConfidence = 0f
    private var missedFrames = 0
    private var pendingBits: String? = null
    private var pendingBitFrames = 0
    private var stableBits: String? = null
    private var debugBestScore = BAD_SCORE
    private var debugBestLine = "best: none"
    private var debugModeLine = "mode: none"
    private var debugRejectOutside = 0
    private var debugRejectDistance = 0
    private var debugRejectLedMarker = 0

    fun decode(image: ImageProxy): DetectionFrame? {
        val reader = YuvReader(image)
        val roi = reader.rotatedRoi()
        beginDebugFrame()
        val fit = findFit(reader, roi) ?: run {
            missedFrames++
            if (missedFrames >= RESET_AFTER_MISSES) {
                previousTheta = null
                previousConfidence = 0f
            }
            resetBitDebounce()
            finishDebugFrame(
                status = "MISS",
                fit = null,
                bits = null,
            )
            return null
        }

        previousTheta = fit.theta
        previousConfidence = fit.confidence
        missedFrames = 0

        val model = modelForTheta(fit.theta)
        val frame = decodeWithModel(reader, model)
        val debugLines = finishDebugFrame(
                status = "HIT",
                fit = fit,
                bits = frame.bits,
            )
        return frame.copy(debugLines = debugLines)
    }

    fun resetTracking() {
        previousTheta = null
        previousConfidence = 0f
        missedFrames = 0
        resetBitDebounce()
    }

    private fun beginDebugFrame() {
        debugBestScore = BAD_SCORE
        debugBestLine = "best: none"
        debugModeLine = "mode: none"
        debugRejectOutside = 0
        debugRejectDistance = 0
        debugRejectLedMarker = 0
    }

    private fun finishDebugFrame(status: String, fit: Fit?, bits: String?): List<String> {
        val lines = ArrayList<String>(8)
        lines.add("$status bits=${bits ?: "null"}")
        lines.add(debugModeLine)
        if (fit != null) {
            lines.add("fit score=${fmt(fit.score)} conf=${fmt(fit.confidence)} scale=${fmt(fit.theta.scale)} angle=${fmt(fit.theta.angle)}")
        } else {
            lines.add("fit: none prevConf=${fmt(previousConfidence)} missed=$missedFrames")
        }
        lines.add(debugBestLine)
        lines.add("reject outside=$debugRejectOutside dist=$debugRejectDistance ledMarker=$debugRejectLedMarker")
        lines.add("threshold score=$SCORE_THRESHOLD accept=$ACCEPT_CONFIDENCE")
        lastDebugLines = lines
        return lines
    }

    private fun fmt(value: Float): String = String.format(Locale.US, "%.2f", value)

    private fun findFit(reader: YuvReader, roi: RotatedRoi): Fit? {
        val prior = previousTheta
        val priorFit = if (prior != null && previousConfidence >= RECOVERY_PRIOR_CONFIDENCE) {
            debugModeLine = "mode: prior prevConf=${fmt(previousConfidence)} missed=$missedFrames"
            val refined = agstRefine(reader, roi, prior, trackingSchedule())
            refined.takeIf { isAcceptedFit(it) }
        } else {
            null
        }
        if (priorFit != null) return priorFit

        val centeredFit = centerRoiFit(reader, roi)
        if (centeredFit != null) return centeredFit
        if (missedFrames > 0 && missedFrames % BROAD_ACQUIRE_SCAN_PERIOD != 0) {
            debugModeLine = "mode: center miss; broad throttled missed=$missedFrames"
            return null
        }

        val seeds = coarseSeeds(reader, roi)
        debugModeLine = "mode: acquire seeds=${seeds.size} prevConf=${fmt(previousConfidence)} missed=$missedFrames"
        if (prior != null) {
            val priorScore = scoreTheta(reader, roi, prior)
            if (priorScore.isFiniteScore()) {
                seeds.add(priorScore to prior)
            }
        }
        if (seeds.isEmpty()) return null

        var best: Fit? = null
        for ((_, seed) in seeds
            .sortedByDescending { it.first }
            .take(if (previousConfidence > 0f) RECOVERY_SEEDS else ACQUIRE_SEEDS)
        ) {
            val fit = agstRefine(reader, roi, seed, acquireSchedule())
            if (best == null || fit.score > best!!.score) best = fit
        }
        return best?.takeIf { isAcceptedFit(it) }
    }

    private fun isAcceptedFit(fit: Fit): Boolean {
        return fit.confidence >= ACCEPT_CONFIDENCE && fit.score >= SCORE_THRESHOLD
    }

    private fun centerRoiFit(reader: YuvReader, roi: RotatedRoi): Fit? {
        var bestTheta: Theta? = null
        var bestScore = BAD_SCORE
        for (offsetY in CENTER_Y_OFFSETS) {
            val ry = roi.top + roi.height * (0.5f + offsetY)
            for (offsetX in CENTER_X_OFFSETS) {
                val rx = roi.left + roi.width * (0.5f + offsetX)
                for (angle in CENTER_ANGLES) {
                    for (distanceFraction in CENTER_DISTANCE_FRACTIONS) {
                        val markerDistance = roi.width * distanceFraction
                        val scale = markerDistance / constants.markerDistanceToSizeRatio()
                        val theta = thetaFromRotated(reader, rx, ry, angle, scale)
                        val score = scoreTheta(reader, roi, theta)
                        if (score > bestScore) {
                            bestScore = score
                            bestTheta = theta
                        }
                    }
                }
            }
        }

        val seed = bestTheta ?: return null
        val refined = agstRefine(reader, roi, seed, centerAcquireSchedule())
        debugModeLine = "mode: center score=${fmt(bestScore)} refined=${fmt(refined.score)} conf=${fmt(refined.confidence)} missed=$missedFrames"
        return refined.takeIf { isAcceptedFit(it) }
    }

    private fun coarseSeeds(reader: YuvReader, roi: RotatedRoi): MutableList<Pair<Float, Theta>> {
        val seeds = ArrayList<Pair<Float, Theta>>(ACQUIRE_SEEDS)
        val scaleMin = max(4f, roi.width * 0.030f)
        val scaleMax = max(scaleMin + 1f, roi.width * 0.145f)
        val distanceFractions = floatArrayOf(0.48f, 0.56f, 0.64f, 0.74f, 0.84f)
        val angleSteps = 16
        val columns = 3
        val rows = 3

        for (cyIndex in 0 until rows) {
            val ry = roi.top + roi.height * (cyIndex + 1f) / (rows + 1f)
            for (cxIndex in 0 until columns) {
                val rx = roi.left + roi.width * (cxIndex + 1f) / (columns + 1f)
                for (angleIndex in 0 until angleSteps) {
                    val rotatedAngle = (2.0 * PI * angleIndex / angleSteps).toFloat()
                    for (distanceFraction in distanceFractions) {
                        val markerDistance = roi.width * distanceFraction
                        val scale = (markerDistance / constants.markerDistanceToSizeRatio())
                            .coerceIn(scaleMin, scaleMax)
                        val theta = thetaFromRotated(reader, rx, ry, rotatedAngle, scale)
                        val score = scoreTheta(reader, roi, theta)
                        if (score.isFiniteScore()) insertTopSeed(seeds, score to theta, COARSE_TOP_SEEDS)
                    }
                }
            }
        }
        return seeds
    }

    private fun insertTopSeed(
        seeds: MutableList<Pair<Float, Theta>>,
        candidate: Pair<Float, Theta>,
        limit: Int,
    ) {
        val insertAt = seeds.indexOfFirst { candidate.first > it.first }
        if (insertAt < 0) {
            if (seeds.size < limit) seeds.add(candidate)
            return
        }
        seeds.add(insertAt, candidate)
        if (seeds.size > limit) seeds.removeAt(seeds.lastIndex)
    }

    private fun thetaFromRotated(
        reader: YuvReader,
        x: Float,
        y: Float,
        rotatedAngle: Float,
        scale: Float,
    ): Theta {
        val p0 = reader.rotatedToImage(x, y)
        val p1 = reader.rotatedToImage(x + cos(rotatedAngle), y + sin(rotatedAngle))
        return Theta(
            cx = p0.x,
            cy = p0.y,
            angle = atan2(p1.y - p0.y, p1.x - p0.x),
            scale = scale,
        )
    }

    private fun agstRefine(
        reader: YuvReader,
        roi: RotatedRoi,
        seed: Theta,
        schedule: List<NoiseLevel>,
    ): Fit {
        var theta = seed
        var thetaScore = scoreTheta(reader, roi, theta)
        var bestTheta = theta
        var bestScore = thetaScore

        for (level in schedule) {
            repeat(level.steps) {
                var localBestTheta = theta
                var localBestScore = thetaScore
                val candidates = ArrayList<Pair<Float, FloatArray>>(level.samples + 1)
                candidates.add(thetaScore to floatArrayOf(0f, 0f, 0f, 0f))

                repeat(level.samples) {
                    val dx = nextGaussian() * level.translationPx
                    val dy = nextGaussian() * level.translationPx
                    val da = nextGaussian() * level.angleRad
                    val ds = nextGaussian() * level.scaleRatio
                    val candidateTheta = normalizeTheta(
                        theta.copy(
                            cx = theta.cx + dx,
                            cy = theta.cy + dy,
                            angle = theta.angle + da,
                            scale = theta.scale * (1f + ds).coerceIn(0.72f, 1.32f),
                        ),
                    )
                    val score = scoreTheta(reader, roi, candidateTheta)
                    candidates.add(score to floatArrayOf(dx, dy, da, ds))
                    if (score > localBestScore) {
                        localBestScore = score
                        localBestTheta = candidateTheta
                    }
                }

                val maxScore = candidates.maxOf { it.first.toDouble() }.toFloat()
                var sumW = 0f
                var sumDx = 0f
                var sumDy = 0f
                var sumDa = 0f
                var sumDs = 0f
                for ((score, delta) in candidates) {
                    if (!score.isFiniteScore()) continue
                    val w = exp(((score - maxScore) / level.temperature).toDouble()).toFloat()
                    sumW += w
                    sumDx += w * delta[0]
                    sumDy += w * delta[1]
                    sumDa += w * delta[2]
                    sumDs += w * delta[3]
                }

                if (sumW > 0f) {
                    val updated = normalizeTheta(
                        theta.copy(
                            cx = theta.cx + level.alpha * sumDx / sumW,
                            cy = theta.cy + level.alpha * sumDy / sumW,
                            angle = theta.angle + level.alpha * sumDa / sumW,
                            scale = theta.scale * (1f + level.alpha * sumDs / sumW).coerceIn(0.82f, 1.22f),
                        ),
                    )
                    val updatedScore = scoreTheta(reader, roi, updated)
                    if (updatedScore >= localBestScore - 0.18f) {
                        theta = updated
                        thetaScore = updatedScore
                    } else {
                        theta = localBestTheta
                        thetaScore = localBestScore
                    }
                } else {
                    theta = localBestTheta
                    thetaScore = localBestScore
                }

                if (thetaScore > bestScore) {
                    bestScore = thetaScore
                    bestTheta = theta
                }
            }
        }

        val polished = polish(reader, roi, bestTheta, bestScore)
        val confidence = sigmoid((polished.second - SCORE_THRESHOLD) / SCORE_CONFIDENCE_WIDTH)
        return Fit(polished.first, polished.second, confidence)
    }

    private fun polish(
        reader: YuvReader,
        roi: RotatedRoi,
        seed: Theta,
        seedScore: Float,
    ): Pair<Theta, Float> {
        var bestTheta = seed
        var bestScore = seedScore
        val move = max(0.8f, seed.scale * 0.08f)
        val fineMove = max(0.45f, seed.scale * 0.035f)
        val angle = 0.012f
        val scale = 0.010f
        val model = modelForTheta(seed)
        val candidates = arrayOf(
            seed.copy(cx = seed.cx - move),
            seed.copy(cx = seed.cx + move),
            seed.copy(cy = seed.cy - move),
            seed.copy(cy = seed.cy + move),
            seed.copy(cx = seed.cx + model.ux * fineMove, cy = seed.cy + model.uy * fineMove),
            seed.copy(cx = seed.cx - model.ux * fineMove, cy = seed.cy - model.uy * fineMove),
            seed.copy(cx = seed.cx + model.vx * fineMove, cy = seed.cy + model.vy * fineMove),
            seed.copy(cx = seed.cx - model.vx * fineMove, cy = seed.cy - model.vy * fineMove),
            seed.copy(cx = seed.cx + model.vx * move, cy = seed.cy + model.vy * move),
            seed.copy(cx = seed.cx - model.vx * move, cy = seed.cy - model.vy * move),
            seed.copy(angle = seed.angle - angle),
            seed.copy(angle = seed.angle + angle),
            seed.copy(scale = seed.scale * (1f - scale)),
            seed.copy(scale = seed.scale * (1f + scale)),
        )
        for (candidate in candidates) {
            val normalized = normalizeTheta(candidate)
            val score = scoreTheta(reader, roi, normalized)
            if (score > bestScore) {
                bestScore = score
                bestTheta = normalized
            }
        }
        return bestTheta to bestScore
    }

    private fun trackingSchedule(): List<NoiseLevel> {
        return listOf(
            NoiseLevel(5f, deg(2.6f), 0.026f, 0.68f, 0.95f, 8, 1),
            NoiseLevel(2.2f, deg(1.0f), 0.011f, 0.44f, 0.88f, 8, 1),
            NoiseLevel(0.9f, deg(0.40f), 0.005f, 0.30f, 0.78f, 6, 1),
        )
    }

    private fun centerAcquireSchedule(): List<NoiseLevel> {
        return listOf(
            NoiseLevel(10f, deg(4.0f), 0.048f, 0.76f, 0.96f, 8, 1),
            NoiseLevel(4f, deg(1.8f), 0.018f, 0.48f, 0.90f, 8, 1),
            NoiseLevel(1.5f, deg(0.7f), 0.007f, 0.30f, 0.80f, 6, 1),
        )
    }

    private fun acquireSchedule(): List<NoiseLevel> {
        return listOf(
            NoiseLevel(14f, deg(7.0f), 0.070f, 0.92f, 1.00f, 10, 1),
            NoiseLevel(6f, deg(3.0f), 0.030f, 0.58f, 0.92f, 10, 1),
            NoiseLevel(2.2f, deg(1.0f), 0.012f, 0.36f, 0.82f, 8, 1),
        )
    }

    private fun scoreTheta(reader: YuvReader, roi: RotatedRoi, theta: Theta): Float {
        val model = modelForTheta(theta)
        if (!modelInsideRoi(reader, roi, model)) {
            debugRejectOutside++
            return BAD_SCORE
        }
        val rotatedStart = reader.imageToRotated(model.start)
        val rotatedEnd = reader.imageToRotated(model.end)
        val markerDistanceInRoi = hypot(rotatedEnd.x - rotatedStart.x, rotatedEnd.y - rotatedStart.y) / roi.width
        if (markerDistanceInRoi !in 0.42f..0.94f) {
            debugRejectDistance++
            return BAD_SCORE
        }

        val square = squareTemplateScore(reader, model.start, model.squareSizePx, model.ux, model.uy, model.vx, model.vy)
        val triangle = triangleTemplateScore(reader, model.end, model.triangleSizePx, model.ux, model.uy, model.vx, model.vy)
        if (square < MIN_SQUARE_SCORE || triangle < MIN_TRIANGLE_SCORE) {
            val shapeScore = square * 2.75f + triangle * 2.80f
            if (shapeScore > debugBestScore) {
                debugBestScore = shapeScore
                debugBestLine = "best shapeReject=${fmt(shapeScore)} s=${fmt(square)} t=${fmt(triangle)} dist=${fmt(markerDistanceInRoi)}"
            }
            return BAD_SCORE
        }
        val squareAsTriangle = triangleTemplateScore(reader, model.start, model.squareSizePx, model.ux, model.uy, model.vx, model.vy)
        val triangleAsSquare = squareTemplateScore(reader, model.end, model.triangleSizePx, model.ux, model.uy, model.vx, model.vy)
        val markerOrder = max(0f, squareAsTriangle - square + 0.10f) +
            max(0f, triangleAsSquare - triangle + 0.10f)

        val lampObject = model.slots
            .map { lampObjectScore(reader, it, model.ledRadiusPx, model.ux, model.uy, model.vx, model.vy) }
            .average()
            .toFloat()
        val lampOn = model.slots
            .map { lampOnValue(reader, it, model.ledRadiusPx, model.ux, model.uy, model.vx, model.vy).coerceIn(0f, 1f) }
            .average()
            .toFloat()
        val backgroundPenalty = backgroundPenalty(reader, model)
        val startBluePenalty = blueMarkerPenalty(reader, model.start, model.squareSizePx)
        val endBluePenalty = blueMarkerPenalty(reader, model.end, model.triangleSizePx)
        val startGlowPenalty = markerGlowPenalty(reader, model.start, model.squareSizePx)
        val endGlowPenalty = markerGlowPenalty(reader, model.end, model.triangleSizePx)
        val startLampPenalty = markerLampPenalty(reader, model.start, model.squareSizePx, model.ux, model.uy, model.vx, model.vy)
        val endLampPenalty = markerLampPenalty(reader, model.end, model.triangleSizePx, model.ux, model.uy, model.vx, model.vy)
        val strongStartShape = square >= 0.55f
        val strongEndShape = triangle >= 0.52f
        if (
            (startBluePenalty > 0.72f && startGlowPenalty > 0.38f && !strongStartShape) ||
            (endBluePenalty > 0.72f && endGlowPenalty > 0.38f && !strongEndShape) ||
            (startLampPenalty > 0.76f && !strongStartShape) ||
            (endLampPenalty > 0.76f && !strongEndShape) ||
            (startGlowPenalty > 0.90f && square < 0.72f) ||
            (endGlowPenalty > 0.90f && triangle < 0.68f)
        ) {
            debugRejectLedMarker++
            return BAD_SCORE
        }
        val markerBluePenalty = startBluePenalty + endBluePenalty
        val markerGlowPenalty = startGlowPenalty + endGlowPenalty
        val markerLampPenalty = startLampPenalty + endLampPenalty

        val score = square * 2.75f +
            triangle * 2.80f +
            lampObject * 0.55f +
            lampOn * 0.08f -
            backgroundPenalty * 0.55f -
            markerBluePenalty * 0.75f -
            markerGlowPenalty * 0.45f -
            markerLampPenalty * 0.35f -
            markerOrder * 1.85f
        if (score > debugBestScore) {
            debugBestScore = score
            debugBestLine = "best score=${fmt(score)} s=${fmt(square)} t=${fmt(triangle)} dist=${fmt(markerDistanceInRoi)}"
            debugBestLine += " bg=${fmt(backgroundPenalty)} blue=${fmt(markerBluePenalty)} glow=${fmt(markerGlowPenalty)} lamp=${fmt(markerLampPenalty)}"
        }
        return score
    }

    private fun modelForTheta(theta: Theta): PatternModel {
        val angle = normalizeAngle(theta.angle)
        val ux = cos(angle)
        val uy = sin(angle)
        val vx = -uy
        val vy = ux
        val distance = theta.scale * constants.markerDistanceToSizeRatio()
        val start = pointOnPattern(theta, ux, uy, constants.squareCenterFromPatternCenter())
        val end = pointOnPattern(theta, ux, uy, constants.triangleCenterFromPatternCenter())
        val slots = constants.slotCenterOffsetsFromPatternCenter().map { offset ->
            pointOnPattern(theta, ux, uy, offset)
        }
        return PatternModel(
            start = start,
            end = end,
            slots = slots,
            markerSizePx = theta.scale,
            squareSizePx = theta.scale * constants.squareSizeToMarkerSizeRatio(),
            triangleSizePx = theta.scale * constants.triangleSizeToMarkerSizeRatio(),
            ledRadiusPx = theta.scale * constants.ledRadiusToMarkerSizeRatio(),
            slotStepPx = distance * constants.slotStepFraction(),
            ux = ux,
            uy = uy,
            vx = vx,
            vy = vy,
        )
    }

    private fun pointOnPattern(theta: Theta, ux: Float, uy: Float, offsetFromCenterMarkerSizes: Float): ImagePoint {
        return ImagePoint(
            x = theta.cx + ux * theta.scale * offsetFromCenterMarkerSizes,
            y = theta.cy + uy * theta.scale * offsetFromCenterMarkerSizes,
        )
    }

    private fun modelInsideRoi(reader: YuvReader, roi: RotatedRoi, model: PatternModel): Boolean {
        val margin = max(2f, model.markerSizePx * 0.65f)
        val points = model.slots + model.start + model.end
        return points.all { point ->
            if (point.x < 0f || point.x >= reader.width.toFloat() || point.y < 0f || point.y >= reader.height.toFloat()) {
                return@all false
            }
            val rotated = reader.imageToRotated(point)
            rotated.x >= roi.left + margin &&
                rotated.x <= roi.left + roi.width - margin &&
                rotated.y >= roi.top + margin &&
                rotated.y <= roi.top + roi.height - margin
        }
    }

    private fun decodeWithModel(reader: YuvReader, model: PatternModel): DetectionFrame {
        val bitMeasurements = model.slots.map { slot ->
            ledBitMeasurement(reader, slot, model.ledRadiusPx, model.ux, model.uy, model.vx, model.vy)
        }
        val lowConfidence = bitMeasurements.any { it.confidence < BIT_MIN_CONFIDENCE }
        val rawBits = buildString(capacity = 5) {
            bitMeasurements.forEach { append(if (it.isOn) '1' else '0') }
        }
        val bits = if (lowConfidence) stableBits ?: rawBits else stabilizeBits(rawBits)
        val overlayRadius = (model.ledRadiusPx * 1.28f).coerceIn(3.5f, 13f)
        return DetectionFrame(
            timestampNs = reader.timestampNs,
            imageWidth = reader.width,
            imageHeight = reader.height,
            cropLeft = reader.cropLeft,
            cropTop = reader.cropTop,
            cropWidth = reader.cropWidth,
            cropHeight = reader.cropHeight,
            rotationDegrees = reader.rotationDegrees,
            bits = bits,
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

    private fun squareTemplateScore(
        reader: YuvReader,
        center: ImagePoint,
        sizePx: Float,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
    ): Float {
        val r = (sizePx * 0.075f).roundToInt().coerceIn(1, 4)
        val corners = arrayOf(
            sample(center, ux, uy, vx, vy, -0.32f, -0.32f, sizePx),
            sample(center, ux, uy, vx, vy, 0.32f, -0.32f, sizePx),
            sample(center, ux, uy, vx, vy, -0.32f, 0.32f, sizePx),
            sample(center, ux, uy, vx, vy, 0.32f, 0.32f, sizePx),
        ).map { markerValue(reader, it, r) }.average().toFloat()
        val body = arrayOf(
            center,
            sample(center, ux, uy, vx, vy, -0.18f, 0f, sizePx),
            sample(center, ux, uy, vx, vy, 0.18f, 0f, sizePx),
            sample(center, ux, uy, vx, vy, 0f, -0.18f, sizePx),
            sample(center, ux, uy, vx, vy, 0f, 0.18f, sizePx),
        ).map { markerValue(reader, it, r) }.average().toFloat()
        val outside = squareOutsideSamples(center, ux, uy, vx, vy, sizePx)
            .map { markerValue(reader, it, r) }
            .average()
            .toFloat()
        val bluePenalty = blueObjectValue(reader, center, (sizePx * 0.30f).roundToInt().coerceIn(2, 9))
        val glowPenalty = markerGlowPenalty(reader, center, sizePx)
        if (corners < 0.22f || body < 0.28f) return 0f
        return (corners * 0.46f + body * 0.34f + (1f - outside) * 0.20f - bluePenalty * 0.55f - glowPenalty * 0.45f)
            .coerceIn(0f, 1f)
    }

    private fun triangleTemplateScore(
        reader: YuvReader,
        center: ImagePoint,
        sizePx: Float,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
    ): Float {
        val r = (sizePx * 0.075f).roundToInt().coerceIn(1, 4)
        val inside = arrayOf(
            sample(center, ux, uy, vx, vy, 0f, -0.34f, sizePx),
            sample(center, ux, uy, vx, vy, -0.24f, 0.24f, sizePx),
            sample(center, ux, uy, vx, vy, 0.24f, 0.24f, sizePx),
            sample(center, ux, uy, vx, vy, 0f, 0.06f, sizePx),
            sample(center, ux, uy, vx, vy, 0f, 0.30f, sizePx),
        ).map { markerValue(reader, it, r) }.average().toFloat()
        val outside = arrayOf(
            sample(center, ux, uy, vx, vy, -0.52f, 0.04f, sizePx),
            sample(center, ux, uy, vx, vy, 0.52f, 0.04f, sizePx),
            sample(center, ux, uy, vx, vy, -0.32f, -0.34f, sizePx),
            sample(center, ux, uy, vx, vy, 0.32f, -0.34f, sizePx),
            sample(center, ux, uy, vx, vy, 0f, -0.62f, sizePx),
            sample(center, ux, uy, vx, vy, 0f, 0.62f, sizePx),
        ).map { markerValue(reader, it, r) }.average().toFloat()
        val bluePenalty = blueObjectValue(reader, center, (sizePx * 0.30f).roundToInt().coerceIn(2, 9))
        val glowPenalty = markerGlowPenalty(reader, center, sizePx)
        if (inside < 0.24f) return 0f
        return (inside * 0.68f + (1f - outside) * 0.32f - bluePenalty * 0.55f - glowPenalty * 0.45f).coerceIn(0f, 1f)
    }

    private fun squareOutsideSamples(
        center: ImagePoint,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
        sizePx: Float,
    ): Array<ImagePoint> {
        return arrayOf(
            sample(center, ux, uy, vx, vy, -0.62f, 0f, sizePx),
            sample(center, ux, uy, vx, vy, 0.62f, 0f, sizePx),
            sample(center, ux, uy, vx, vy, 0f, -0.62f, sizePx),
            sample(center, ux, uy, vx, vy, 0f, 0.62f, sizePx),
            sample(center, ux, uy, vx, vy, -0.58f, -0.58f, sizePx),
            sample(center, ux, uy, vx, vy, 0.58f, -0.58f, sizePx),
            sample(center, ux, uy, vx, vy, -0.58f, 0.58f, sizePx),
            sample(center, ux, uy, vx, vy, 0.58f, 0.58f, sizePx),
        )
    }

    private fun lampObjectScore(
        reader: YuvReader,
        center: ImagePoint,
        radiusPx: Float,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
    ): Float {
        val r = radiusPx.roundToInt().coerceIn(2, 9)
        val side = radiusPx * 2.65f
        val centerLuma = lumaMean(reader, center, r)
        val bgLuma = arrayOf(
            lumaMean(reader, center.offset(ux, uy, vx, vy, 0f, side), r),
            lumaMean(reader, center.offset(ux, uy, vx, vy, 0f, -side), r),
            lumaMean(reader, center.offset(ux, uy, vx, vy, side, 0f), r),
            lumaMean(reader, center.offset(ux, uy, vx, vy, -side, 0f), r),
        ).average().toFloat()
        val blue = blueObjectValue(reader, center, r)
        val contrast = (abs(centerLuma - bgLuma) / 75f).coerceIn(0f, 1f)
        val ringQuiet = 1f - backgroundBusy(reader, center, radiusPx * 2.4f, ux, uy, vx, vy).coerceIn(0f, 1f)
        return (blue * 0.46f + contrast * 0.36f + ringQuiet * 0.18f).coerceIn(0f, 1f)
    }

    private fun lampOnValue(
        reader: YuvReader,
        center: ImagePoint,
        radiusPx: Float,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
    ): Float {
        val r = radiusPx.roundToInt().coerceIn(2, 9)
        val side = radiusPx * 2.65f
        val centerLuma = lumaMean(reader, center, r)
        val centerPeak = lumaMax(reader, center, max(r, (r * 1.35f).roundToInt()))
        val bgLuma = arrayOf(
            lumaMean(reader, center.offset(ux, uy, vx, vy, 0f, side), r),
            lumaMean(reader, center.offset(ux, uy, vx, vy, 0f, -side), r),
            lumaMean(reader, center.offset(ux, uy, vx, vy, side, 0f), r),
            lumaMean(reader, center.offset(ux, uy, vx, vy, -side, 0f), r),
        ).average().toFloat()
        val peakScore = (centerPeak - max(118f, bgLuma + 20f)) / 85f
        val lumaScore = (centerLuma - bgLuma - 15f) / 55f
        return (peakScore * 0.58f + lumaScore * 0.42f).coerceIn(-1f, 1.4f)
    }

    private fun ledBitMeasurement(
        reader: YuvReader,
        center: ImagePoint,
        radiusPx: Float,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
    ): LedBitMeasurement {
        val on = lampOnValue(reader, center, radiusPx, ux, uy, vx, vy)
        val isOn = on > BIT_ON_THRESHOLD
        val confidence = abs(on - BIT_ON_THRESHOLD) * 55f
        return LedBitMeasurement(isOn = isOn, confidence = confidence)
    }

    private fun backgroundPenalty(reader: YuvReader, model: PatternModel): Float {
        var penalty = 0f
        var samples = 0
        val fractions = floatArrayOf(0.11f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f, 0.70f, 0.80f, 0.89f)
        val slotFractions = constants.slotFractions()
        val distance = model.markerSizePx * constants.markerDistanceToSizeRatio()
        for (fraction in fractions) {
            if (slotFractions.any { abs(it - fraction) < 0.040f }) continue
            val p = ImagePoint(
                model.start.x + model.ux * distance * fraction,
                model.start.y + model.uy * distance * fraction,
            )
            val radius = (model.ledRadiusPx * 0.75f).roundToInt().coerceIn(1, 6)
            val busy = max(
                blueObjectValue(reader, p, radius),
                grayValue(reader, p, radius),
            ) + max(0f, lumaMean(reader, p, radius) - 145f) / 95f
            penalty += busy.coerceIn(0f, 1f)
            samples++
        }
        penalty += backgroundBusy(reader, model.start, model.markerSizePx * 1.20f, model.ux, model.uy, model.vx, model.vy) * 0.35f
        penalty += backgroundBusy(reader, model.end, model.markerSizePx * 1.20f, model.ux, model.uy, model.vx, model.vy) * 0.35f
        return if (samples == 0) penalty else (penalty / samples).coerceIn(0f, 1.5f)
    }

    private fun backgroundBusy(
        reader: YuvReader,
        center: ImagePoint,
        offsetPx: Float,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
    ): Float {
        val r = (offsetPx * 0.16f).roundToInt().coerceIn(1, 6)
        val points = arrayOf(
            center.offset(ux, uy, vx, vy, offsetPx, 0f),
            center.offset(ux, uy, vx, vy, -offsetPx, 0f),
            center.offset(ux, uy, vx, vy, 0f, offsetPx),
            center.offset(ux, uy, vx, vy, 0f, -offsetPx),
        )
        return points
            .map { max(blueObjectValue(reader, it, r), grayValue(reader, it, r)) }
            .average()
            .toFloat()
    }

    private fun blueMarkerPenalty(reader: YuvReader, center: ImagePoint, markerSizePx: Float): Float {
        return blueObjectValue(reader, center, (markerSizePx * 0.28f).roundToInt().coerceIn(2, 8))
    }

    private fun markerGlowPenalty(reader: YuvReader, center: ImagePoint, markerSizePx: Float): Float {
        val peakRadius = (markerSizePx * 0.16f).roundToInt().coerceIn(2, 7)
        val meanRadius = (markerSizePx * 0.34f).roundToInt().coerceIn(3, 10)
        val peak = lumaMax(reader, center, peakRadius)
        val mean = lumaMean(reader, center, meanRadius)
        val blue = blueObjectValue(reader, center, peakRadius)
        val compactPeak = ((peak - mean - 34f) / 86f).coerceIn(0f, 1f)
        return (compactPeak * 0.78f + blue * 0.22f).coerceIn(0f, 1f)
    }

    private fun markerLampPenalty(
        reader: YuvReader,
        center: ImagePoint,
        markerSizePx: Float,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
    ): Float {
        val ledLikeRadius = markerSizePx * constants.ledRadiusToMarkerSizeRatio()
        val on = lampOnValue(reader, center, ledLikeRadius, ux, uy, vx, vy)
        val blue = blueObjectValue(reader, center, ledLikeRadius.roundToInt().coerceIn(2, 8))
        val gray = grayValue(reader, center, (markerSizePx * 0.30f).roundToInt().coerceIn(2, 8))
        val ledLike = ((on - 0.30f) / 0.74f).coerceIn(0f, 1f)
        return (ledLike * 0.38f + blue * 0.48f - gray * 0.48f).coerceIn(0f, 1f)
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
            val brightness = ((yy - 82f) / 145f).coerceIn(0f, 1f)
            val bluePenalty = (max(0f, reader.blueDominance(x, y) - 42f) / 95f).coerceIn(0f, 1f)
            val w = SAMPLE_W[i]
            score += brightness * (0.55f + neutral * 0.45f) * (1f - bluePenalty * 0.55f) * w
            weightSum += w
        }
        return if (weightSum == 0f) 0f else score / weightSum
    }

    private fun grayValue(reader: YuvReader, center: ImagePoint, radiusPx: Int): Float {
        var score = 0f
        var weightSum = 0f
        val cx = center.x.roundToInt()
        val cy = center.y.roundToInt()
        for (i in SAMPLE_X.indices) {
            val x = cx + (SAMPLE_X[i] * radiusPx).roundToInt()
            val y = cy + (SAMPLE_Y[i] * radiusPx).roundToInt()
            if (x !in 0 until reader.width || y !in 0 until reader.height) continue
            val yy = reader.y(x, y)
            val neutral = (1f - (abs(reader.u(x, y) - 128) + abs(reader.v(x, y) - 128)) / 92f)
                .coerceIn(0f, 1f)
            val brightness = ((yy - 92f) / 135f).coerceIn(0f, 1f)
            val notBlue = (1f - max(0f, reader.blueDominance(x, y) - 24f) / 80f).coerceIn(0f, 1f)
            val w = SAMPLE_W[i]
            score += neutral * brightness * notBlue * w
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

    private fun sample(
        center: ImagePoint,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
        alongUnit: Float,
        normalUnit: Float,
        sizePx: Float,
    ): ImagePoint {
        return center.offset(ux, uy, vx, vy, alongUnit * sizePx, normalUnit * sizePx)
    }

    private fun ImagePoint.offset(
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
        along: Float,
        normal: Float,
    ): ImagePoint {
        return ImagePoint(
            x = x + ux * along + vx * normal,
            y = y + uy * along + vy * normal,
        )
    }

    private fun stabilizeBits(rawBits: String): String {
        val currentStable = stableBits
        if (currentStable == null) {
            stableBits = rawBits
            pendingBits = null
            pendingBitFrames = 0
            return rawBits
        }
        if (rawBits == currentStable) {
            pendingBits = null
            pendingBitFrames = 0
            return currentStable
        }
        if (rawBits == pendingBits) {
            pendingBitFrames++
        } else {
            pendingBits = rawBits
            pendingBitFrames = 1
        }
        return if (pendingBitFrames >= BIT_CHANGE_CONFIRM_FRAMES) {
            stableBits = rawBits
            pendingBits = null
            pendingBitFrames = 0
            rawBits
        } else {
            currentStable
        }
    }

    private fun resetBitDebounce() {
        pendingBits = null
        pendingBitFrames = 0
        stableBits = null
    }

    private fun nextGaussian(): Float {
        var u1 = rng.nextDouble()
        if (u1 < 1e-9) u1 = 1e-9
        val u2 = rng.nextDouble()
        return (sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)).toFloat()
    }

    private fun normalizeTheta(theta: Theta): Theta {
        return theta.copy(angle = normalizeAngle(theta.angle))
    }

    private fun normalizeAngle(angle: Float): Float {
        var a = angle
        val twoPi = (2.0 * PI).toFloat()
        while (a <= -PI.toFloat()) a += twoPi
        while (a > PI.toFloat()) a -= twoPi
        return a
    }

    private fun deg(value: Float): Float = (value * PI / 180.0).toFloat()

    private fun sigmoid(value: Float): Float {
        return (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()
    }

    private fun Float.isFiniteScore(): Boolean = this > BAD_SCORE * 0.5f && this.isFinite()

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

        fun slotFractions(): List<Float> {
            return List(5) { index -> (firstLedOffsetMm + index * stepMm) / markerDistanceMm }
        }

        fun slotCenterOffsetsFromPatternCenter(): List<Float> {
            return List(5) { index ->
                (firstLedOffsetMm + index * stepMm - markerDistanceMm * 0.5f) / markerMm
            }
        }

        fun squareCenterFromPatternCenter(): Float = -markerDistanceMm * 0.5f / markerMm

        fun triangleCenterFromPatternCenter(): Float = markerDistanceMm * 0.5f / markerMm

        fun markerDistanceToSizeRatio(): Float = markerDistanceMm / markerMm

        fun slotStepFraction(): Float = stepMm / markerDistanceMm

        fun ledRadiusToMarkerSizeRatio(): Float = (ledMm * 0.5f) / markerMm

        fun squareSizeToMarkerSizeRatio(): Float = squareMm / markerMm

        fun triangleSizeToMarkerSizeRatio(): Float = triangleMm / markerMm
    }

    private companion object {
        const val BIT_CHANGE_CONFIRM_FRAMES = 2
        const val RESET_AFTER_MISSES = 5
        const val RECOVERY_PRIOR_CONFIDENCE = 0.18f
        const val ACCEPT_CONFIDENCE = 0.34f
        const val SCORE_THRESHOLD = 2.70f
        const val SCORE_CONFIDENCE_WIDTH = 0.42f
        const val BIT_ON_THRESHOLD = 0.48f
        const val BIT_MIN_CONFIDENCE = 7.0f
        const val MIN_SQUARE_SCORE = 0.34f
        const val MIN_TRIANGLE_SCORE = 0.22f
        const val ACQUIRE_SEEDS = 4
        const val RECOVERY_SEEDS = 3
        const val COARSE_TOP_SEEDS = 6
        const val BROAD_ACQUIRE_SCAN_PERIOD = 4
        const val BAD_SCORE = -1_000_000f
        val CENTER_ANGLES = floatArrayOf(
            0f,
            (-4f * PI / 180.0).toFloat(),
            (4f * PI / 180.0).toFloat(),
            (-8f * PI / 180.0).toFloat(),
            (8f * PI / 180.0).toFloat(),
            (-12f * PI / 180.0).toFloat(),
            (12f * PI / 180.0).toFloat(),
        )
        val CENTER_X_OFFSETS = floatArrayOf(-0.07f, 0f, 0.07f)
        val CENTER_Y_OFFSETS = floatArrayOf(-0.20f, 0f, 0.20f)
        val CENTER_DISTANCE_FRACTIONS = floatArrayOf(0.40f, 0.46f, 0.52f, 0.58f, 0.64f, 0.70f)
        val SAMPLE_X = floatArrayOf(0f, 1f, -1f, 0f, 0f, 0.70f, -0.70f, 0.70f, -0.70f, 0.45f, -0.45f, 0f, 0f)
        val SAMPLE_Y = floatArrayOf(0f, 0f, 0f, 1f, -1f, 0.70f, 0.70f, -0.70f, -0.70f, 0f, 0f, 0.45f, -0.45f)
        val SAMPLE_W = floatArrayOf(0.24f, 0.09f, 0.09f, 0.09f, 0.09f, 0.06f, 0.06f, 0.06f, 0.06f, 0.04f, 0.04f, 0.04f, 0.04f)
    }
}
