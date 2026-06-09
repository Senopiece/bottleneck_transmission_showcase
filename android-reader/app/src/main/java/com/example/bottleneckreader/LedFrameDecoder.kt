package com.example.bottleneckreader

import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class LedFrameDecoder {
    private data class Component(
        val cx: Float,
        val cy: Float,
        val area: Int,
        val minX: Int,
        val minY: Int,
        val maxX: Int,
        val maxY: Int,
        val fill: Float,
        val cornerFill: Float,
    )

    private data class MarkerPair(
        val start: Component,
        val end: Component,
        val score: Float,
    )

    private data class Geometry(
        val start: ImagePoint,
        val end: ImagePoint,
        val startSizePx: Float,
        val endSizePx: Float,
        val slots: List<ImagePoint>,
        val slotStepPx: Float,
    )

    private data class RotatedRoi(
        val left: Float,
        val top: Float,
        val width: Float,
        val height: Float,
    )

    private data class RefinedMarker(
        val center: ImagePoint,
        val sizePx: Float,
        val fill: Float,
    )

    private val constants = GeometryConstants()

    private var mask = BooleanArray(0)
    private var seen = BooleanArray(0)
    private var queue = IntArray(0)
    private var sampleXs = IntArray(0)
    private var sampleYs = IntArray(0)
    private var lastGeometry: Geometry? = null
    private var missedFrames = 0

    fun decode(image: ImageProxy): DetectionFrame? {
        val reader = YuvReader(image)
        val scanned = findGeometry(reader)
        val geometry = when {
            scanned != null -> scanned.also {
                lastGeometry = it
                missedFrames = 0
            }
            missedFrames < MAX_GEOMETRY_HOLD_FRAMES -> {
                val held = lastGeometry?.let { holdGeometry(reader, it) }
                if (held != null) {
                    missedFrames++
                    held.also { lastGeometry = it }
                } else {
                    missedFrames = MAX_GEOMETRY_HOLD_FRAMES
                    null
                }
            }
            else -> null
        } ?: return null
        return decodeWithGeometry(reader, geometry)
    }

    fun resetTracking() {
        lastGeometry = null
        missedFrames = 0
    }

    private fun decodeWithGeometry(reader: YuvReader, geometry: Geometry): DetectionFrame {
        val sampleRadius = (geometry.slotStepPx * 0.12f).toInt().coerceIn(2, 5)
        val lumaScores = geometry.slots.map { slot -> lumaScore(reader, slot, radiusPx = sampleRadius) }
        val blueScores = geometry.slots.map { slot -> blueScore(reader, slot, radiusPx = sampleRadius) }
        val minLuma = lumaScores.minOrNull() ?: 0f
        val maxLuma = lumaScores.maxOrNull() ?: 0f
        val lumaThreshold = if (maxLuma - minLuma > 30f) {
            minLuma + (maxLuma - minLuma) * 0.54f
        } else {
            135f
        }
        val bits = buildString(capacity = 5) {
            geometry.slots.indices.forEach { index ->
                val isOn = lumaScores[index] > lumaThreshold && blueScores[index] > 12f
                append(if (isOn) '1' else '0')
            }
        }
        val overlayRadius = (geometry.slotStepPx * 0.20f).coerceIn(4f, 12f)
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
            slots = geometry.slots.mapIndexed { index, point ->
                LedSlot(
                    imagePoint = point,
                    bitIndex = 4 - index,
                    isFirst = index == 0,
                    imageRadius = overlayRadius,
                )
            },
            markers = listOf(
                MarkerSlot(
                    imagePoint = geometry.start,
                    imageAlongPoint = geometry.end,
                    kind = MarkerKind.StartSquare,
                    imageSize = geometry.startSizePx,
                ),
                MarkerSlot(
                    imagePoint = geometry.end,
                    imageAlongPoint = ImagePoint(
                        x = geometry.end.x + geometry.end.x - geometry.start.x,
                        y = geometry.end.y + geometry.end.y - geometry.start.y,
                    ),
                    kind = MarkerKind.EndTriangle,
                    imageSize = geometry.endSizePx,
                ),
            ),
        )
    }

    private fun findGeometry(reader: YuvReader): Geometry? {
        val roi = reader.rotatedRoi()
        val markers = grayMarkerComponents(reader, roi)
            .filter { it.isPlausibleMarker() }
            .sortedByDescending { it.area }
            .take(24)
        if (markers.size < 2) return null

        val pair = bestMarkerPair(reader, roi, markers) ?: return null
        val square = pair.start
        val triangle = pair.end

        val refinedSquare = refineMarker(reader, square) ?: return null
        val refinedTriangle = refineMarker(reader, triangle) ?: return null
        val fitted = fitMarkerCenters(reader, refinedSquare, refinedTriangle) ?: return null
        val start = fitted.first.center
        val end = fitted.second.center
        val dx = end.x - start.x
        val dy = end.y - start.y
        val distance = hypot(dx, dy)
        if (distance < 32f) return null

        val slots = fixedSlots(start, end)
        if (!candidateInsideRoi(reader, roi, start, end, slots)) return null

        return Geometry(
            start = start,
            end = end,
            startSizePx = fitted.first.sizePx,
            endSizePx = fitted.second.sizePx,
            slots = slots,
            slotStepPx = distance * constants.slotStepFraction(),
        )
    }

    private fun holdGeometry(reader: YuvReader, previous: Geometry): Geometry? {
        val roi = reader.rotatedRoi()
        if (!candidateInsideRoi(reader, roi, previous.start, previous.end, previous.slots)) return null

        val dx = previous.end.x - previous.start.x
        val dy = previous.end.y - previous.start.y
        val distance = hypot(dx, dy)
        if (distance < 24f) return null
        val ux = dx / distance
        val uy = dy / distance
        val vx = -uy
        val vy = ux

        val start = fitShapeCenter(
            reader = reader,
            marker = RefinedMarker(previous.start, previous.startSizePx, 1f),
            ux = ux,
            uy = uy,
            vx = vx,
            vy = vy,
            kind = ShapeKind.Square,
            minScore = 0.50f,
            searchMultiplier = 0.62f,
        ) ?: return null
        val end = fitShapeCenter(
            reader = reader,
            marker = RefinedMarker(previous.end, previous.endSizePx, 1f),
            ux = ux,
            uy = uy,
            vx = vx,
            vy = vy,
            kind = ShapeKind.Triangle,
            minScore = 0.45f,
            searchMultiplier = 0.62f,
        ) ?: return null

        val newDx = end.center.x - start.center.x
        val newDy = end.center.y - start.center.y
        val newDistance = hypot(newDx, newDy)
        if (newDistance !in distance * 0.84f..distance * 1.18f) return null
        val newUx = newDx / newDistance
        val newUy = newDy / newDistance
        val alignment = ux * newUx + uy * newUy
        if (alignment < 0.94f) return null

        val newVx = -newUy
        val newVy = newUx
        val squareScore = squareDeviceShapeScore(reader, start.center, start.sizePx, newUx, newUy, newVx, newVy)
        val triangleScore = triangleDeviceShapeScore(reader, end.center, end.sizePx, newUx, newUy, newVx, newVy)
        val reverseScore = squareDeviceShapeScore(reader, end.center, end.sizePx, -newUx, -newUy, -newVx, -newVy) +
            triangleDeviceShapeScore(reader, start.center, start.sizePx, -newUx, -newUy, -newVx, -newVy)
        if (reverseScore > squareScore + triangleScore - 0.22f) return null

        val slots = fixedSlots(start.center, end.center)
        if (!candidateInsideRoi(reader, roi, start.center, end.center, slots)) return null
        return Geometry(
            start = start.center,
            end = end.center,
            startSizePx = start.sizePx,
            endSizePx = end.sizePx,
            slots = slots,
            slotStepPx = newDistance * constants.slotStepFraction(),
        )
    }

    private fun fixedSlots(start: ImagePoint, end: ImagePoint): List<ImagePoint> {
        val dx = end.x - start.x
        val dy = end.y - start.y
        return constants.slotFractions().map { fraction ->
            ImagePoint(start.x + dx * fraction, start.y + dy * fraction)
        }
    }

    private fun bestMarkerPair(reader: YuvReader, roi: RotatedRoi, markers: List<Component>): MarkerPair? {
        var best: MarkerPair? = null
        for (start in markers) {
            if (!start.isSquareLike()) continue
            for (end in markers) {
                if (start === end || !end.isTriangleLike()) continue
                val pair = scoreMarkerPair(reader, roi, start, end) ?: continue
                if (best == null || pair.score > best.score) best = pair
            }
        }
        return best?.takeIf { it.score >= 2.85f }
    }

    private fun scoreMarkerPair(
        reader: YuvReader,
        roi: RotatedRoi,
        start: Component,
        end: Component,
    ): MarkerPair? {
        val refinedStart = refineMarker(reader, start) ?: return null
        val refinedEnd = refineMarker(reader, end) ?: return null
        val fitted = fitMarkerCenters(reader, refinedStart, refinedEnd) ?: return null
        val fittedStart = fitted.first
        val fittedEnd = fitted.second
        val startAnchor = fittedStart.center
        val endAnchor = fittedEnd.center
        val dx = endAnchor.x - startAnchor.x
        val dy = endAnchor.y - startAnchor.y
        val distance = hypot(dx, dy)
        if (distance < 24f) return null
        val ux = dx / distance
        val uy = dy / distance
        val vx = -uy
        val vy = ux

        val avgMarkerSize = (fittedStart.sizePx + fittedEnd.sizePx) * 0.5f
        val distanceToMarkerRatio = distance / max(1f, avgMarkerSize)
        if (distanceToMarkerRatio !in 4.2f..19.0f) return null

        val sizeRatio = min(fittedStart.sizePx, fittedEnd.sizePx) / max(fittedStart.sizePx, fittedEnd.sizePx)
        if (sizeRatio < 0.42f) return null

        if (markerBlueLeak(reader, startAnchor, fittedStart.sizePx) > 42f) return null
        if (markerBlueLeak(reader, endAnchor, fittedEnd.sizePx) > 42f) return null

        val squareShapeScore = squareDeviceShapeScore(reader, startAnchor, fittedStart.sizePx, ux, uy, vx, vy)
        if (squareShapeScore < 0.48f) return null
        val triangleShapeScore = triangleDeviceShapeScore(reader, endAnchor, fittedEnd.sizePx, ux, uy, vx, vy)
        if (triangleShapeScore < 0.42f) return null

        val reverseSquareScore = squareDeviceShapeScore(reader, endAnchor, fittedEnd.sizePx, -ux, -uy, -vx, -vy)
        val reverseTriangleScore = triangleDeviceShapeScore(reader, startAnchor, fittedStart.sizePx, -ux, -uy, -vx, -vy)
        val forwardShapeScore = squareShapeScore + triangleShapeScore
        val reverseShapeScore = reverseSquareScore + reverseTriangleScore
        if (reverseShapeScore > forwardShapeScore - 0.12f) return null

        if (squareShapeScore < triangleDeviceShapeScore(reader, startAnchor, fittedStart.sizePx, ux, uy, vx, vy) + 0.06f) {
            return null
        }
        if (triangleShapeScore < squareDeviceShapeScore(reader, endAnchor, fittedEnd.sizePx, ux, uy, vx, vy) + 0.06f) {
            return null
        }

        val slots = fixedSlots(startAnchor, endAnchor)
        if (!candidateInsideRoi(reader, roi, startAnchor, endAnchor, slots)) return null

        val backgroundScore = darkBackgroundScore(reader, startAnchor, endAnchor)
        if (backgroundScore < 0.16f) return null

        val brightIntrusions = brightIntrusionsNearLine(reader, startAnchor, endAnchor)
        if (brightIntrusions > 9) return null

        val squareScore = start.squareScore()
        val startTriangleScore = start.triangleScore()
        if (squareScore < startTriangleScore - 0.18f) return null

        val triangleScore = end.triangleScore()
        val endSquareScore = end.squareScore()
        if (triangleScore < endSquareScore - 0.18f) return null

        val ratioScore = 1f - min(1f, abs(distanceToMarkerRatio - constants.markerDistanceToSizeRatio()) / 7f)
        val score = squareScore * 1.6f +
            triangleScore * 1.6f +
            squareShapeScore * 1.1f +
            triangleShapeScore * 1.1f +
            backgroundScore * 1.4f +
            ratioScore +
            markerQuietnessScore(reader, startAnchor, fittedStart.sizePx) +
            markerQuietnessScore(reader, endAnchor, fittedEnd.sizePx) +
            fittedStart.fill.coerceIn(0f, 1f) * 0.25f +
            fittedEnd.fill.coerceIn(0f, 1f) * 0.25f -
            brightIntrusions * 0.14f

        return MarkerPair(start = start, end = end, score = score)
    }

    private fun fitMarkerCenters(
        reader: YuvReader,
        start: RefinedMarker,
        end: RefinedMarker,
    ): Pair<RefinedMarker, RefinedMarker>? {
        val dx = end.center.x - start.center.x
        val dy = end.center.y - start.center.y
        val distance = hypot(dx, dy)
        if (distance < 1f) return null
        val ux = dx / distance
        val uy = dy / distance
        val vx = -uy
        val vy = ux

        val fittedStart = fitShapeCenter(reader, start, ux, uy, vx, vy, ShapeKind.Square) ?: return null
        val fittedEnd = fitShapeCenter(reader, end, ux, uy, vx, vy, ShapeKind.Triangle) ?: return null
        return fittedStart to fittedEnd
    }

    private enum class ShapeKind {
        Square,
        Triangle,
    }

    private fun fitShapeCenter(
        reader: YuvReader,
        marker: RefinedMarker,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
        kind: ShapeKind,
        minScore: Float = when (kind) {
            ShapeKind.Square -> 0.44f
            ShapeKind.Triangle -> 0.38f
        },
        searchMultiplier: Float = 0.42f,
    ): RefinedMarker? {
        val size = marker.sizePx.coerceIn(4f, 56f)
        val search = (size * searchMultiplier).coerceIn(2.5f, 15f)
        val step = (size * 0.13f).coerceIn(1.2f, 3.8f)
        var bestCenter = marker.center
        var bestScore = shapeScore(reader, marker.center, size, ux, uy, vx, vy, kind)
        var along = -search
        while (along <= search + 0.01f) {
            var normal = -search
            while (normal <= search + 0.01f) {
                val center = marker.center.offset(ux, uy, vx, vy, along, normal)
                val score = shapeScore(reader, center, size, ux, uy, vx, vy, kind)
                if (score > bestScore) {
                    bestScore = score
                    bestCenter = center
                }
                normal += step
            }
            along += step
        }
        if (bestScore < minScore) return null
        return marker.copy(center = bestCenter)
    }

    private fun shapeScore(
        reader: YuvReader,
        center: ImagePoint,
        sizePx: Float,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
        kind: ShapeKind,
    ): Float {
        return when (kind) {
            ShapeKind.Square -> squareDeviceShapeScore(reader, center, sizePx, ux, uy, vx, vy)
            ShapeKind.Triangle -> triangleDeviceShapeScore(reader, center, sizePx, ux, uy, vx, vy)
        }
    }

    private fun refineMarker(reader: YuvReader, component: Component): RefinedMarker? {
        val anchor = component.anchorPoint()
        val coarseSize = component.longSide().coerceAtLeast(4f)
        val radius = (coarseSize * 1.25f).toInt().coerceIn(6, 42)
        val step = if (radius <= 16) 1 else 2
        val cx = anchor.x.roundToInt()
        val cy = anchor.y.roundToInt()
        val localW = radius * 2 / step + 1
        val localH = radius * 2 / step + 1
        val localSize = localW * localH
        val localMask = BooleanArray(localSize)
        val localSeen = BooleanArray(localSize)
        val localQueue = IntArray(localSize)
        val originX = cx - radius
        val originY = cy - radius

        var seed = -1
        var seedDistance = Int.MAX_VALUE
        for (gy in 0 until localH) {
            val y = originY + gy * step
            for (gx in 0 until localW) {
                val x = originX + gx * step
                val idx = gy * localW + gx
                val isMarker = reader.isGrayMarker(x, y)
                localMask[idx] = isMarker
                if (isMarker) {
                    val dx = gx - localW / 2
                    val dy = gy - localH / 2
                    val d2 = dx * dx + dy * dy
                    if (d2 < seedDistance) {
                        seedDistance = d2
                        seed = idx
                    }
                }
            }
        }
        if (seed < 0) return null

        var head = 0
        var tail = 0
        localQueue[tail++] = seed
        localSeen[seed] = true

        var count = 0
        var minX = Int.MAX_VALUE
        var minY = Int.MAX_VALUE
        var maxX = Int.MIN_VALUE
        var maxY = Int.MIN_VALUE
        var neutralSum = 0f

        while (head < tail) {
            val idx = localQueue[head++]
            val gx = idx % localW
            val gy = idx / localW
            val x = originX + gx * step
            val y = originY + gy * step
            count++
            minX = min(minX, x)
            minY = min(minY, y)
            maxX = max(maxX, x)
            maxY = max(maxY, y)
            neutralSum += reader.grayNeutrality(x, y)

            if (gx > 0) {
                val ni = idx - 1
                if (localMask[ni] && !localSeen[ni]) {
                    localSeen[ni] = true
                    localQueue[tail++] = ni
                }
            }
            if (gx + 1 < localW) {
                val ni = idx + 1
                if (localMask[ni] && !localSeen[ni]) {
                    localSeen[ni] = true
                    localQueue[tail++] = ni
                }
            }
            if (gy > 0) {
                val ni = idx - localW
                if (localMask[ni] && !localSeen[ni]) {
                    localSeen[ni] = true
                    localQueue[tail++] = ni
                }
            }
            if (gy + 1 < localH) {
                val ni = idx + localW
                if (localMask[ni] && !localSeen[ni]) {
                    localSeen[ni] = true
                    localQueue[tail++] = ni
                }
            }
        }

        if (count < 3) return null
        val width = maxX - minX + 1
        val height = maxY - minY + 1
        val boxArea = max(1, (width / step + 1) * (height / step + 1))
        val fill = count.toFloat() / boxArea
        val size = max(width, height).toFloat()
        if (size !in coarseSize * 0.45f..coarseSize * 2.35f) return null
        if (neutralSum / count < 4f) return null

        return RefinedMarker(
            center = ImagePoint(
                x = (minX + maxX) * 0.5f,
                y = (minY + maxY) * 0.5f,
            ),
            sizePx = size,
            fill = fill,
        )
    }

    private fun markerBlueLeak(reader: YuvReader, center: ImagePoint, markerSizePx: Float): Float {
        val radius = (markerSizePx * 0.62f).toInt().coerceIn(4, 16)
        return blueScore(reader, center, radiusPx = radius)
    }

    private fun markerQuietnessScore(reader: YuvReader, center: ImagePoint, markerSizePx: Float): Float {
        val leak = markerBlueLeak(reader, center, markerSizePx)
        return (1f - (leak / 34f).coerceIn(0f, 1f)) * 0.85f
    }

    private fun squareDeviceShapeScore(
        reader: YuvReader,
        center: ImagePoint,
        markerSizePx: Float,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
    ): Float {
        val half = markerSizePx * 0.34f
        val radius = (markerSizePx * 0.10f).toInt().coerceIn(1, 5)
        val cornerPoints = arrayOf(
            center.offset(ux, uy, vx, vy, -half, -half),
            center.offset(ux, uy, vx, vy, half, -half),
            center.offset(ux, uy, vx, vy, -half, half),
            center.offset(ux, uy, vx, vy, half, half),
        )
        val bodyPoints = arrayOf(
            center,
            center.offset(ux, uy, vx, vy, -half * 0.65f, 0f),
            center.offset(ux, uy, vx, vy, half * 0.65f, 0f),
            center.offset(ux, uy, vx, vy, 0f, -half * 0.65f),
            center.offset(ux, uy, vx, vy, 0f, half * 0.65f),
        )
        val outsidePoints = arrayOf(
            center.offset(ux, uy, vx, vy, 0f, -markerSizePx * 0.78f),
            center.offset(ux, uy, vx, vy, 0f, markerSizePx * 0.78f),
            center.offset(ux, uy, vx, vy, -markerSizePx * 0.78f, 0f),
            center.offset(ux, uy, vx, vy, markerSizePx * 0.78f, 0f),
            center.offset(ux, uy, vx, vy, -markerSizePx * 0.72f, -markerSizePx * 0.72f),
            center.offset(ux, uy, vx, vy, markerSizePx * 0.72f, -markerSizePx * 0.72f),
            center.offset(ux, uy, vx, vy, -markerSizePx * 0.72f, markerSizePx * 0.72f),
            center.offset(ux, uy, vx, vy, markerSizePx * 0.72f, markerSizePx * 0.72f),
        )
        val cornerScore = cornerPoints.map { grayCoverage(reader, it, radius) }.average().toFloat()
        val bodyScore = bodyPoints.map { grayCoverage(reader, it, radius) }.average().toFloat()
        val outsideScore = 1f - outsidePoints.map { grayCoverage(reader, it, radius) }.average().toFloat()
        return (cornerScore * 0.46f + bodyScore * 0.36f + outsideScore * 0.18f).coerceIn(0f, 1f)
    }

    private fun triangleDeviceShapeScore(
        reader: YuvReader,
        center: ImagePoint,
        markerSizePx: Float,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
    ): Float {
        val radius = (markerSizePx * 0.10f).toInt().coerceIn(1, 5)
        val positive = arrayOf(
            center.offset(ux, uy, vx, vy, 0f, -markerSizePx * 0.38f),
            center.offset(ux, uy, vx, vy, -markerSizePx * 0.32f, markerSizePx * 0.32f),
            center.offset(ux, uy, vx, vy, markerSizePx * 0.32f, markerSizePx * 0.32f),
            center.offset(ux, uy, vx, vy, 0f, markerSizePx * 0.24f),
            center.offset(ux, uy, vx, vy, 0f, 0f),
        )
        val negative = arrayOf(
            center.offset(ux, uy, vx, vy, -markerSizePx * 0.38f, -markerSizePx * 0.30f),
            center.offset(ux, uy, vx, vy, markerSizePx * 0.38f, -markerSizePx * 0.30f),
            center.offset(ux, uy, vx, vy, -markerSizePx * 0.58f, 0f),
            center.offset(ux, uy, vx, vy, markerSizePx * 0.58f, 0f),
            center.offset(ux, uy, vx, vy, 0f, -markerSizePx * 0.72f),
            center.offset(ux, uy, vx, vy, 0f, markerSizePx * 0.72f),
        )
        val hitScore = positive.map { grayCoverage(reader, it, radius) }.average().toFloat()
        val emptyScore = 1f - negative.map { grayCoverage(reader, it, radius) }.average().toFloat()
        return (hitScore * 0.64f + emptyScore * 0.36f).coerceIn(0f, 1f)
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

    private fun grayCoverage(reader: YuvReader, center: ImagePoint, radiusPx: Int): Float {
        var hits = 0
        var total = 0
        val cx = center.x.roundToInt()
        val cy = center.y.roundToInt()
        for (dy in -radiusPx..radiusPx) {
            for (dx in -radiusPx..radiusPx) {
                if (dx * dx + dy * dy > radiusPx * radiusPx) continue
                total++
                if (reader.isGrayMarker(cx + dx, cy + dy)) hits++
            }
        }
        return if (total == 0) 0f else hits.toFloat() / total
    }

    private fun candidateInsideRoi(
        reader: YuvReader,
        roi: RotatedRoi,
        start: ImagePoint,
        end: ImagePoint,
        slots: List<ImagePoint>,
    ): Boolean {
        val points = slots + start + end
        val margin = max(2f, hypot(end.x - start.x, end.y - start.y) * 0.035f)
        return points.all { point ->
            val rotated = reader.imageToRotated(point)
            rotated.x >= roi.left + margin &&
                rotated.x <= roi.left + roi.width - margin &&
                rotated.y >= roi.top + margin &&
                rotated.y <= roi.top + roi.height - margin
        }
    }

    private fun brightIntrusionsNearLine(reader: YuvReader, start: ImagePoint, end: ImagePoint): Int {
        val dx = end.x - start.x
        val dy = end.y - start.y
        val distance = hypot(dx, dy)
        if (distance < 1f) return 99

        val nx = -dy / distance
        val ny = dx / distance
        var intrusions = 0
        val expected = constants.slotFractions()
        val samples = 28
        for (i in 2 until samples - 2) {
            val t = i.toFloat() / (samples - 1)
            if (expected.any { abs(it - t) < 0.045f }) continue
            val baseX = start.x + dx * t
            val baseY = start.y + dy * t
            var localBright = false
            for (offset in -2..2) {
                val x = (baseX + nx * offset * 3f).toInt()
                val y = (baseY + ny * offset * 3f).toInt()
                if (x !in 0 until reader.width || y !in 0 until reader.height) continue
                if (reader.y(x, y) > 178 && reader.blueDominance(x, y) < 18f) {
                    localBright = true
                    break
                }
            }
            if (localBright) intrusions++
        }
        return intrusions
    }

    private fun darkBackgroundScore(reader: YuvReader, start: ImagePoint, end: ImagePoint): Float {
        val dx = end.x - start.x
        val dy = end.y - start.y
        val distance = hypot(dx, dy)
        if (distance < 1f) return 0f

        val nx = -dy / distance
        val ny = dx / distance
        var dark = 0
        var total = 0
        val fractions = floatArrayOf(0.20f, 0.32f, 0.44f, 0.56f, 0.68f, 0.80f)
        val offsets = floatArrayOf(-8f, 8f)
        for (t in fractions) {
            val baseX = start.x + dx * t
            val baseY = start.y + dy * t
            for (offset in offsets) {
                val x = (baseX + nx * offset).toInt()
                val y = (baseY + ny * offset).toInt()
                if (x !in 0 until reader.width || y !in 0 until reader.height) continue
                val yy = reader.y(x, y)
                if (yy < 118) dark++
                total++
            }
        }
        return if (total == 0) 0f else dark.toFloat() / total
    }

    private fun grayMarkerComponents(reader: YuvReader, roi: RotatedRoi): List<Component> {
        val smallW = ReaderRoi.GRID_COLUMNS
        val smallH = ReaderRoi.GRID_ROWS
        val size = smallW * smallH
        ensureBuffers(size)

        var i = 0
        for (sy in 0 until smallH) {
            val ry = roi.top + (sy + 0.5f) * roi.height / smallH
            for (sx in 0 until smallW) {
                val rx = roi.left + (sx + 0.5f) * roi.width / smallW
                val p = reader.rotatedToImage(rx, ry)
                val x = p.x.roundToInt()
                val y = p.y.roundToInt()
                sampleXs[i] = x
                sampleYs[i] = y
                mask[i] = reader.isGrayMarker(x, y)
                seen[i] = false
                i++
            }
        }

        val components = ArrayList<Component>(8)
        for (idx in 0 until size) {
            if (!mask[idx] || seen[idx]) continue
            val c = flood(mask, seen, queue, idx, smallW, smallH)
            if (c.area >= 3) components.add(c)
        }
        return components
    }

    private fun Component.widthPx(): Int = maxX - minX + 1

    private fun Component.heightPx(): Int = maxY - minY + 1

    private fun Component.longSide(): Float = max(widthPx(), heightPx()).toFloat()

    private fun Component.shortSide(): Float = min(widthPx(), heightPx()).toFloat()

    private fun Component.anchorPoint(): ImagePoint {
        return ImagePoint(
            x = (minX + maxX) * 0.5f,
            y = (minY + maxY) * 0.5f,
        )
    }

    private fun Component.aspectScore(): Float {
        val ratio = shortSide() / max(1f, longSide())
        return ((ratio - 0.48f) / 0.52f).coerceIn(0f, 1f)
    }

    private fun Component.isPlausibleMarker(): Boolean {
        val w = widthPx()
        val h = heightPx()
        if (w !in 2..180 || h !in 2..180) return false
        if (area !in 3..1600) return false
        if (shortSide() / max(1f, longSide()) < 0.25f) return false
        return true
    }

    private fun Component.isSquareLike(): Boolean {
        return fill > 0.38f && cornerFill > 0.18f && aspectScore() > 0.30f
    }

    private fun Component.isTriangleLike(): Boolean {
        return fill in 0.16f..0.86f && aspectScore() > 0.22f
    }

    private fun Component.squareScore(): Float {
        val fillScore = ((fill - 0.36f) / 0.48f).coerceIn(0f, 1f)
        val cornerScore = (cornerFill / 0.55f).coerceIn(0f, 1f)
        return fillScore * 0.44f + aspectScore() * 0.30f + cornerScore * 0.26f
    }

    private fun Component.triangleScore(): Float {
        val fillScore = 1f - (abs(fill - 0.45f) / 0.38f).coerceIn(0f, 1f)
        return fillScore * 0.58f + aspectScore() * 0.42f
    }

    private fun ensureBuffers(size: Int) {
        if (mask.size < size || sampleXs.size < size || sampleYs.size < size) {
            mask = BooleanArray(size)
            seen = BooleanArray(size)
            queue = IntArray(size)
            sampleXs = IntArray(size)
            sampleYs = IntArray(size)
        }
    }

    private fun flood(
        mask: BooleanArray,
        seen: BooleanArray,
        queue: IntArray,
        start: Int,
        width: Int,
        height: Int,
    ): Component {
        var head = 0
        var tail = 0
        queue[tail++] = start
        seen[start] = true

        var count = 0
        var sumX = 0f
        var sumY = 0f
        var minX = Int.MAX_VALUE
        var minY = Int.MAX_VALUE
        var maxX = Int.MIN_VALUE
        var maxY = Int.MIN_VALUE
        var minGx = Int.MAX_VALUE
        var minGy = Int.MAX_VALUE
        var maxGx = Int.MIN_VALUE
        var maxGy = Int.MIN_VALUE
        var cornerHits = 0
        var cornerTotal = 0

        while (head < tail) {
            val idx = queue[head++]
            val sx = idx % width
            val sy = idx / width
            val ix = sampleXs[idx]
            val iy = sampleYs[idx]
            count++
            sumX += ix
            sumY += iy
            minX = min(minX, ix)
            minY = min(minY, iy)
            maxX = max(maxX, ix)
            maxY = max(maxY, iy)
            minGx = min(minGx, sx)
            minGy = min(minGy, sy)
            maxGx = max(maxGx, sx)
            maxGy = max(maxGy, sy)

            if (sx > 0) {
                val ni = idx - 1
                if (mask[ni] && !seen[ni]) {
                    seen[ni] = true
                    queue[tail++] = ni
                }
            }
            if (sx + 1 < width) {
                val ni = idx + 1
                if (mask[ni] && !seen[ni]) {
                    seen[ni] = true
                    queue[tail++] = ni
                }
            }
            if (sy > 0) {
                val ni = idx - width
                if (mask[ni] && !seen[ni]) {
                    seen[ni] = true
                    queue[tail++] = ni
                }
            }
            if (sy + 1 < height) {
                val ni = idx + width
                if (mask[ni] && !seen[ni]) {
                    seen[ni] = true
                    queue[tail++] = ni
                }
            }
        }

        val boxArea = max(1, (maxGx - minGx + 1) * (maxGy - minGy + 1))
        val gxSpan = max(1, maxGx - minGx + 1)
        val gySpan = max(1, maxGy - minGy + 1)
        for (gy in minGy..maxGy) {
            for (gx in minGx..maxGx) {
                val xEdge = min(gx - minGx, maxGx - gx)
                val yEdge = min(gy - minGy, maxGy - gy)
                if (xEdge >= max(1, gxSpan / 4) || yEdge >= max(1, gySpan / 4)) continue
                cornerTotal++
                val idx = gy * width + gx
                if (idx in mask.indices && mask[idx]) cornerHits++
            }
        }
        return Component(
            cx = sumX / count,
            cy = sumY / count,
            area = count,
            minX = minX,
            minY = minY,
            maxX = maxX,
            maxY = maxY,
            fill = count.toFloat() / boxArea,
            cornerFill = if (cornerTotal == 0) 0f else cornerHits.toFloat() / cornerTotal,
        )
    }

    private fun blueScore(reader: YuvReader, center: ImagePoint, radiusPx: Int): Float {
        var score = 0f
        var samples = 0
        val cx = center.x.toInt()
        val cy = center.y.toInt()
        for (dy in -radiusPx..radiusPx step 2) {
            for (dx in -radiusPx..radiusPx step 2) {
                if (dx * dx + dy * dy > radiusPx * radiusPx) continue
                val x = cx + dx
                val y = cy + dy
                if (x !in 0 until reader.width || y !in 0 until reader.height) continue
                val yy = reader.y(x, y)
                val u = reader.u(x, y)
                val v = reader.v(x, y)
                score += (u - v) + (yy - 45) * 0.12f
                samples++
            }
        }
        return if (samples == 0) 0f else score / samples
    }

    private fun lumaScore(reader: YuvReader, center: ImagePoint, radiusPx: Int): Float {
        var score = 0f
        var samples = 0
        val cx = center.x.toInt()
        val cy = center.y.toInt()
        for (dy in -radiusPx..radiusPx) {
            for (dx in -radiusPx..radiusPx) {
                if (dx * dx + dy * dy > radiusPx * radiusPx) continue
                val x = cx + dx
                val y = cy + dy
                if (x !in 0 until reader.width || y !in 0 until reader.height) continue
                score += reader.y(x, y)
                samples++
            }
        }
        return if (samples == 0) 0f else score / samples
    }

    private fun ledSearchScore(reader: YuvReader, center: ImagePoint, radiusPx: Int): Float {
        val luma = lumaScore(reader, center, radiusPx)
        val blue = blueScore(reader, center, radiusPx)
        return (luma - 70f) * 0.45f + blue * 0.55f
    }

    private fun grayScore(reader: YuvReader, center: ImagePoint, radiusPx: Int): Float {
        var score = 0f
        var samples = 0
        val cx = center.x.toInt()
        val cy = center.y.toInt()
        for (dy in -radiusPx..radiusPx step 2) {
            for (dx in -radiusPx..radiusPx step 2) {
                if (dx * dx + dy * dy > radiusPx * radiusPx) continue
                val x = cx + dx
                val y = cy + dy
                if (x !in 0 until reader.width || y !in 0 until reader.height) continue
                val yy = reader.y(x, y)
                if (yy <= 125) continue
                val u = reader.u(x, y)
                val v = reader.v(x, y)
                val chromaNeutrality = 36 - abs(u - 128) - abs(v - 128)
                score += max(0, chromaNeutrality) + max(0, yy - 125) * 0.18f
                samples++
            }
        }
        return if (samples == 0) 0f else score / samples
    }

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

        fun grayNeutrality(x: Int, y: Int): Float {
            return (42 - abs(u(x, y) - 128) - abs(v(x, y) - 128)).coerceAtLeast(0).toFloat()
        }

        fun isGrayMarker(x: Int, y: Int): Boolean {
            if (x !in 0 until width || y !in 0 until height) return false
            val yy = this.y(x, y)
            if (yy <= 115) return false
            return abs(u(x, y) - 128) < 42 && abs(v(x, y) - 128) < 42
        }
    }

    private class GeometryConstants {
        private val ledMm = 3f
        private val gapMm = 2.5f
        private val markerMm = 4f
        private val markerGapMm = 4f
        private val stepMm = ledMm + gapMm
        private val markerDistanceMm = markerMm + markerGapMm * 2 + ledMm + stepMm * 4
        private val firstLedOffsetMm = markerMm / 2f + markerGapMm + ledMm / 2f

        fun slotFractions(): List<Float> {
            return List(5) { index -> (firstLedOffsetMm + index * stepMm) / markerDistanceMm }
        }

        fun firstSlotFraction(): Float = firstLedOffsetMm / markerDistanceMm

        fun markerDistanceToSizeRatio(): Float = markerDistanceMm / markerMm

        fun slotStepFraction(): Float = stepMm / markerDistanceMm
    }

    private companion object {
        const val MAX_GEOMETRY_HOLD_FRAMES = 8
        const val LOST_TRACK_SCAN_PERIOD = 3
        const val TRACK_RESCAN_PERIOD = 12
    }
}
