package com.example.bottleneckreader

import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

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
    )

    private data class MarkerPair(
        val start: Component,
        val end: Component,
        val score: Float,
    )

    private data class Geometry(
        val start: ImagePoint,
        val end: ImagePoint,
        val slots: List<ImagePoint>,
        val slotStepPx: Float,
    )

    private data class SlotLayout(
        val slots: List<ImagePoint>,
        val stepPx: Float,
        val ledHits: Int,
    )

    private data class AxisHit(
        val index: Int,
        val fraction: Float,
        val score: Float,
    )

    private val constants = GeometryConstants()
    private var previousGeometry: Geometry? = null
    private var frameIndex = 0

    private var mask = BooleanArray(0)
    private var seen = BooleanArray(0)
    private var queue = IntArray(0)

    fun decode(image: ImageProxy): DetectionFrame? {
        val reader = YuvReader(image)
        frameIndex++

        val fast = previousGeometry?.let { geometry ->
            if (frameIndex % TRACK_RESCAN_PERIOD != 0 && markersStillVisible(reader, geometry)) {
                decodeWithGeometry(reader, geometry)
            } else {
                null
            }
        }
        if (fast != null) return fast

        if (previousGeometry == null && frameIndex % LOST_TRACK_SCAN_PERIOD != 0) {
            return null
        }

        val geometry = findGeometry(reader) ?: run {
            previousGeometry = null
            return null
        }
        previousGeometry = geometry
        return decodeWithGeometry(reader, geometry)
    }

    fun resetTracking() {
        previousGeometry = null
        frameIndex = 0
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
        )
    }

    private fun markersStillVisible(reader: YuvReader, geometry: Geometry): Boolean {
        val startGray = grayScore(reader, geometry.start, radiusPx = 5)
        val endGray = grayScore(reader, geometry.end, radiusPx = 6)
        return startGray > 24f && endGray > 20f &&
            darkBackgroundScore(reader, geometry.start, geometry.end) > 0.62f &&
            brightIntrusionsNearLine(reader, geometry.start, geometry.end) <= 2
    }

    private fun findGeometry(reader: YuvReader): Geometry? {
        return findGeometry(reader, stride = 4) ?: findGeometry(reader, stride = 2)
    }

    private fun findGeometry(reader: YuvReader, stride: Int): Geometry? {
        val markers = whiteMarkerComponents(reader, stride)
            .filter { it.isPlausibleMarker() }
            .sortedByDescending { it.area }
            .take(24)
        if (markers.size < 2) return null

        val pair = bestMarkerPair(reader, markers) ?: return null
        val square = pair.start
        val triangle = pair.end

        val start = square.anchorPoint()
        val end = triangle.anchorPoint()
        val dx = end.x - start.x
        val dy = end.y - start.y
        val distance = hypot(dx, dy)
        if (distance < 32f) return null

        val layout = fitSlotsOnAxis(reader, start, end)
        val background = darkBackgroundScore(reader, start, end)
        if (layout.ledHits < 2 && background < 0.93f) return null

        return Geometry(
            start = start,
            end = end,
            slots = layout.slots,
            slotStepPx = layout.stepPx,
        )
    }

    private fun fitSlotsOnAxis(reader: YuvReader, start: ImagePoint, end: ImagePoint): SlotLayout {
        val dx = end.x - start.x
        val dy = end.y - start.y
        val distance = hypot(dx, dy)
        if (distance < 1f) {
            return SlotLayout(emptyList(), stepPx = 0f, ledHits = 0)
        }

        val baseFirst = constants.firstSlotFraction()
        val baseStep = constants.slotStepFraction()
        val hits = ArrayList<AxisHit>(5)
        val searchSpan = baseStep * 0.42f
        val searchSteps = 8
        val sampleRadius = (distance * baseStep * 0.11f).toInt().coerceIn(2, 5)

        for (index in 0 until 5) {
            val expected = baseFirst + index * baseStep
            var bestFraction = expected
            var bestScore = Float.NEGATIVE_INFINITY
            for (offset in -searchSteps..searchSteps) {
                val fraction = expected + searchSpan * offset / searchSteps
                val point = ImagePoint(start.x + dx * fraction, start.y + dy * fraction)
                val score = ledSearchScore(reader, point, radiusPx = sampleRadius)
                if (score > bestScore) {
                    bestScore = score
                    bestFraction = fraction
                }
            }
            if (bestScore > 24f) {
                hits.add(AxisHit(index = index, fraction = bestFraction, score = bestScore))
            }
        }

        var step = baseStep
        if (hits.size >= 2) {
            var sum = 0f
            var weight = 0f
            for (i in 0 until hits.size - 1) {
                for (j in i + 1 until hits.size) {
                    val indexDelta = hits[j].index - hits[i].index
                    if (indexDelta <= 0) continue
                    val pairWeight = min(hits[i].score, hits[j].score).coerceAtLeast(1f)
                    sum += ((hits[j].fraction - hits[i].fraction) / indexDelta) * pairWeight
                    weight += pairWeight
                }
            }
            if (weight > 0f) {
                step = (sum / weight).coerceIn(baseStep * 0.84f, baseStep * 1.16f)
            }
        }

        var first = baseFirst
        if (hits.isNotEmpty()) {
            var sum = 0f
            var weight = 0f
            hits.forEach { hit ->
                val w = hit.score.coerceAtLeast(1f)
                sum += (hit.fraction - hit.index * step) * w
                weight += w
            }
            if (weight > 0f) {
                first = (sum / weight).coerceIn(baseFirst - baseStep * 0.42f, baseFirst + baseStep * 0.42f)
            }
        }

        val slots = List(5) { index ->
            val fraction = first + index * step
            ImagePoint(start.x + dx * fraction, start.y + dy * fraction)
        }
        val ledHits = slots.count { ledSearchScore(reader, it, radiusPx = sampleRadius) > 24f }
        return SlotLayout(
            slots = slots,
            stepPx = distance * step,
            ledHits = ledHits,
        )
    }

    private fun bestMarkerPair(reader: YuvReader, markers: List<Component>): MarkerPair? {
        var best: MarkerPair? = null
        for (start in markers) {
            if (!start.isSquareLike()) continue
            for (end in markers) {
                if (start === end || !end.isTriangleLike()) continue
                val pair = scoreMarkerPair(reader, start, end) ?: continue
                if (best == null || pair.score > best.score) best = pair
            }
        }
        return best?.takeIf { it.score >= 5.15f }
    }

    private fun scoreMarkerPair(reader: YuvReader, start: Component, end: Component): MarkerPair? {
        val startAnchor = start.anchorPoint()
        val endAnchor = end.anchorPoint()
        val dx = endAnchor.x - startAnchor.x
        val dy = endAnchor.y - startAnchor.y
        val distance = hypot(dx, dy)
        if (distance < 36f) return null

        val avgMarkerSize = (start.longSide() + end.longSide()) * 0.5f
        val distanceToMarkerRatio = distance / max(1f, avgMarkerSize)
        if (distanceToMarkerRatio !in 6.8f..13.2f) return null

        val sizeRatio = min(start.longSide(), end.longSide()) / max(start.longSide(), end.longSide())
        if (sizeRatio < 0.58f) return null

        val backgroundScore = darkBackgroundScore(reader, startAnchor, endAnchor)
        if (backgroundScore < 0.72f) return null

        val layout = fitSlotsOnAxis(
            reader = reader,
            start = startAnchor,
            end = endAnchor,
        )
        val ledHits = layout.ledHits
        val brightIntrusions = brightIntrusionsNearLine(reader, startAnchor, endAnchor)
        if (brightIntrusions > 2) return null
        if (ledHits < 2 && backgroundScore < 0.88f) return null

        val squareScore = start.squareScore()
        val startTriangleScore = start.triangleScore()
        if (squareScore < startTriangleScore + 0.10f) return null

        val triangleScore = end.triangleScore()
        val endSquareScore = end.squareScore()
        if (triangleScore < endSquareScore + 0.06f) return null

        val ratioScore = 1f - min(1f, abs(distanceToMarkerRatio - constants.markerDistanceToSizeRatio()) / 7f)
        val score = squareScore * 1.6f +
            triangleScore * 1.6f +
            backgroundScore * 1.4f +
            ratioScore +
            ledHits * 0.45f -
            brightIntrusions * 0.55f

        return MarkerPair(start = start, end = end, score = score)
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

    private fun whiteMarkerComponents(reader: YuvReader, stride: Int): List<Component> {
        val smallW = reader.width / stride
        val smallH = reader.height / stride
        val size = smallW * smallH
        ensureBuffers(size)

        var i = 0
        for (sy in 0 until smallH) {
            val y = sy * stride
            for (sx in 0 until smallW) {
                val x = sx * stride
                mask[i] = reader.isWhiteMarker(x, y)
                seen[i] = false
                i++
            }
        }

        val components = ArrayList<Component>(8)
        for (idx in 0 until size) {
            if (!mask[idx] || seen[idx]) continue
            val c = flood(mask, seen, queue, idx, smallW, smallH, stride)
            if (c.area >= 4) components.add(c)
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
        if (w !in 4..96 || h !in 4..96) return false
        if (area !in 3..900) return false
        if (shortSide() / max(1f, longSide()) < 0.42f) return false
        return true
    }

    private fun Component.isSquareLike(): Boolean {
        return fill > 0.66f && aspectScore() > 0.52f
    }

    private fun Component.isTriangleLike(): Boolean {
        return fill in 0.24f..0.66f && aspectScore() > 0.35f
    }

    private fun Component.squareScore(): Float {
        val fillScore = ((fill - 0.64f) / 0.28f).coerceIn(0f, 1f)
        return fillScore * 0.72f + aspectScore() * 0.28f
    }

    private fun Component.triangleScore(): Float {
        val fillScore = 1f - (abs(fill - 0.46f) / 0.24f).coerceIn(0f, 1f)
        return fillScore * 0.76f + aspectScore() * 0.24f
    }

    private fun ensureBuffers(size: Int) {
        if (mask.size < size) {
            mask = BooleanArray(size)
            seen = BooleanArray(size)
            queue = IntArray(size)
        }
    }

    private fun flood(
        mask: BooleanArray,
        seen: BooleanArray,
        queue: IntArray,
        start: Int,
        width: Int,
        height: Int,
        stride: Int,
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

        while (head < tail) {
            val idx = queue[head++]
            val sx = idx % width
            val sy = idx / width
            val ix = sx * stride
            val iy = sy * stride
            count++
            sumX += ix
            sumY += iy
            minX = min(minX, ix)
            minY = min(minY, iy)
            maxX = max(maxX, ix)
            maxY = max(maxY, iy)

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

        val boxArea = max(1, ((maxX - minX) / stride + 1) * ((maxY - minY) / stride + 1))
        return Component(
            cx = sumX / count,
            cy = sumY / count,
            area = count,
            minX = minX,
            minY = minY,
            maxX = maxX,
            maxY = maxY,
            fill = count.toFloat() / boxArea,
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

        fun isWhiteMarker(x: Int, y: Int): Boolean {
            val yy = this.y(x, y)
            if (yy <= 150) return false
            return abs(u(x, y) - 128) < 26 && abs(v(x, y) - 128) < 26
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
        const val LOST_TRACK_SCAN_PERIOD = 3
        const val TRACK_RESCAN_PERIOD = 12
    }
}
