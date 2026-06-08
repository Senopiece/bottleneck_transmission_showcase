package com.example.bottleneckreader

import androidx.camera.core.ImageProxy
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

    private data class Geometry(
        val start: ImagePoint,
        val end: ImagePoint,
        val slots: List<ImagePoint>,
    )

    private val constants = GeometryConstants()
    private var previousGeometry: Geometry? = null

    private var mask = BooleanArray(0)
    private var seen = BooleanArray(0)
    private var queue = IntArray(0)

    fun decode(image: ImageProxy): DetectionFrame? {
        val fast = previousGeometry?.let { geometry ->
            if (markersStillVisible(image, geometry)) decodeWithGeometry(image, geometry)
            else null
        }
        if (fast != null) return fast

        val geometry = findGeometry(image) ?: run {
            previousGeometry = null
            return null
        }
        previousGeometry = geometry
        return decodeWithGeometry(image, geometry)
    }

    fun resetTracking() {
        previousGeometry = null
    }

    private fun decodeWithGeometry(image: ImageProxy, geometry: Geometry): DetectionFrame {
        val bits = buildString(capacity = 5) {
            geometry.slots.forEach { slot ->
                append(if (blueScore(image, slot, radiusPx = 5) > 24f) '1' else '0')
            }
        }
        return DetectionFrame(
            timestampNs = image.imageInfo.timestamp,
            imageWidth = image.width,
            imageHeight = image.height,
            rotationDegrees = image.imageInfo.rotationDegrees,
            bits = bits,
            slots = geometry.slots.mapIndexed { index, point ->
                LedSlot(
                    imagePoint = point,
                    bitIndex = 4 - index,
                    isFirst = index == 0,
                )
            },
        )
    }

    private fun markersStillVisible(image: ImageProxy, geometry: Geometry): Boolean {
        val startGray = grayScore(image, geometry.start, radiusPx = 5)
        val endGray = grayScore(image, geometry.end, radiusPx = 6)
        return startGray > 18f && endGray > 12f
    }

    private fun findGeometry(image: ImageProxy): Geometry? {
        val gray = grayComponents(image)
            .filter { it.area >= 5 }
            .sortedByDescending { it.area }
            .take(12)
        if (gray.size < 2) return null

        val square = gray.maxByOrNull { it.fill * min(it.maxX - it.minX + 1, it.maxY - it.minY + 1) }
            ?: return null
        val triangle = gray
            .filter { it !== square }
            .maxByOrNull { triangleScore(square, it) }
            ?: return null

        val start = ImagePoint(square.cx, square.cy)
        val end = ImagePoint(triangle.cx, triangle.cy)
        val dx = end.x - start.x
        val dy = end.y - start.y
        val distance = hypot(dx, dy)
        if (distance < 32f) return null

        val slotFractions = constants.slotFractions()
        val slots = slotFractions.map { t ->
            ImagePoint(start.x + dx * t, start.y + dy * t)
        }

        val activeHits = slots.count { blueScore(image, it, radiusPx = 6) > 18f }
        if (activeHits == 0) return null

        return Geometry(start = start, end = end, slots = slots)
    }

    private fun triangleScore(square: Component, candidate: Component): Float {
        val dx = candidate.cx - square.cx
        val dy = candidate.cy - square.cy
        val distance = hypot(dx, dy)
        val areaScore = min(candidate.area, 80) / 80f
        val triangularFill = 1f - abs(candidate.fill - 0.52f)
        val horizontalBias = abs(dx) / max(1f, abs(dy) + 1f)
        return distance * 0.04f + areaScore + triangularFill + horizontalBias * 0.3f
    }

    private fun grayComponents(image: ImageProxy): List<Component> {
        val stride = 2
        val smallW = image.width / stride
        val smallH = image.height / stride
        val size = smallW * smallH
        ensureBuffers(size)

        var i = 0
        for (sy in 0 until smallH) {
            val y = sy * stride
            for (sx in 0 until smallW) {
                val x = sx * stride
                val sample = yuvAt(image, x, y)
                mask[i] = sample.isGrayMarker()
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

            fun push(nx: Int, ny: Int) {
                if (nx !in 0 until width || ny !in 0 until height) return
                val ni = ny * width + nx
                if (!mask[ni] || seen[ni]) return
                seen[ni] = true
                queue[tail++] = ni
            }
            push(sx - 1, sy)
            push(sx + 1, sy)
            push(sx, sy - 1)
            push(sx, sy + 1)
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

    private fun blueScore(image: ImageProxy, center: ImagePoint, radiusPx: Int): Float {
        var score = 0f
        var samples = 0
        val cx = center.x.toInt()
        val cy = center.y.toInt()
        for (dy in -radiusPx..radiusPx step 2) {
            for (dx in -radiusPx..radiusPx step 2) {
                if (dx * dx + dy * dy > radiusPx * radiusPx) continue
                val x = cx + dx
                val y = cy + dy
                if (x !in 0 until image.width || y !in 0 until image.height) continue
                val s = yuvAt(image, x, y)
                score += (s.u - s.v) + (s.y - 45) * 0.12f
                samples++
            }
        }
        return if (samples == 0) 0f else score / samples
    }

    private fun grayScore(image: ImageProxy, center: ImagePoint, radiusPx: Int): Float {
        var score = 0f
        var samples = 0
        val cx = center.x.toInt()
        val cy = center.y.toInt()
        for (dy in -radiusPx..radiusPx step 2) {
            for (dx in -radiusPx..radiusPx step 2) {
                if (dx * dx + dy * dy > radiusPx * radiusPx) continue
                val x = cx + dx
                val y = cy + dy
                if (x !in 0 until image.width || y !in 0 until image.height) continue
                val s = yuvAt(image, x, y)
                val chromaNeutrality = 32 - abs(s.u - 128) - abs(s.v - 128)
                score += max(0, chromaNeutrality) + max(0, s.y - 40) * 0.25f
                samples++
            }
        }
        return if (samples == 0) 0f else score / samples
    }

    private fun yuvAt(image: ImageProxy, x: Int, y: Int): YuvSample {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yy = yPlane.buffer.get(y * yPlane.rowStride + x).toInt() and 0xff
        val uvX = x / 2
        val uvY = y / 2
        val u = uPlane.buffer.get(uvY * uPlane.rowStride + uvX * uPlane.pixelStride).toInt() and 0xff
        val v = vPlane.buffer.get(uvY * vPlane.rowStride + uvX * vPlane.pixelStride).toInt() and 0xff
        return YuvSample(yy, u, v)
    }

    private data class YuvSample(
        val y: Int,
        val u: Int,
        val v: Int,
    ) {
        fun isGrayMarker(): Boolean {
            return y > 48 &&
                abs(u - 128) < 18 &&
                abs(v - 128) < 18
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
    }
}
