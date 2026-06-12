package com.example.bottleneckreader

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import java.util.Locale

@Composable
fun DiagnosticsOverlay(
    viewModel: ReaderViewModel,
    frame: DetectionFrame?,
) {
    val decoderTiming by viewModel.decoderTiming.collectAsStateWithLifecycle()
    Column(
        modifier = Modifier
            .padding(top = 18.dp, start = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        DebugPanel(lines = frame?.debugLines.orEmpty())
        DecoderTimingPanel(timing = decoderTiming)
    }
}

@Composable
private fun DecoderTimingPanel(
    timing: DecoderTimingWindow,
    modifier: Modifier = Modifier,
) {
    if (timing.samplesMs.isEmpty()) return
    Row(
        modifier = modifier
            .background(Color(0xB8000000), RoundedCornerShape(6.dp))
            .padding(horizontal = 10.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Canvas(
            modifier = Modifier
                .width(210.dp)
                .height(58.dp),
        ) {
            val samples = timing.samplesMs
            if (samples.size < 2) return@Canvas
            val maxValue = maxOf(timing.maxMs, 1f)
            val path = Path()
            samples.forEachIndexed { index, sample ->
                val x = size.width * index / samples.lastIndex.coerceAtLeast(1)
                val y = size.height - (sample / maxValue).coerceIn(0f, 1f) * size.height
                if (index == 0) {
                    path.moveTo(x, y)
                } else {
                    path.lineTo(x, y)
                }
            }
            drawRect(
                color = Color(0x55E8EAEE),
                size = size,
                style = Stroke(width = 1.dp.toPx()),
            )
            drawPath(
                path = path,
                color = Color(0xFFFFEB3B),
                style = Stroke(width = 1.6.dp.toPx()),
            )
        }
        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            TimingLabel("avg", timing.avgMs)
            TimingLabel("max", timing.maxMs)
            TimingLabel("min", timing.minMs)
        }
    }
}

@Composable
private fun TimingLabel(label: String, value: Float) {
    Text(
        text = "$label ${String.format(Locale.US, "%.1f", value)} ms",
        color = Color(0xFFE8EAEE),
        fontSize = 11.sp,
        lineHeight = 13.sp,
        fontWeight = FontWeight.Medium,
    )
}

@Composable
private fun DebugPanel(
    lines: List<String>,
    modifier: Modifier = Modifier,
) {
    if (lines.isEmpty()) return
    Column(
        modifier = modifier
            .widthIn(max = 340.dp)
            .background(Color(0xB8000000), RoundedCornerShape(6.dp))
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        lines.take(10).forEach { line ->
            Text(
                text = line,
                color = Color(0xFFE8EAEE),
                fontSize = 11.sp,
                lineHeight = 13.sp,
                fontWeight = FontWeight.Medium,
            )
        }
    }
}
