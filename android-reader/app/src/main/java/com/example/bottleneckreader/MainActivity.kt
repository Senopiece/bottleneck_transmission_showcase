package com.example.bottleneckreader

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.EaseInQuad
import androidx.compose.animation.core.EaseOutQuad
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.clickable
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.delay

private val PercentFontFamily = FontFamily(
    Font(R.font.oswald, weight = FontWeight.Medium),
)

private val DarkOverlaySurface = Color(0x90101010)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        hideNavigationBar()
        setContent {
            MaterialTheme {
                Surface(color = Color.Black) {
                    ReaderApp()
                }
            }
        }
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) hideNavigationBar()
    }

    private fun hideNavigationBar() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.navigationBars())
        }
    }
}

@Composable
private fun ReaderApp(viewModel: ReaderViewModel = viewModel()) {
    val context = LocalContext.current
    var cameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED,
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        cameraPermission = granted
        if (!granted) {
            viewModel.onReaderEvent(
                ReaderEvent.CameraIssue(
                    CameraProblem(
                        title = "Camera permission denied",
                        message = "Camera access is required to read the optical device.",
                    ),
                ),
            )
        }
    }

    LaunchedEffect(Unit) {
        if (!cameraPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    val lifecycleOwner = LocalLifecycleOwner.current
    var stoppedOnce by rememberSaveable { mutableStateOf(false) }
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_STOP -> {
                    stoppedOnce = true
                    if (cameraPermission) viewModel.markStreamInterrupted()
                }
                Lifecycle.Event.ON_START -> {
                    if (stoppedOnce && cameraPermission) {
                        viewModel.resumeCamera("foreground after pause")
                        stoppedOnce = false
                    }
                }
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    val frame by viewModel.frame.collectAsStateWithLifecycle()
    val decodeProgress by viewModel.decodeProgress.collectAsStateWithLifecycle()
    val decodedMessage by viewModel.decodedMessage.collectAsStateWithLifecycle()
    val resultEventId by viewModel.resultEventId.collectAsStateWithLifecycle()
    val failureEventId by viewModel.failureEventId.collectAsStateWithLifecycle()
    val failureEventPhase by viewModel.failureEventPhase.collectAsStateWithLifecycle()
    val liveMessageBits by viewModel.liveMessageBits.collectAsStateWithLifecycle()
    val liveBitConfidences by viewModel.liveBitConfidences.collectAsStateWithLifecycle()
    val liveDecoding by viewModel.liveDecoding.collectAsStateWithLifecycle()
    val problem by viewModel.problem.collectAsStateWithLifecycle()
    val restartToken by viewModel.restartToken.collectAsStateWithLifecycle()

    LaunchedEffect(failureEventId) {
        if (failureEventId != 0L) {
            vibrate(context, VIBRATION_FAIL_MS)
        }
    }
    LaunchedEffect(resultEventId) {
        if (resultEventId != 0L) {
            vibrate(context, VIBRATION_SUCCESS_MS)
        }
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        val result = decodedMessage
        if (cameraPermission && result == null) {
            CameraPreview(
                restartToken = restartToken,
                eventSink = viewModel::onReaderEvent,
                modifier = Modifier.fillMaxSize(),
            )
        }

        DetectionOverlay(
            frame = frame,
            modifier = Modifier.fillMaxSize(),
        )

        Box(Modifier.fillMaxSize().safeDrawingPadding()) {
            DecodeOverlay(
                progress = decodeProgress,
                liveMessageBits = liveMessageBits,
                liveBitConfidences = liveBitConfidences,
                liveDecoding = liveDecoding,
                failureEventId = failureEventId,
                failureEventPhase = failureEventPhase,
                onStop = viewModel::stopDecoding,
            )

            DiagnosticsOverlay(viewModel = viewModel, frame = frame)
        }

        AnimatedVisibility(
            visible = result != null,
            enter = fadeIn(tween(180, easing = EaseInQuad)),
            exit = slideOutHorizontally(targetOffsetX = { it }, animationSpec = tween(240)) +
                fadeOut(tween(180)),
        ) {
            if (result != null) {
                DecodeResultScreen(
                    message = result,
                    onBack = viewModel::closeDecodedMessage,
                    modifier = Modifier.fillMaxSize().safeDrawingPadding(),
                )
            }
        }

        if (!cameraPermission) {
            Text(
                text = "Camera permission required",
                color = Color.White,
                modifier = Modifier
                    .align(Alignment.Center)
                    .safeDrawingPadding(),
            )
        }

        val activeProblem = problem
        if (activeProblem != null) {
            Text(
                text = "${activeProblem.title}\n${activeProblem.message}",
                color = Color.White,
                fontSize = 18.sp,
                lineHeight = 24.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier
                    .align(Alignment.Center)
                    .safeDrawingPadding()
                    .padding(28.dp),
            )
        }
    }
}

@Composable
private fun CameraPreview(
    restartToken: Int,
    eventSink: (ReaderEvent) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var previewView by remember { mutableStateOf<PreviewView?>(null) }

    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            PreviewView(ctx).also { previewView = it }
        },
    )

    val view = previewView
    DisposableEffect(view, lifecycleOwner, restartToken) {
        if (view == null) {
            onDispose { }
        } else {
            val reader = CameraReader(
                context = context.applicationContext,
                lifecycleOwner = lifecycleOwner,
                previewView = view,
                eventSink = eventSink,
            )
            reader.start()
            onDispose { reader.release() }
        }
    }
}

@Composable
private fun DetectionOverlay(
    frame: DetectionFrame?,
    modifier: Modifier = Modifier,
) {
    Canvas(modifier) {
        val roiWidth = size.width * ReaderRoi.WIDTH_FRACTION
        val roiHeight = (roiWidth * ReaderRoi.ROI_ASPECT_RATIO).coerceAtMost(size.height)
        val roiTopLeft = Offset(
            x = (size.width - roiWidth) * 0.5f,
            y = (size.height - roiHeight) * 0.5f,
        )

        if (frame.isAcquireMode()) {
            drawAcquireGuidePattern(
                roiTopLeft = roiTopLeft,
                roiSize = Size(roiWidth, roiHeight),
                strokeWidth = 1.5.dp.toPx(),
            )
        }

        val current = frame ?: return@Canvas
        val overlay = current.toOverlay(size)
        overlay.markers.forEach { marker ->
            val markerStroke = Stroke(width = 1.5.dp.toPx())
            val ax = marker.alongPoint.x - marker.point.x
            val ay = marker.alongPoint.y - marker.point.y
            val length = kotlin.math.hypot(ax, ay).coerceAtLeast(1f)
            val ux = ax / length
            val uy = ay / length
            val vx = -uy
            val vy = ux
            when (marker.kind) {
                MarkerKind.StartSquare -> {
                    val half = marker.sizePx * 0.5f
                    val path = Path().apply {
                        val p0 = marker.point.offsetBy(ux, uy, vx, vy, -half, -half)
                        val p1 = marker.point.offsetBy(ux, uy, vx, vy, half, -half)
                        val p2 = marker.point.offsetBy(ux, uy, vx, vy, half, half)
                        val p3 = marker.point.offsetBy(ux, uy, vx, vy, -half, half)
                        moveTo(p0.x, p0.y)
                        lineTo(p1.x, p1.y)
                        lineTo(p2.x, p2.y)
                        lineTo(p3.x, p3.y)
                        close()
                    }
                    drawPath(path = path, color = Color(0xFFE8EAEE), style = markerStroke)
                }
                MarkerKind.EndTriangle -> {
                    val half = marker.sizePx * 0.38f
                    val path = Path().apply {
                        val apex = marker.point.offsetBy(ux, uy, vx, vy, 0f, -half)
                        val right = marker.point.offsetBy(ux, uy, vx, vy, half * 0.92f, half)
                        val left = marker.point.offsetBy(ux, uy, vx, vy, -half * 0.92f, half)
                        moveTo(apex.x, apex.y)
                        lineTo(right.x, right.y)
                        lineTo(left.x, left.y)
                        close()
                    }
                    drawPath(
                        path = path,
                        color = Color(0xFFE8EAEE),
                        style = markerStroke,
                    )
                }
            }
        }
        overlay.slots.forEach { slot ->
            val radius = slot.radiusPx.coerceIn(5.dp.toPx(), 14.dp.toPx())
            drawCircle(
                color = Color.Yellow,
                radius = radius,
                center = slot.point,
                style = Stroke(width = 2.dp.toPx()),
            )
            if (slot.isFirst) {
                drawCircle(
                    color = Color.Yellow,
                    radius = (radius * 0.28f).coerceAtLeast(3.dp.toPx()),
                    center = slot.point,
                )
            }
        }
    }
}

@Composable
private fun DecodeOverlay(
    progress: DecodeProgress,
    liveMessageBits: List<Boolean>,
    liveBitConfidences: List<Float>,
    liveDecoding: Boolean,
    failureEventId: Long,
    failureEventPhase: DecodePhase,
    onStop: () -> Unit,
) {
    val preambleActive = progress.visible && progress.phase == DecodePhase.Preamble
    var preambleExitFailed by remember { mutableStateOf(false) }
    LaunchedEffect(failureEventId, failureEventPhase) {
        if (failureEventId != 0L && failureEventPhase == DecodePhase.Preamble) {
            preambleExitFailed = true
            delay(160L)
            preambleExitFailed = false
        }
    }
    val preambleVisible = preambleActive && !progress.failed
    var preambleDisplayProgress by remember { mutableStateOf(0f) }
    LaunchedEffect(preambleActive, progress.confidenceProgress) {
        if (preambleActive) {
            preambleDisplayProgress = progress.confidenceProgress
        }
    }
    val decodeOnScreen = progress.visible && progress.phase == DecodePhase.Decoding
    val decodeActive = decodeOnScreen && !progress.failed
    var decodeSnapshot by remember {
        mutableStateOf(
            DecodeDisplaySnapshot(
                bits = liveMessageBits,
                confidences = liveBitConfidences,
                progress = progress.confidenceProgress,
                showButton = true,
            ),
        )
    }
    LaunchedEffect(decodeActive, liveMessageBits, liveBitConfidences, progress.confidenceProgress) {
        if (decodeActive) {
            decodeSnapshot = DecodeDisplaySnapshot(
                bits = liveMessageBits,
                confidences = liveBitConfidences,
                progress = progress.confidenceProgress,
                showButton = true,
            )
        }
    }
    var decodeExitFailed by remember { mutableStateOf(false) }
    LaunchedEffect(failureEventId, failureEventPhase) {
        if (failureEventId != 0L && failureEventPhase == DecodePhase.Decoding) {
            decodeExitFailed = true
            delay(160L)
            decodeExitFailed = false
        }
    }
    val decodeVisible = decodeOnScreen
    val decodeFailed = progress.failed || decodeExitFailed
    val decodeDisplay = if (!decodeActive) {
        decodeSnapshot
    } else {
        DecodeDisplaySnapshot(
            bits = liveMessageBits,
            confidences = liveBitConfidences,
            progress = progress.confidenceProgress,
            showButton = true,
        )
    }
    val frozenDecodeDisplay = if (decodeFailed) {
        decodeSnapshot
    } else {
        decodeDisplay
    }

    Box(Modifier.fillMaxSize()) {
        AnimatedVisibility(
            visible = preambleVisible,
            modifier = Modifier
                .align(Alignment.Center)
                .graphicsLayer { translationY = 76.dp.toPx() },
            enter = fadeIn(tween(80, easing = EaseInQuad)) +
                slideInVertically(
                    initialOffsetY = { it / 8 },
                    animationSpec = tween(80, easing = EaseOutQuad),
                ),
            exit = fadeOut(tween(80, easing = EaseInQuad)) +
                slideOutVertically(
                    targetOffsetY = { it / 8 },
                    animationSpec = tween(80, easing = EaseOutQuad),
                ),
        ) {
            PreambleStatus(
                progress = preambleDisplayProgress,
                failed = preambleExitFailed,
            )
        }

        AnimatedVisibility(
            visible = decodeVisible,
            modifier = Modifier.fillMaxSize(),
            enter = fadeIn(tween(80, easing = EaseInQuad)) +
                slideInVertically(
                    initialOffsetY = { it / 24 },
                    animationSpec = tween(80, easing = EaseOutQuad),
                ),
            exit = fadeOut(tween(80, easing = EaseInQuad)) +
                slideOutVertically(
                    targetOffsetY = { it / 24 },
                    animationSpec = tween(80, easing = EaseOutQuad),
                ),
        ) {
            DecodeBottomControls(
                display = frozenDecodeDisplay,
                failed = decodeFailed,
                onStop = onStop,
            )
        }
    }
}

private data class DecodeDisplaySnapshot(
    val bits: List<Boolean>,
    val confidences: List<Float>,
    val progress: Float,
    val showButton: Boolean,
)

@Composable
private fun PreambleStatus(
    progress: Float,
    failed: Boolean,
) {
    val textColor = if (failed) Color(0xFFFF453A) else Color.White.copy(alpha = 0.92f)
    val labelColor = if (failed) Color(0xFFFF453A) else Color.White.copy(alpha = 0.54f)
    Column(
        modifier = Modifier
            .width(128.dp)
            .shadow(
                elevation = 8.dp,
                shape = RoundedCornerShape(10.dp),
                ambientColor = Color.Black.copy(alpha = 0.18f),
                spotColor = Color.Black.copy(alpha = 0.26f),
            )
            .background(DarkOverlaySurface, RoundedCornerShape(10.dp))
            .padding(horizontal = 18.dp, vertical = 3.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy((-5).dp),
    ) {
        Text(
            text = "PREAMBLE",
            color = labelColor,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold,
        )
        PercentText(
            text = "${(progress.coerceIn(0f, 1f) * 100f).toInt()}%",
            color = textColor,
        )
    }
}

@Composable
private fun PercentText(
    text: String,
    color: Color,
    modifier: Modifier = Modifier,
    lineHeight: TextUnit = 34.sp,
) {
    Text(
        text = text,
        modifier = modifier,
        color = color,
        fontSize = 30.sp,
        lineHeight = lineHeight,
        fontFamily = PercentFontFamily,
        fontWeight = FontWeight.Medium,
    )
}

@Composable
private fun DecodeBottomControls(
    display: DecodeDisplaySnapshot,
    failed: Boolean,
    onStop: () -> Unit,
) {
    var redFlash by remember { mutableStateOf(false) }
    LaunchedEffect(failed) {
        if (failed) {
            redFlash = true
            delay(45L)
            redFlash = false
        } else {
            redFlash = false
        }
    }
    val flashFailed = failed || redFlash
    Box(Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(start = 54.dp, bottom = 104.dp),
            verticalAlignment = Alignment.Bottom,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            LiveBitCertaintyGrid(
                bits = display.bits,
                confidences = display.confidences,
                failed = flashFailed,
                modifier = Modifier.size(104.dp),
            )
            PercentText(
                text = "${(display.progress.coerceIn(0f, 1f) * 100f).toInt()}%",
                color = if (flashFailed) Color(0xCCFF453A) else Color.White.copy(alpha = 0.88f),
                lineHeight = 30.sp,
            )
        }

        if (display.showButton) {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(end = 54.dp, bottom = 176.dp)
                    .size(50.dp)
                    .shadow(
                        elevation = 8.dp,
                        shape = CircleShape,
                        ambientColor = Color.Black.copy(alpha = 0.16f),
                        spotColor = Color.Black.copy(alpha = 0.24f),
                    )
                    .background(DarkOverlaySurface, CircleShape)
                    .clickable(onClick = onStop),
                contentAlignment = Alignment.Center,
            ) {
                CancelIcon()
            }
        }
    }

}

@Composable
private fun CancelIcon(modifier: Modifier = Modifier) {
    Canvas(modifier = modifier.size(22.dp)) {
        val strokeWidth = 2.dp.toPx()
        val inset = 4.2.dp.toPx()
        drawLine(
            color = Color.White,
            start = Offset(inset, inset),
            end = Offset(size.width - inset, size.height - inset),
            strokeWidth = strokeWidth,
            cap = StrokeCap.Round,
        )
        drawLine(
            color = Color.White,
            start = Offset(size.width - inset, inset),
            end = Offset(inset, size.height - inset),
            strokeWidth = strokeWidth,
            cap = StrokeCap.Round,
        )
    }
}

@Composable
private fun LiveBitCertaintyGrid(
    bits: List<Boolean>,
    confidences: List<Float>,
    failed: Boolean,
    modifier: Modifier = Modifier,
) {
    val bitsCount = minOf(bits.size, 36)
    Canvas(
        modifier = modifier,
    ) {
        val gridSize = 6
        val gap = 2.dp.toPx()
        val cell = (minOf(size.width, size.height) - gap * (gridSize - 1)) / gridSize

        for (row in 0 until gridSize) {
            for (col in 0 until gridSize) {
                val index = row * gridSize + col
                if (index >= bitsCount) continue
                val bit = bits.getOrNull(index) == true
                val confidence = confidences.getOrNull(index) ?: 0f
                val intensity = confidence.coerceIn(0f, 1f)
                val fill = when {
                    failed -> Color(0x99FF453A)
                    intensity < 0.40f -> Color(0xFFB8BDC6).copy(alpha = 0.70f)
                    bit -> Color.Black.copy(alpha = 0.34f + intensity * 0.58f)
                    else -> Color.White.copy(alpha = 0.62f + intensity * 0.34f)
                }
                drawRect(
                    color = fill,
                    topLeft = Offset(col * (cell + gap), row * (cell + gap)),
                    size = Size(cell, cell),
                )
                drawRect(
                    color = Color.White.copy(alpha = 0.22f),
                    topLeft = Offset(col * (cell + gap), row * (cell + gap)),
                    size = Size(cell, cell),
                    style = Stroke(width = 1.dp.toPx()),
                )
            }
        }
    }
}

@Composable
private fun DecodeResultScreen(
    message: DecodedMessage,
    onBack: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier.background(Color.Black),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(24.dp),
        ) {
            DecodedMessageGrid(
                message = message,
                modifier = Modifier.size(220.dp),
            )
            Box(
                modifier = Modifier
                    .size(54.dp)
                    .shadow(
                        elevation = 10.dp,
                        shape = CircleShape,
                        ambientColor = Color.Black.copy(alpha = 0.18f),
                        spotColor = Color.Black.copy(alpha = 0.30f),
                    )
                    .background(Color.White.copy(alpha = 0.90f), CircleShape)
                    .clickable(onClick = onBack),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = "←",
                    color = Color.Transparent,
                    fontSize = 30.sp,
                    fontWeight = FontWeight.Bold,
                )
                BackArrowIcon()
            }
        }
    }

}

@Composable
private fun BackArrowIcon(modifier: Modifier = Modifier) {
    Canvas(modifier = modifier.size(25.dp)) {
        val stroke = 2.2.dp.toPx()
        val y = size.height * 0.5f
        val left = size.width * 0.22f
        val right = size.width * 0.78f
        val wing = size.width * 0.24f
        drawLine(
            color = Color(0xFF151515),
            start = Offset(right, y),
            end = Offset(left, y),
            strokeWidth = stroke,
            cap = StrokeCap.Round,
        )
        drawLine(
            color = Color(0xFF151515),
            start = Offset(left, y),
            end = Offset(left + wing, y - wing),
            strokeWidth = stroke,
            cap = StrokeCap.Round,
        )
        drawLine(
            color = Color(0xFF151515),
            start = Offset(left, y),
            end = Offset(left + wing, y + wing),
            strokeWidth = stroke,
            cap = StrokeCap.Round,
        )
    }
}

@Composable
private fun DecodedMessageGrid(
    message: DecodedMessage,
    modifier: Modifier = Modifier,
) {
    Canvas(
        modifier = modifier
            .background(Color.White, RoundedCornerShape(3.dp))
            .padding(8.dp),
    ) {
        val gap = 2.dp.toPx()
        val cell = (minOf(size.width, size.height) - gap * 5f) / 6f
        val bits = message.bits
        for (row in 0 until 6) {
            for (col in 0 until 6) {
                val index = row * 6 + col
                val on = index < bits.size && bits[index]
                drawRect(
                    color = if (on) Color.Black else Color.White,
                    topLeft = Offset(col * (cell + gap), row * (cell + gap)),
                    size = Size(cell, cell),
                )
                drawRect(
                    color = Color(0x33000000),
                    topLeft = Offset(col * (cell + gap), row * (cell + gap)),
                    size = Size(cell, cell),
                    style = Stroke(width = 1.dp.toPx()),
                )
            }
        }
    }
}

private fun DetectionFrame?.isAcquireMode(): Boolean = this?.isAcquireMode == true

private fun DrawScope.drawAcquireGuidePattern(
    roiTopLeft: Offset,
    roiSize: Size,
    strokeWidth: Float,
) {
    val distance = roiSize.width * AcquireGuideGeometry.patternDistanceFraction
    val markerSize = distance / AcquireGuideGeometry.markerDistanceToSizeRatio
    val ledRadius = markerSize * AcquireGuideGeometry.ledRadiusToMarkerSizeRatio
    val squareSize = markerSize * AcquireGuideGeometry.squareSizeToMarkerSizeRatio
    val triangleSize = markerSize * AcquireGuideGeometry.triangleSizeToMarkerSizeRatio
    val center = Offset(
        x = roiTopLeft.x + roiSize.width * 0.5f,
        y = roiTopLeft.y + roiSize.height * 0.5f,
    )
    val start = Offset(center.x - distance * 0.5f, center.y)
    val triangleCenter = Offset(center.x + distance * 0.5f, center.y)
    val markerColor = Color(0x70E8EAEE)
    val ledColor = Color(0x60E8EAEE)

    AcquireGuideGeometry.slotFractions.forEach { fraction ->
        drawCircle(
            color = ledColor,
            radius = ledRadius,
            center = Offset(start.x + distance * fraction, center.y),
            style = Stroke(width = strokeWidth),
        )
    }

    val half = triangleSize * 0.32f
    val triangle = Path().apply {
        moveTo(triangleCenter.x, triangleCenter.y - half)
        lineTo(triangleCenter.x + half * 0.92f, triangleCenter.y + half)
        lineTo(triangleCenter.x - half * 0.92f, triangleCenter.y + half)
        close()
    }
    drawRect(
        color = markerColor,
        topLeft = Offset(start.x - squareSize * 0.5f, start.y - squareSize * 0.5f),
        size = Size(squareSize, squareSize),
        style = Stroke(width = strokeWidth),
    )
    drawPath(path = triangle, color = markerColor, style = Stroke(width = strokeWidth))
}

private object AcquireGuideGeometry {
    private const val ledMm = 3f
    private const val gapMm = 2.5f
    private const val markerMm = 4f
    private const val squareMm = ledMm + 0.45f
    private const val triangleMm = ledMm + 2f
    private const val markerGapMm = 4f
    private const val stepMm = ledMm + gapMm
    private const val markerDistanceMm = markerMm + markerGapMm * 2f + ledMm + stepMm * 4f
    private const val firstLedOffsetMm = markerMm / 2f + markerGapMm + ledMm / 2f

    const val patternDistanceFraction = 0.76f
    const val markerDistanceToSizeRatio = markerDistanceMm / markerMm
    const val ledRadiusToMarkerSizeRatio = (ledMm * 0.5f) / markerMm
    const val squareSizeToMarkerSizeRatio = squareMm / markerMm
    const val triangleSizeToMarkerSizeRatio = triangleMm / markerMm

    val slotFractions: FloatArray = FloatArray(5) { index ->
        (firstLedOffsetMm + index * stepMm) / markerDistanceMm
    }
}

private fun DetectionFrame.toOverlay(canvasSize: Size): OverlayFrame {
    val transform = imageToPreviewTransform(
        cropLeft = cropLeft,
        cropTop = cropTop,
        cropWidth = cropWidth,
        cropHeight = cropHeight,
        rotationDegrees = rotationDegrees,
        canvasSize = canvasSize,
    )
    return OverlayFrame(
        slots = slots.map { slot ->
            OverlaySlot(
                point = transform.map(
                    point = slot.imagePoint,
                ),
                isFirst = slot.isFirst,
                radiusPx = slot.imageRadius * transform.scale,
            )
        },
        markers = markers.map { marker ->
            OverlayMarker(
                point = transform.map(marker.imagePoint),
                alongPoint = transform.map(marker.imageAlongPoint),
                kind = marker.kind,
                sizePx = marker.imageSize * transform.scale,
            )
        },
    )
}

private fun Offset.offsetBy(
    ux: Float,
    uy: Float,
    vx: Float,
    vy: Float,
    along: Float,
    normal: Float,
): Offset {
    return Offset(
        x = x + ux * along + vx * normal,
        y = y + uy * along + vy * normal,
    )
}

private fun imageToPreviewTransform(
    cropLeft: Int,
    cropTop: Int,
    cropWidth: Int,
    cropHeight: Int,
    rotationDegrees: Int,
    canvasSize: Size,
): ImageToPreviewTransform {
    val rotatedSize = when (rotationDegrees) {
        90, 270 -> Size(cropHeight.toFloat(), cropWidth.toFloat())
        else -> Size(cropWidth.toFloat(), cropHeight.toFloat())
    }
    val scale = maxOf(canvasSize.width / rotatedSize.width, canvasSize.height / rotatedSize.height)
    val dx = (canvasSize.width - rotatedSize.width * scale) / 2f
    val dy = (canvasSize.height - rotatedSize.height * scale) / 2f
    return ImageToPreviewTransform(
        cropLeft = cropLeft,
        cropTop = cropTop,
        cropWidth = cropWidth,
        cropHeight = cropHeight,
        rotationDegrees = rotationDegrees,
        scale = scale,
        dx = dx,
        dy = dy,
    )
}

private data class ImageToPreviewTransform(
    val cropLeft: Int,
    val cropTop: Int,
    val cropWidth: Int,
    val cropHeight: Int,
    val rotationDegrees: Int,
    val scale: Float,
    val dx: Float,
    val dy: Float,
) {
    fun map(point: ImagePoint): Offset {
        val xInCrop = point.x - cropLeft
        val yInCrop = point.y - cropTop
        val rotated = when (rotationDegrees) {
        90 -> RotatedPoint(
            x = cropHeight - yInCrop,
            y = xInCrop,
        )
        180 -> RotatedPoint(
            x = cropWidth - xInCrop,
            y = cropHeight - yInCrop,
        )
        270 -> RotatedPoint(
            x = yInCrop,
            y = cropWidth - xInCrop,
        )
        else -> RotatedPoint(
            x = xInCrop,
            y = yInCrop,
        )
    }

        return Offset(
            x = dx + rotated.x * scale,
            y = dy + rotated.y * scale,
        )
    }
}

private data class RotatedPoint(
    val x: Float,
    val y: Float,
)

private fun vibrate(context: Context, durationMs: Long) {
    val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val manager = context.getSystemService(VibratorManager::class.java)
        manager.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
    }
    if (!vibrator.hasVibrator()) return
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        vibrator.vibrate(
            VibrationEffect.createOneShot(
                durationMs,
                VibrationEffect.DEFAULT_AMPLITUDE,
            ),
        )
    } else {
        @Suppress("DEPRECATION")
        vibrator.vibrate(durationMs)
    }
}

private const val VIBRATION_SUCCESS_MS = 55L
private const val VIBRATION_FAIL_MS = 120L
