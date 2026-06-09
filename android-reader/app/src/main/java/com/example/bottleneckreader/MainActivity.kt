package com.example.bottleneckreader

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(color = Color.Black) {
                    ReaderApp()
                }
            }
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
                Lifecycle.Event.ON_STOP -> stoppedOnce = true
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
    val notices by viewModel.notices.collectAsStateWithLifecycle()
    val problem by viewModel.problem.collectAsStateWithLifecycle()
    val restartToken by viewModel.restartToken.collectAsStateWithLifecycle()

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (cameraPermission) {
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

        DecodedBadge(
            value = frame?.bits ?: "null",
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 28.dp),
        )

        NoticeQueue(
            notices = notices,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 20.dp, end = 16.dp),
        )

        if (!cameraPermission) {
            Text(
                text = "Camera permission required",
                color = Color.White,
                modifier = Modifier.align(Alignment.Center),
            )
        }
    }

    val activeProblem = problem
    if (activeProblem != null) {
        AlertDialog(
            onDismissRequest = viewModel::dismissProblem,
            title = { Text(activeProblem.title) },
            text = { Text(activeProblem.message) },
            confirmButton = {
                Button(
                    onClick = {
                        if (!cameraPermission) {
                            permissionLauncher.launch(Manifest.permission.CAMERA)
                        } else {
                            viewModel.retryCamera()
                        }
                    },
                ) {
                    Text(if (cameraPermission) "Retry camera" else "Grant permission")
                }
            },
            dismissButton = {
                TextButton(onClick = viewModel::dismissProblem) {
                    Text("Dismiss")
                }
            },
        )
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
        val roiHeight = size.height * ReaderRoi.HEIGHT_FRACTION
        val roiTopLeft = Offset(
            x = (size.width - roiWidth) * 0.5f,
            y = (size.height - roiHeight) * 0.5f,
        )
        drawRect(
            color = Color(0xB0E8EAEE),
            topLeft = roiTopLeft,
            size = Size(roiWidth, roiHeight),
            style = Stroke(width = 1.5.dp.toPx()),
        )

        val current = frame ?: return@Canvas
        val overlay = current.toOverlay(size)
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
private fun DecodedBadge(
    value: String,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .background(Color.White, RoundedCornerShape(3.dp))
            .padding(horizontal = 22.dp, vertical = 9.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = value,
            color = Color.Black,
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 0.sp,
        )
    }
}

@Composable
private fun NoticeQueue(
    notices: List<ReaderNotice>,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(8.dp),
        horizontalAlignment = Alignment.End,
    ) {
        notices.forEach { notice ->
            AnimatedVisibility(
                visible = !notice.exiting,
                enter = slideInHorizontally(
                    initialOffsetX = { it + 80 },
                    animationSpec = tween(220),
                ) + fadeIn(tween(220)),
                exit = slideOutHorizontally(
                    targetOffsetX = { it + 80 },
                    animationSpec = tween(340),
                ) + fadeOut(tween(280)) + shrinkVertically(tween(420)),
            ) {
                Box(
                    modifier = Modifier
                        .widthIn(max = 300.dp)
                        .background(Color(0xDD090A0C), RoundedCornerShape(7.dp))
                        .padding(PaddingValues(horizontal = 12.dp, vertical = 9.dp)),
                ) {
                    Text(
                        text = notice.message,
                        color = Color(0xFFE8EAEE),
                        fontSize = 13.sp,
                        lineHeight = 17.sp,
                    )
                }
            }
        }
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
        bits = bits,
        slots = slots.map { slot ->
            OverlaySlot(
                point = transform.map(
                    point = slot.imagePoint,
                ),
                isFirst = slot.isFirst,
                radiusPx = slot.imageRadius * transform.scale,
            )
        },
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
            width = cropHeight.toFloat(),
            height = cropWidth.toFloat(),
        )
        180 -> RotatedPoint(
            x = cropWidth - xInCrop,
            y = cropHeight - yInCrop,
            width = cropWidth.toFloat(),
            height = cropHeight.toFloat(),
        )
        270 -> RotatedPoint(
            x = yInCrop,
            y = cropWidth - xInCrop,
            width = cropHeight.toFloat(),
            height = cropWidth.toFloat(),
        )
        else -> RotatedPoint(
            x = xInCrop,
            y = yInCrop,
            width = cropWidth.toFloat(),
            height = cropHeight.toFloat(),
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
    val width: Float,
    val height: Float,
)
