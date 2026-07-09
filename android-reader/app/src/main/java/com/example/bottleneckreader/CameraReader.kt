package com.example.bottleneckreader

import android.content.Context
import android.hardware.camera2.CaptureRequest
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class CameraReader(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val eventSink: (ReaderEvent) -> Unit,
) {
    private val analyzerExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val decoder = LedFrameDecoder(context)
    private val closed = AtomicBoolean(false)
    private var provider: ProcessCameraProvider? = null
    private var analyzer: FrameAnalyzer? = null

    fun start() {
        closed.set(false)
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
        previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE

        val future = ProcessCameraProvider.getInstance(context)
        future.addListener(
            {
                runCatching {
                    provider = future.get()
                    bind(provider ?: return@addListener)
                }.onFailure { error ->
                    eventSink(
                        ReaderEvent.CameraIssue(
                            CameraProblem(
                                title = "Camera start failed",
                                message = error.message ?: error::class.java.simpleName,
                            ),
                        ),
                    )
                }
            },
            ContextCompat.getMainExecutor(context),
        )
    }

    fun stop() {
        closed.set(true)
        analyzer?.stop()
        provider?.unbindAll()
        decoder.resetTracking()
    }

    fun release() {
        stop()
        analyzerExecutor.shutdown()
    }

    @Suppress("DEPRECATION")
    private fun bind(cameraProvider: ProcessCameraProvider) {
        if (closed.get()) return
        if (previewView.viewPort == null && (previewView.width == 0 || previewView.height == 0)) {
            previewView.post {
                bind(cameraProvider)
            }
            return
        }

        val fpsRange = Range(30, 30)
        val previewBuilder = Preview.Builder()
        Camera2Interop.Extender(previewBuilder)
            .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange)

        val analysisBuilder = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetResolution(Size(640, 480))
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
        Camera2Interop.Extender(analysisBuilder)
            .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange)

        val preview = previewBuilder.build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val analysis = analysisBuilder.build()
        val nextAnalyzer = FrameAnalyzer(
            decoder = decoder,
            eventSink = eventSink,
        )
        analyzer = nextAnalyzer
        analysis.setAnalyzer(analyzerExecutor, nextAnalyzer)

        cameraProvider.unbindAll()
        val viewPort = previewView.viewPort
        val boundCamera = if (viewPort != null) {
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                UseCaseGroup.Builder()
                    .addUseCase(preview)
                    .addUseCase(analysis)
                    .setViewPort(viewPort)
                    .build(),
            )
        } else {
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis,
            )
        }
        boundCamera.cameraInfo.cameraState.observe(lifecycleOwner) { state ->
            val error = state.error ?: return@observe
            eventSink(
                ReaderEvent.CameraIssue(
                    CameraProblem(
                        title = "Camera issue",
                        message = "CameraX reported ${error.code}: ${error.cause?.message ?: "unknown cause"}",
                    ),
                ),
            )
        }
    }

    private class FrameAnalyzer(
        private val decoder: LedFrameDecoder,
        private val eventSink: (ReaderEvent) -> Unit,
    ) : ImageAnalysis.Analyzer {
        private var stopped = false

        fun stop() {
            stopped = true
        }

        override fun analyze(image: ImageProxy) {
            if (stopped) {
                image.close()
                return
            }

            val started = System.nanoTime()
            try {
                val frame = decoder.decode(image)
                if (frame == null) {
                    eventSink(
                        ReaderEvent.Detection(
                            DetectionFrame(
                                timestampNs = image.imageInfo.timestamp,
                                imageWidth = image.width,
                                imageHeight = image.height,
                                cropLeft = image.cropRect.left,
                                cropTop = image.cropRect.top,
                                cropWidth = image.cropRect.width(),
                                cropHeight = image.cropRect.height(),
                                rotationDegrees = image.imageInfo.rotationDegrees,
                                slots = emptyList(),
                                isAcquireMode = decoder.isAcquireMode,
                                debugLines = decoder.lastDebugLines,
                            ),
                        ),
                    )
                } else {
                    eventSink(ReaderEvent.Detection(frame))
                }
            } finally {
                image.close()
            }

            val elapsedNs = System.nanoTime() - started
            val elapsedMs = elapsedNs / 1_000_000f
            if (Diagnostics.enabled) {
                eventSink(ReaderEvent.DecoderTiming(elapsedMs))
            }
        }
    }
}
