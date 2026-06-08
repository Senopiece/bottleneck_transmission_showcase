package com.example.bottleneckreader

import android.content.Context
import android.hardware.camera2.CaptureRequest
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
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
    private val decoder = LedFrameDecoder()
    private val closed = AtomicBoolean(false)
    private var provider: ProcessCameraProvider? = null
    private var camera: Camera? = null
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
        camera = null
        decoder.resetTracking()
    }

    fun release() {
        stop()
        analyzerExecutor.shutdown()
    }

    @Suppress("DEPRECATION")
    private fun bind(cameraProvider: ProcessCameraProvider) {
        if (closed.get()) return

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
            stopCamera = {
                ContextCompat.getMainExecutor(context).execute {
                    stop()
                }
            },
        )
        analyzer = nextAnalyzer
        analysis.setAnalyzer(analyzerExecutor, nextAnalyzer)

        cameraProvider.unbindAll()
        camera = cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            analysis,
        )
        camera?.cameraInfo?.cameraState?.observe(lifecycleOwner) { state ->
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
        private val stopCamera: () -> Unit,
    ) : ImageAnalysis.Analyzer {
        private var stopped = false
        private var slowFrames = 0

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
                                rotationDegrees = image.imageInfo.rotationDegrees,
                                bits = null,
                                slots = emptyList(),
                            ),
                        ),
                    )
                } else {
                    eventSink(ReaderEvent.Detection(frame))
                }
            } catch (error: Throwable) {
                eventSink(
                    ReaderEvent.CameraIssue(
                        CameraProblem(
                            title = "Decoder failed",
                            message = error.message ?: error::class.java.simpleName,
                        ),
                    ),
                )
            } finally {
                image.close()
            }

            val elapsedMs = (System.nanoTime() - started) / 1_000_000
            if (elapsedMs > FRAME_BUDGET_MS) slowFrames++ else slowFrames = 0
            if (slowFrames >= MAX_SLOW_FRAMES) {
                stopped = true
                eventSink(ReaderEvent.SlowDecoderTerminated)
                stopCamera()
            }
        }

        private companion object {
            const val FRAME_BUDGET_MS = 33L
            const val MAX_SLOW_FRAMES = 8
        }
    }
}
