package com.example.bottleneckreader

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong

class ReaderViewModel : ViewModel() {
    private val ids = AtomicLong(0)

    private val _frame = MutableStateFlow<DetectionFrame?>(null)
    val frame: StateFlow<DetectionFrame?> = _frame.asStateFlow()

    private val _notices = MutableStateFlow<List<ReaderNotice>>(emptyList())
    val notices: StateFlow<List<ReaderNotice>> = _notices.asStateFlow()

    private val _problem = MutableStateFlow<CameraProblem?>(null)
    val problem: StateFlow<CameraProblem?> = _problem.asStateFlow()

    private val _restartToken = MutableStateFlow(0)
    val restartToken: StateFlow<Int> = _restartToken.asStateFlow()

    fun onReaderEvent(event: ReaderEvent) {
        when (event) {
            is ReaderEvent.Detection -> _frame.value = event.frame
            is ReaderEvent.Notice -> enqueueNotice(event.message)
            is ReaderEvent.CameraIssue -> {
                _problem.value = event.problem
                enqueueNotice(event.problem.title)
            }
            ReaderEvent.SlowDecoderTerminated -> {
                val problem = CameraProblem(
                    title = "Decoder too slow",
                    message = "Frame processing could not keep up with 30 fps. The camera stream was terminated.",
                )
                _problem.value = problem
                enqueueNotice("Stream terminated: decoder too slow")
            }
        }
    }

    fun notifyResumed(reason: String) {
        enqueueNotice("Resumed: $reason")
    }

    fun resumeCamera(reason: String) {
        _restartToken.update { it + 1 }
        enqueueNotice("Resumed: $reason")
    }

    fun retryCamera() {
        _problem.value = null
        _restartToken.update { it + 1 }
        enqueueNotice("Camera restart requested")
    }

    fun dismissProblem() {
        _problem.value = null
    }

    private fun enqueueNotice(message: String) {
        val id = ids.incrementAndGet()
        _notices.update { it + ReaderNotice(id = id, message = message) }
        viewModelScope.launch {
            delay(NOTICE_VISIBLE_MS)
            _notices.update { notices ->
                notices.map { if (it.id == id) it.copy(exiting = true) else it }
            }
            delay(NOTICE_EXIT_MS)
            _notices.update { notices -> notices.filterNot { it.id == id } }
        }
    }

    private companion object {
        const val NOTICE_VISIBLE_MS = 3_200L
        const val NOTICE_EXIT_MS = 420L
    }
}
