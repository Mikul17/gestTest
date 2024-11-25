package com.mikul17.gesttest
import android.Manifest
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var model: Module

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Wczytaj model PyTorch
        model = Module.load(assetFilePath("best.torchscript"))

        cameraExecutor = Executors.newSingleThreadExecutor()

        setContent {
            CameraWithYolo(model = model, cameraExecutor = cameraExecutor)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private fun assetFilePath(assetName: String): String {
        val file = this.filesDir.resolve(assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath

        assets.open(assetName).use { inputStream ->
            file.outputStream().use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }
        return file.absolutePath
    }
}


@Composable
fun CameraWithYolo(model: Module, cameraExecutor: ExecutorService) {
    val context = LocalContext.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    var bitmapResult by remember { mutableStateOf<Bitmap?>(null) }

    Column(modifier = Modifier.fillMaxSize()) {
        // Wyświetlanie kamerki
        AndroidView(
            modifier = Modifier
                .weight(1f)
                .fillMaxSize(),
            factory = { AndroidViewContext ->
                val previewView = androidx.camera.view.PreviewView(AndroidViewContext)

                // Konfiguracja CameraX
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    val bitmap = imageProxy.toBitmap()
                    bitmapResult = runYoloDetection(model, bitmap)
                    imageProxy.close()
                }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                cameraProvider.bindToLifecycle(
                    AndroidViewContext as ComponentActivity,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )

                previewView
            }
        )

        // Wyświetlanie wyników
        bitmapResult?.let { detectedBitmap ->
            Text(
                text = "Wykryto obiekty!",
                color = Color.Green,
                modifier = Modifier.fillMaxWidth()
            )
            // Możesz dodać komponent do wizualizacji bitmapy z wykrytymi obiektami
        } ?: Text(
            text = "Czekam na obraz z kamery...",
            color = Color.Red,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

fun ImageProxy.toBitmap(): Bitmap {
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}

fun runYoloDetection(model: Module, bitmap: Bitmap): Bitmap {
    // Przygotowanie tensoru wejściowego
    val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
        bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
        TensorImageUtils.TORCHVISION_NORM_STD_RGB
    )

    // Predykcja
    val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
    val scores = outputTensor.dataAsFloatArray

    // TODO: Obsługa wyników (np. narysowanie na bitmapie wykrytych obiektów)

    return bitmap // Możesz zwrócić bitmapę z naniesionymi wynikami
}


