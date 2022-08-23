## BentoML Tensorflow Fashion MNIST DEMO

### APIs

1. **/predict_ndarray**
   - Input: `numpy.ndarray` 
      - Shape: `({batch_size}, 28, 28)`
      - Dtype: `uint8`
      <!-- - Input normalization needed (`input / 255.0`) -->
    - Output: `numpy.ndarray`
      - Shape: `({batch_size}, 10)` (possibilities for each label)
      - Dtype: `float32`

2. **/predict_image**
   - Input: image file
      - Size: 28x28 pixels
      - Channels: 1 (grayscale)
   - Output: `numpy.ndarray`
      - Shape: `({batch_size}, 10)` (possibilities for each label)
      - Dtype: `float32`