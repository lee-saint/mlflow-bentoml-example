import typing as t

import numpy as np
from PIL.Image import Image as PILImage

import bentoml
from bentoml.io import Image
from bentoml.io import NumpyNdarray


mnist_runner = bentoml.keras.get("tf_fashion_mnist_model").to_runner()

svc = bentoml.Service(
    name="tf-fashion-mnist-demo",
    runners=[
        mnist_runner,
    ],
)


@svc.api(
    input=NumpyNdarray(dtype="uint8", enforce_dtype=True),
    output=NumpyNdarray(dtype="float32"),
)
async def predict_ndarray(
    inp: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    inp = inp.astype('float32')
    inp = inp / 255.0
    inp = np.expand_dims(inp, 3)

    output_tensor = await mnist_runner.predict.async_run(inp)
    return output_tensor


@svc.api(input=Image(), output=NumpyNdarray(dtype="float32"))
async def predict_image(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    arr = np.array(f) / 255.0
    assert arr.shape == (28, 28)

    # extra channel dimension
    arr = np.expand_dims(arr, (0, 3)).astype("float32")
    output_tensor = await mnist_runner.predict.async_run(arr)
    return output_tensor