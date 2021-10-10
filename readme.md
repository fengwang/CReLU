# CReLU

CReLU layer with TensorFlow2.

----

### Example usage

A simple model applying crelu activation to an input layer, with save/load functionality enabled.

```python
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model, load_model
    ip = Input(shape=(3, 3, 6))
    x = CReLU()(ip)
    model = Model(ip, x)
    model.summary()
    model.save( 'model.h5' )

    print( '*'*80 )
    nm = load_model( 'model.h5' )
    print( 'new model loaded successfully' )
```

produces output of

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3, 3, 6)]         0
_________________________________________________________________
c_re_lu (CReLU)              (None, 3, 3, 12)          0
=================================================================
Total params: 0
Trainable params: 0
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
********************************************************************************
WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
new model loaded successfully
```

### Note

Before loading a model designed with `CReLU`, make sure `from crelu import CReLU` has been executed.

### Reference

Shi, Wenzhe, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. “Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.” ArXiv:1609.05158 [Cs, Stat], September 16, 2016. http://arxiv.org/abs/1609.05158.

### License

AGPL-3

