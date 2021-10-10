# CReLU

CReLU layer with TensorFlow2.

----

### Example usage

A simple model applying CReLU activation to an input layer, with save/load function:

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
********************************************************************************
new model loaded successfully
```

### Note

Before loading a model designed with `CReLU`, make sure `from crelu import CReLU` has been executed.

### Reference

Shang, Wenling, Kihyuk Sohn, Diogo Almeida, and Honglak Lee. “Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units.” ArXiv:1603.05201 [Cs], July 19, 2016. http://arxiv.org/abs/1603.05201.


### License

AGPL-3

