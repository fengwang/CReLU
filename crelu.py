from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import get_custom_objects
from tensorflow.nn import crelu

class CReLU(Layer):
    """ CReLU Layer

    # Argument:
    axis: The axis that the output values are concatenated along. Default is -1.

    # Reference:
    #     Shang, Wenling, Kihyuk Sohn, Diogo Almeida, and Honglak Lee.
    #     “Understanding and Improving Convolutional Neural Networks via
    #     Concatenated Rectified Linear Units.” ArXiv:1603.05201 [Cs],
    #     July 19, 2016. http://arxiv.org/abs/1603.05201.

    """

    def __init__(self, axis=-1, **kwargs):
        super(CReLU, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        super(CReLU, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return crelu( inputs, self.axis )

    def get_config(self):
        config = { 'axis': self.axis, }
        base_config = super(CReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[self.axis] *= 2
        return tuple(output_shape)

get_custom_objects().update({'CReLU': CReLU})

if __name__ == '__main__':
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

