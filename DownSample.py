from keras.models import *
from keras.layers import *
from Final_Model.dhaspp import DHASPP
from Final_Model.DownSampleBlock import _DownSample
# from Final_Model.res_cbam import attention_module
from Final_Model.ResSE import ResSEModule
from Final_Model.conv_block import conv2d_block_1, conv2d_block


def _DownSample_Net(input_height=48, imput_weight=48):

    img_input = Input(shape=(input_height, imput_weight, 1))

    x = conv2d_block(img_input, 64)
    # x = IR_Block(x, 64)

    e1 = x
    e11 = _DownSample(e1, _filters=64,_strides=(2, 2))
    e12 = _DownSample(e1, _filters=64,_strides=(4, 4))

    x = _DownSample(x, _filters=128,_strides=(2, 2))
    x = conv2d_block(x, 128)
    # x = IR_Block(x, 128)

    e2 = x
    e21 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(e2)
    e22 = _DownSample(e2, _filters=128,_strides=(2, 2))

    x = _DownSample(x, _filters=128,_strides=(2, 2))
    x = conv2d_block(x, 256)
    # x = IR_Block(x, 256)

    e3 = x
    e31 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(e3)
    e32 = Conv2DTranspose(256, (4, 4), strides=(4, 4), padding='same')(e3)

# =======================================================================================================

    bridge = DHASPP(x, 256)

# =======================================================================================================
#     d7 = (concatenate([x, e22, e12], axis=-1))
#     d7 = (concatenate([e3, e22, e12], axis=-1))
    d7 = (concatenate([bridge, e3, e22, e12], axis=-1))
    d7 = conv2d_block_1(d7, 256)
    # d7 = ResSEModule(d7, 4, 256)
    # d7 = attention_module(d7)
    d7 = conv2d_block(d7, 256)

    d8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d7)
    d8 = (concatenate([d8, e2, e11, e31], axis=-1))
    d8 = conv2d_block_1(d8, 128)
    # d8 = ResSEModule(d8, 4, 128)
    # d8 = attention_module(d8)
    d8 = conv2d_block(d8, 128)
    d9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d8)

    o = (concatenate([d9, e1, e21, e32], axis=-1))
    o = conv2d_block_1(o, 64)
    # o = ResSEModule(o, 4, 64)
    # o = attention_module(o)
    o = conv2d_block(o, 64)

    o = Conv2D(2, (1, 1), padding='same')(o)
    # =========================================
    o = Reshape((48*48, 2))(o)
    o = Activation('sigmoid')(o)
    # o = Activation('softmax')(o)
    # =========================================

    model = Model(img_input, o)

    # =========================================
    from tensorflow.keras.optimizers import Adam
    # model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


    # from tensorflow.keras.optimizers import SGD
    # sgd = SGD(lr=0.003, decay=0.0005, momentum=0.9, nesterov=False)
    # model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])
    # =========================================

    return model


if __name__ == '__main__':
    model = _DownSample_Net()
    model.summary()
