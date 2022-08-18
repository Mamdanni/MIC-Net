# time : 2022/5/10 9:13
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

    f1 = x
    f11 = _DownSample(f1, _filters=64,_strides=(2, 2))
    f12 = _DownSample(f1, _filters=64,_strides=(4, 4))

    x = _DownSample(x, _filters=128,_strides=(2, 2))
    x = conv2d_block(x, 128)
    # x = IR_Block(x, 128)

    f2 = x
    f21 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(f2)
    f22 = _DownSample(f2, _filters=128,_strides=(2, 2))

    x = _DownSample(x, _filters=128,_strides=(2, 2))
    x = conv2d_block(x, 256)
    # x = IR_Block(x, 256)

    f3 = x
    f31 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(f3)
    f32 = Conv2DTranspose(256, (4, 4), strides=(4, 4), padding='same')(f3)

# =======================================================================================================

    bridge = DHASPP(x, 256)

# =======================================================================================================
#     c7 = (concatenate([x, f22, f12], axis=-1))
#     c7 = (concatenate([f3, f22, f12], axis=-1))
    c7 = (concatenate([bridge, f3, f22, f12], axis=-1))
    c7 = conv2d_block_1(c7, 256)
    # c7 = ResSEModule(c7, 4, 256)
    # c7 = attention_module(c7)
    c7 = conv2d_block(c7, 256)

    c8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='convtran_3')(c7)
    c8 = (concatenate([c8, f2, f11, f31], axis=-1))
    c8 = conv2d_block_1(c8, 128)
    # c8 = ResSEModule(c8, 4, 128)
    # c8 = attention_module(c8)
    c8 = conv2d_block(c8, 128)
    c9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='convtran_4')(c8)

    o = (concatenate([c9, f1, f21, f32], axis=-1))
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
