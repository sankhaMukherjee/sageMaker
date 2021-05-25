
from tensorflow.keras            import Sequential
from tensorflow.keras.layers     import Conv2D, BatchNormalization
from tensorflow.keras.layers     import Activation, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.losses     import categorical_crossentropy
from tensorflow.keras.optimizers import SGD

def getModel():

    model = Sequential([

        Conv2D(64, (3,3), padding='same', input_shape=(28,28,1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D((2,2), strides=2),

        Conv2D(128, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D((2,2), strides=2),
        
        Flatten(),
        Dense(512),
        Activation('relu'),
        Dropout(0.5),

        Dense(10, activation='softmax')
    ])

    return model

def main():

    epochs     = 10
    lr         = 1e-3
    batch_size = 128
    model_dir  = 'model'
    train_dir = 'data'
    val_dir   = 'data'

    model     = getModel()
    optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss = categorical_crossentropy, optimizer=optimizer)

    print(model.summary())



    return

if __name__ == "__main__":
    main()