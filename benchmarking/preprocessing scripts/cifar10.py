"""
Train CIFAR-10 dataset using CNN, save to .npy.
"""

import os
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_shape, num_classes=10):
    model = models.Sequential(name="cifar_cnn")
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape, name='conv1'))
    model.add(layers.BatchNormalization(axis=3, name='bn_conv1'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3), name='conv2'))
    model.add(layers.BatchNormalization(axis=3, name='bn_conv2'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', name='conv3'))
    model.add(layers.BatchNormalization(axis=3, name='bn_conv3'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), name='conv4'))
    model.add(layers.BatchNormalization(axis=3, name='bn_conv4'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, name='fc1'))
    model.add(layers.BatchNormalization(axis=1, name='bn_fc1'))
    model.add(layers.Activation('relu', name='act_fc1'))  # layer to extract features
    model.add(layers.Dense(num_classes, name='output'))
    model.add(layers.BatchNormalization(axis=1, name='bn_output'))
    model.add(layers.Activation('softmax', name='act_output'))
    return model


def create_feature_extractor(model):
    """Create a feature extractor model that uses layers from the original model up to act_fc1"""
    # Get all layers up to and including 'act_fc1'
    feature_layers = []
    for layer in model.layers:
        feature_layers.append(layer)
        if layer.name == 'act_fc1':
            break
    
    # Create a new sequential model with these layers
    feature_extractor = models.Sequential(name="feature_extractor")
    for layer in feature_layers:
        feature_extractor.add(layer)

    
    return feature_extractor


def extract_features(model, x_train, x_test, y_train, y_test, batch_size, output_dir):
    """Extract features using the trained/saved model."""
    # Create feature extractor using the correct approach
    feature_extractor = create_feature_extractor(model)
    
    features_train = feature_extractor.predict(x_train, batch_size=batch_size, verbose=1)
    features_test = feature_extractor.predict(x_test, batch_size=batch_size, verbose=1)

    features = np.vstack([features_train, features_test])
    labels = np.concatenate([y_train, y_test])

    out_path = os.path.join(output_dir, "representations.npy")
    np.save(os.path.join(output_dir, "cifar10.npy"), features)
    np.save(os.path.join(output_dir, "labels.npy"), labels)


    print(f"Saved representations to: {out_path}")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")


def main(x_train, y_train_int, x_test, y_test_int, epochs, batch_size, lr, output_dir, seed, save_best_only=True):
    set_seeds(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Convert labels to categorical
    y_train_cat = utils.to_categorical(y_train_int, 10)
    y_test_cat = utils.to_categorical(y_test_int, 10)

    input_shape = x_train.shape[1:]
    model = build_model(input_shape, num_classes=10)
    
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    ckpt_path = os.path.join(output_dir, "best_weights.weights.h5")
    checkpoint = callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_loss", verbose=1,
        save_best_only=save_best_only, save_weights_only=True
    )
    csv_logger = callbacks.CSVLogger(os.path.join(output_dir, "training_log.csv"))

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    steps_per_epoch = max(1, x_train.shape[0] // batch_size)
    
    model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(x_test, y_test_cat),
        callbacks=[checkpoint, csv_logger],
        verbose=2
    )

    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    extract_features(model, x_train, x_test, y_train_int, y_test_int, batch_size, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAR-10 CNN and save representations.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default=os.path.expanduser("~/embedor/preprocessed_data/cifar10"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extract_only", action="store_true",
                        help="Skip training and just extract features using saved weights")
    args = parser.parse_args()

    # Load CIFAR-10 once
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train_int = y_train.flatten()
    y_test_int = y_test.flatten()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "best_weights.weights.h5")

    if args.extract_only:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Saved weights not found at {ckpt_path}")
        
        print("Loading pre-trained model for feature extraction...")

        model = build_model(x_train.shape[1:], num_classes=10)
    
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        model.load_weights(ckpt_path)
        
        print("Model loaded successfully. Extracting features...")
        
        extract_features(model, x_train, x_test, y_train_int, y_test_int, args.batch_size, args.output_dir)
        
    else:
        main(x_train, y_train_int, x_test, y_test_int,
             epochs=args.epochs, batch_size=args.batch_size,
             lr=args.lr, output_dir=args.output_dir,
             seed=args.seed)
