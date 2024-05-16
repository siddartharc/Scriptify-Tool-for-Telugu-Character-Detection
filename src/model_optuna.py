import os
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Input, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



def optuna_trials(num_classes, X_train, y_train, X_val, y_val, X_test, y_test):
    epochs = 1
    def objective(trial):
        # Define model architecture
        input_shape = (255,255, 3)  # Update the input shape according to your image size
        inputs = Input(shape=input_shape)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        x1 = Conv2D(64, (3, 3), padding='same')(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling2D(pool_size=(5, 5))(x1)
        x1 = Dropout(0.25)(x1)

        # Two Convolution Layers with skip connection
        x2 = Conv2D(128, (5, 5), padding='same')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = Conv2D(128, (5, 5), padding='same')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)

        # Add x1 to x2 after a 1x1 convolution on x1
        x1 = Conv2D(128, (1, 1))(x1)
        x2 = Add()([x1, x2])

        x2 = MaxPooling2D(pool_size=(7, 7))(x2)
        x2 = Dropout(0.25)(x2)

        # Two Convolution Layers with skip connection
        x3 = Conv2D(256, (3, 3), padding='same')(x2)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Conv2D(256, (7, 7), padding='same')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)

        # Add x2 to x3 after a 1x1 convolution on x2
        x2 = Conv2D(256, (1, 1))(x2)
        x3 = Add()([x2, x3])

        x3 = MaxPooling2D(pool_size=(7, 7))(x3)
        x3 = Dropout(0.25)(x3)

        # Adaptive pooling instead of flatten
        x = GlobalAveragePooling2D()(x3)

        # Fully Connected Layer with ReLU
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.75)(x)

        # Output Layer with Softmax
        outputs = Dense(num_classes, activation='softmax')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        # Define hyperparameters to be tuned
        lr = trial.suggest_loguniform("lr", 1e-5, 1)
        optimizer_dict = {
            'Adagrad': Adagrad(learning_rate=lr),
            'Adadelta': Adadelta(learning_rate=lr),
            'Adam': Adam(learning_rate=lr),
            'RMSprop': RMSprop(learning_rate=lr),
            'SGD': SGD(learning_rate=lr, nesterov=True)
        }
        optimizer_name = trial.suggest_categorical('optimizer', list(optimizer_dict.keys()))
        optimizer = optimizer_dict[optimizer_name]

        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)
        # Define other hyperparameters as needed

        # Compile the model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

        # Train the model
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(X_val, y_val), shuffle=True,
                            callbacks=[early_stopping, checkpoint])

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        
        # Return test accuracy as the objective value for optimization
        return test_accuracy

    # Define study object and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    # Get the best trial
    best_trial = study.best_trial

    # Load the best model from the saved checkpoint
    best_model = load_model('best_model.keras')

    return best_model, best_trial
