import os
from preprocessing import load_data_pp
from data_loader import data_loader, data_model
from model_optuna import optuna_trials
# from model import model
from tensorflow.keras.optimizers import Adagrad, Adadelta, Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint

# def train_optuna(train_dir=r'C:\Users\sidda\Desktop\Tool\data\Preprocessed_data'):
def train_optuna(train_dir='data/Preprocessed_data'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir) 
    train_dir = os.path.join(base_dir, train_dir)

    images, labels = data_loader(train_dir, 100)
    num_classes, X_train, y_train, X_val, y_val, X_test, y_test = data_model(images, labels)
    model, best_trial = optuna_trials(num_classes, X_train, y_train, X_val, y_val, X_test, y_test)
    return model, best_trial

# def train(epochs, num_images, train_dir=r'C:\Users\sidda\Desktop\Tool\data\Preprocessed_data'):
def train(epochs, num_images, train_dir='data/Preprocessed_data'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir) 
    train_dir = os.path.join(base_dir, train_dir)

    images, labels = data_loader(train_dir, num_images)
    num_classes, X_train, y_train, X_val, y_val, X_test, y_test = data_model(images, labels)
    model, best_trial = train_optuna()

    batch_size = best_trial.params['batch_size']
    optimizer_name = best_trial.params['optimizer']
    lr = best_trial.params['lr']
    dropout_rate = best_trial.params['dropout_rate']

    optimizer_dict = {
        'Adagrad': Adagrad(learning_rate=lr),
        'Adadelta': Adadelta(learning_rate=lr),
        'Adam': Adam(learning_rate=lr),
        'RMSprop': RMSprop(learning_rate=lr),
        'SGD': SGD(learning_rate=lr, nesterov=True)
    }

    optimizer = optimizer_dict[optimizer_name]

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with the extracted hyperparameters
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    # Save the trained model
    # save_dir = r'C:\Users\sidda\Desktop\Tool\models'
    save_dir = os.path.join(base_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'trained_model.keras')
    model.save(model_path)

    return model