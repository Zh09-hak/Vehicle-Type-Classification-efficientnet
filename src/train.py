from preprocess import load_data
from model import build_model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os

labels = {name: i for i, name in enumerate(sorted(os.listdir('data')))}

X, y = load_data('data', labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train, num_classes=len(labels))
y_test = to_categorical(y_test, num_classes=len(labels))

model = build_model(len(labels))

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

model.save("model.keras")
