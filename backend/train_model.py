import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Step 1: Preprocessing Functions
def remove_invalid_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                with Image.open(os.path.join(root, file)) as img:
                    img.verify()
            except (IOError, SyntaxError):
                print(f"Removing corrupted image: {file}")
                os.remove(os.path.join(root, file))

def convert_to_rgb(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img.save(img_path)

# Step 2: Preprocess Dataset
dataset_dir = 'dataset'
remove_invalid_images(dataset_dir)
convert_to_rgb(dataset_dir)

# Step 3: ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    subset='training'
)

validation_data = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Step 4: Build a Better CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Regularization to prevent overfitting
    Dense(3, activation='softmax')  # 3 classes (mild, moderate, severe)
])

# Step 5: Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Slightly smaller learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 6: Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/best_acne_model.keras', save_best_only=True, monitor='val_loss')

# Step 7: Train the Model
model.fit(
    train_data,
    epochs=30,  # More epochs but EarlyStopping will stop when needed
    validation_data=validation_data,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save('models/final_acne_model.keras')
