# File: cervical_saliency_explain.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
try:
    model = load_model(r'C:\Users\HFX1KOR\Desktop\Cervical_LBC_Project\cervical_lbc_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load and preprocess the image
img_path = r'C:\Users\HFX1KOR\Desktop\Cervical_LBC_Project\LBC_data\Squamous cell carcinoma\scc_1 (2).jpg'
try:
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Match training preprocessing
    img_array_input = np.expand_dims(img_array, axis=0)
except Exception as e:
    print(f"Error loading or preprocessing image: {e}")
    exit()

# Define class labels (match Grad-CAM's order)
class_labels = ['SCC', 'HSIL', 'LSIL', 'NC']

# Verify raw prediction
predictions = model.predict(img_array_input)
pred_index = np.argmax(predictions[0])
print(f"Raw prediction probabilities: {predictions[0]}")
print(f"Predicted class: {class_labels[pred_index]}")

# Compute gradients for saliency map
img_tensor = tf.convert_to_tensor(img_array_input, dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    predictions = model(img_tensor)
    pred_index = tf.argmax(predictions[0])
    class_score = predictions[:, pred_index]

# Get gradients
grads = tape.gradient(class_score, img_tensor)
grads = tf.abs(grads)  # Take absolute value
saliency_map = tf.reduce_max(grads, axis=-1).numpy()[0]  # Max across color channels
saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-10)  # Normalize

# Visualize the saliency map
plt.figure(figsize=(8, 8))
plt.imshow(img_array)
plt.imshow(saliency_map, cmap='jet', alpha=0.4)
plt.title(f'Saliency Map for Class: {class_labels[pred_index.numpy()]}')
plt.axis('off')
plt.show()