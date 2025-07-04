# File: cervical_shap_explain.py
import numpy as np
import matplotlib.pyplot as plt
import shap
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
img_path = r'C:\Users\HFX1KOR\Desktop\Cervical_LBC_Project\LBC_data\Negative for Intraepithelial malignancy\NL_16_ (4).jpg'
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

# Define background dataset
background = img_array_input

# Define prediction function for SHAP
def predict_fn(images):
    images = np.clip(images, 0, 1)  # Ensure pixel values are in [0, 1]
    return model.predict(images)

# Initialize SHAP masker and explainer
masker = shap.maskers.Image("inpaint_telea", img_array_input.shape[1:])
explainer = shap.PartitionExplainer(predict_fn, masker)

# Compute SHAP values for the top predicted class
try:
    shap_values = explainer(img_array_input, outputs=[pred_index])
    print(f"SHAP values computed for class: {class_labels[pred_index]}")
except Exception as e:
    print(f"Error generating SHAP values: {e}")
    exit()

# Plot SHAP explanation
shap.image_plot(shap_values, img_array_input, labels=[class_labels[pred_index]])

# Alternative: Manual visualization
shap_values_array = shap_values[0].values[..., 0]  # For the top class
plt.figure(figsize=(8, 8))
plt.imshow(img_array)
plt.imshow(shap_values_array, cmap='jet', alpha=0.4)
plt.title(f'SHAP Explanation for Class: {class_labels[pred_index]}')
plt.axis('off')
plt.show()