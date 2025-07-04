
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
model = load_model(r'C:\Users\HFX1KOR\Desktop\Cervical_LBC_Project\cervical_lbc_model.h5')

# Load and preprocess the image
img_path = r"C:\Users\HFX1KOR\Desktop\Cervical_LBC_Project\LBC_data\Negative for Intraepithelial malignancy\NL_16_ (4).jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Match training preprocessing
img_array_input = np.expand_dims(img_array, axis=0)

# Define class labels (match Grad-CAM's order)
class_labels = ['SCC', 'HSIL', 'LSIL', 'NC']

# Verify raw prediction
predictions = model.predict(img_array_input)
pred_index = np.argmax(predictions[0])
print(f"Raw prediction probabilities: {predictions[0]}")
print(f"Predicted class: {class_labels[pred_index]}")

# Define prediction function for LIME
def predict_fn(images):
    images = np.clip(images, 0, 1)  # Ensure pixel values are in [0, 1]
    return model.predict(images)

# Initialize LIME explainer
explainer = lime_image.LimeImageExplainer()

# Generate explanation
try:
    explanation = explainer.explain_instance(
        img_array,
        predict_fn,
        top_labels=4,
        num_samples=1000,
        segmentation_fn=None  # Uses default SLIC segmentation
    )
except Exception as e:
    print(f"Error generating LIME explanation: {e}")
    exit()

# Visualize the explanation for the top predicted class
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=True
)

# Plot the result

plt.figure(figsize=(8, 8))
plt.imshow(mark_boundaries(temp, mask))
plt.title(f'LIME Explanation for Class: {class_labels[explanation.top_labels[0]]}')
plt.axis('off')
plt.show()