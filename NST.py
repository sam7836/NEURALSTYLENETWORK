import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load and preprocess image
def load_and_process_image(image_path, img_size=(512, 512)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = np.expand_dims(image, axis=0)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    return image

# Convert tensor to displayable image
def deprocess_image(image_tensor):
    image = np.squeeze(image_tensor.numpy(), axis=0)
    image = np.clip(image, 0, 255).astype('uint8')
    return image

# Build VGG19 model for feature extraction
def build_vgg_model(layer_names):
    vgg = VGG19(weights="imagenet", include_top=False)
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = Model(inputs=vgg.input, outputs=outputs)
    return model

# Compute Gram matrix (Style Representation)
def gram_matrix(tensor):
    shape = tf.shape(tensor)
    height, width, channels = shape[1], shape[2], shape[3]
    matrix = tf.reshape(tensor, [height * width, channels])  # Flatten spatial dimensions
    gram = tf.matmul(tf.transpose(matrix), matrix)  # Compute Gram matrix
    return gram / tf.cast(height * width * channels, tf.float32)  # Normalize

# Compute total loss (Content + Style)
def compute_loss(model, generated, content, style, style_layers, content_layer):
    generated_features = model(generated)
    content_features = model(content)
    style_features = model(style)

    # Ensure tensors are correctly extracted
    content_loss = tf.reduce_mean(tf.square(generated_features[-1] - content_features[-1]))

    style_loss = 0
    for gen, sty in zip(generated_features[:-1], style_features[:-1]):
        gram_gen = gram_matrix(gen)
        gram_sty = gram_matrix(sty)
        style_loss += tf.reduce_mean(tf.square(gram_gen - gram_sty))
    
    style_loss /= len(style_layers)

    # **Adjusted Weights for Style Transfer**
    total_loss = content_loss * 1e2 + style_loss * 1e4  # Increase style weight
    return total_loss

# Layers to extract style and content
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layer = ['block4_conv2']

# Load images
content_image_path = "content.jpg"   # Ensure correct path
style_image_path = "style.jpg"       # Ensure correct path

content_image = load_and_process_image(content_image_path)
style_image = load_and_process_image(style_image_path)

# Initialize generated image from content image
generated_image = tf.Variable(content_image, trainable=True, dtype=tf.float32)

# Build VGG model for feature extraction
model = build_vgg_model(style_layers + content_layer)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=5.0)

# Training loop
epochs = 2000  # Increase training epochs for better results
for i in range(epochs):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, generated_image, content_image, style_image, style_layers, content_layer)
    
    grads = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(grads, generated_image)])

    # Ensure pixel values remain valid
    generated_image.assign(tf.clip_by_value(generated_image, 0, 255))

    if i % 200 == 0:
        print(f"Epoch {i}, Loss: {loss.numpy()}")
        plt.imshow(deprocess_image(generated_image))
        plt.show()

# Save final stylized image
final_image = deprocess_image(generated_image)
cv2.imwrite("stylized_output.jpg", cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

print("Style transfer completed! Output saved as stylized_output.jpg")
