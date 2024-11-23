# **Project Report: FashionGAN - Generative Adversarial Network for Fashion Image Generation**

---

## **1. Introduction**

Generative Adversarial Networks (GANs) are a revolutionary framework in machine learning, capable of creating data that mimics real-world datasets. This project, **FashionGAN**, applied GAN principles to generate synthetic grayscale images of fashion items using the **Fashion MNIST dataset**. The project involved designing and training two interconnected models—a **generator** to create synthetic images and a **discriminator** to distinguish real from fake images.

This project showcases the ability of GANs to generate realistic images, demonstrating their potential applications in automated design, synthetic dataset generation, and creative modeling.

---

## **2. Objectives**

1. Build a GAN capable of generating realistic fashion images.
2. Prepare and preprocess the **Fashion MNIST dataset** for GAN training.
3. Design and implement the **generator** and **discriminator** models.
4. Train the GAN using adversarial techniques, optimizing both models.
5. Evaluate the generator’s performance through visualizations and loss analysis.

---

## **3. Methodology**

---

### **3.1 Environment Setup**

The project utilized the following frameworks and tools:
- **TensorFlow**: For implementing and training the GAN.
- **TensorFlow Datasets**: For seamless dataset handling.
- **Matplotlib**: For visualizing data and results.

To optimize training, GPU memory was configured for smooth usage:
```python
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

### **3.2 Data Preparation**

1. **Dataset Overview**:
   The **Fashion MNIST dataset**, comprising 60,000 grayscale images of fashion items (e.g., T-shirts, shoes, bags), was used. Each image is of size `28x28`.

2. **Preprocessing**:
   - Pixel values were normalized to the range [0, 1] to improve model convergence.
   - The dataset was shuffled, cached, and batched for efficient data loading during training.

   **Code Implementation**:
   ```python
   def scale_images(data): 
       image = data['image']
       return image / 255
   ```

3. **Visualization**:
   - Initially, the dataset was visualized to confirm its integrity and preprocessing.

   **Visualization**: Raw Dataset Samples  

   ![Alt text](<https://github.com/RhythmAhir/FashionGAN-Using-Tensorflow/blob/main/images/plot%20of%20four%20times%20and%20get%20images.png>)

   **Placement**: This visualization belongs in the **Data Preparation** section, as it depicts the original Fashion MNIST dataset.

   **Code Implementation**:
   ```python
   fig, ax = plt.subplots(ncols=4, figsize=(20,20))
   for idx in range(4):
       sample = dataiterator.next()
       ax[idx].imshow(np.squeeze(sample['image']))
       ax[idx].title.set_text(sample['label'])
   ```

---

### **3.3 GAN Architecture**

The GAN was built with two main components: the **generator** and the **discriminator**.

---

#### **Generator**

- **Purpose**: Transform random noise into meaningful fashion images.
- **Architecture**:
  - **Input**: A 128-dimensional noise vector.
  - **Layers**: Dense → Reshape → Upsampling → Convolutional.
  - **Output**: A grayscale image of size `28x28`.

- **Code Implementation**:
  ```python
  def build_generator(): 
    model = Sequential()
    
    # Takes in random values and reshapes it to 7x7x128
    # Beginnings of a generated image
    model.add(Dense(7*7*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))
    
    # Upsampling block 1 
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Upsampling block 2 
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Convolutional block 1
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Convolutional block 2
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Conv layer to get to one channel
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))
    
    return model
  ```

- **Visualization**:
  After building the generator, its outputs were tested using random noise as input.

  **Visualization**: Generator Outputs After Construction  
  
   ![Alt text](<https://github.com/RhythmAhir/FashionGAN-Using-Tensorflow/blob/main/images/Generator%20plot%20four%20times%20and%20get%20images.png>)
 
  **Placement**: This visualization should appear in the **Generator** section, showcasing its initial performance.

---

#### **Discriminator**

- **Purpose**: Distinguish between real and generated images.
- **Architecture**:
  - **Input**: A grayscale image of size `28x28`.
  - **Layers**: Convolutional → Leaky ReLU → Dropout → Dense.
  - **Output**: A binary classification (0 = fake, 1 = real).

- **Code Implementation**:
  ```python
  def build_discriminator(): 
    model = Sequential()
    
    # First Conv Block
    model.add(Conv2D(32, 5, input_shape = (28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Second Conv Block
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Third Conv Block
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Fourth Conv Block
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Flatten then pass to dense layer
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    return model 
  ```

---

### **3.4 Training Process**

1. **Custom Training Loop**:
   - A custom `Model` subclass was created to manage adversarial training. The generator and discriminator were alternately optimized to ensure balanced learning.

2. **Loss Functions**:
   - Binary Cross-Entropy Loss was used for both models:
     - The generator minimized the discriminator's ability to classify its outputs as fake.
     - The discriminator maximized its ability to distinguish between real and fake images.

3. **Progress Monitoring**:
   - Generated images were saved at each epoch using a callback function:
     ```python
     class ModelMonitor(Callback):
          def __init__(self, num_img=3, latent_dim=128):
              self.num_img = num_img
              self.latent_dim = latent_dim
      
          def on_epoch_end(self, epoch, logs=None):
              random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim,1))
              generated_images = self.model.generator(random_latent_vectors)
              generated_images *= 255
              generated_images.numpy()
              for i in range(self.num_img):
                  img = array_to_img(generated_images[i])
                  img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))
     ```

---

## **4. Results**

---

### **4.1 Evaluation of the Generator**

At the end of training, the generator was evaluated using random noise to generate synthetic images.

**Visualization**: Evaluated Generator Outputs  

   ![Alt text](<https://github.com/RhythmAhir/FashionGAN-Using-Tensorflow/blob/main/images/Test%20Out%20the%20Generator.png>)

**Placement**: This visualization should be included in the **Evaluation** section to demonstrate the final quality of the generator's outputs.

---

### **4.2 Loss Curves**

The loss curves depicted the adversarial dynamics:
- The generator's loss decreased as it learned to fool the discriminator.
- The discriminator's loss fluctuated and stabilized as it adapted to the generator's outputs.

**Code Implementation**:
```python
plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.show()
```

   ![Alt text](<https://github.com/RhythmAhir/FashionGAN-Using-Tensorflow/blob/main/images/Loss.png>)

---

## **5. Challenges**

1. **Training Instability**:
   - Balancing the adversarial dynamics between the generator and discriminator was critical to prevent overfitting or mode collapse.

2. **Computational Demand**:
   - GAN training required significant computational resources and optimization techniques.

---

## **6. Future Work**

1. **Conditional GANs**:
   - Extend the model to generate specific types of images by incorporating labels.
2. **High-Resolution Outputs**:
   - Scale the architecture to produce higher-quality images.
3. **Applications in Fashion**:
   - Explore applications in automated design and style transfer.

---

## **7. Conclusion**

The FashionGAN project successfully implemented a GAN to generate realistic synthetic images of fashion items. Through the effective design of the generator and discriminator and systematic adversarial training, the generator achieved significant improvements. This project demonstrates the potential of GANs in creative fields and provides a foundation for future exploration in synthetic data generation and automated design.
