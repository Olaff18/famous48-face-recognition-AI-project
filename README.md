# Famous48 Face Recognition AI

### **Overview**

This project explores the evolution of an Image Classification pipeline, taking a custom dataset of 48 famous individuals (Famous48) from a baseline accuracy of ~70% all the way to **91.95%**.

The dataset consists of low-resolution 24x24 grayscale facial images (flattened to 576 pixels). The project documents a systematic progression from Classical Machine Learning (`scikit-learn`) through manual feature engineering, data augmentation, and finally to modern Deep Learning (`PyTorch`).

---

## The Engineering Journey & Methodology

### 1. The Classical ML Baseline

I initially tested standard algorithmic approaches using raw, flattened 1D pixel arrays (576 features).

* **Random Forest:** Struggled with raw pixel data, capping out at **70.59%**.
* **Artificial Neural Network (MLP):** A 2-layer feed-forward network achieved a strong baseline of **85.88%**.
* **Support Vector Machine (SVM):** Proved to be the champion of the classical algorithms. Navigating the high-dimensional geometric space allowed the SVM to hit **~86.61%** without any data augmentation.

### 2. Feature Engineering: The HOG Pitfall

I attempted to upgrade the Random Forest and MLP by swapping raw pixels for **HOG (Histograms of Oriented Gradients)** edge-detection features.

* **Result:** Accuracy plummeted (RF: 52.82%, MLP: 74.98%).
* **Conclusion:** Because the images are a tiny 24x24 resolution, sharp edges don't really exist. HOG destroyed the raw skin-tone shading and blurry gradients that the models were relying on. *Raw pixels proved superior.*

### 3. Data Augmentation: Flipping vs. Shifting

Neural Networks are data-hungry, so I tested two augmentation strategies to expand the 5,400 image training set.

* **Horizontal Flipping (10k images):** Helped the MLP slightly (**86.17%**), but actually *hurt* the SVM a little (86.54% vs 86.61%). Flipping destroyed the natural human asymmetry (hair parts, lighting shadows) that the SVM had memorized.
* **Pixel Shifting (27k images):** I shifted every image 1 pixel Up, Down, Left, and Right. This preserved asymmetry but forced the MLP to stop memorizing fixed pixel coordinates.
* **Result:** The MLP surged to **88.51%**, officially dethroning the SVM.

### 4. The Deep Learning Frontier (PyTorch CNN)

To break the 90% ceiling, I abandoned 1D flattened arrays and transitioned to **PyTorch**.
I built a custom Convolutional Neural Network (`FaceCNN`) that mathematically folded the pixels back into a 24x24 2D grid. By using a 3-layer architecture with `Conv2d`, `MaxPool2d`, and `Dropout(0.5)`, the model was able to learn its own spatial feature extractors (eyes, noses, jawlines) natively.
Coupled with the pixel-shifted dataset and automated brain-state checkpointing, the CNN locked in an ultimate test accuracy of **91.95%**.

---

## Final Results Matrix

| Model Architecture           | Data Processing / Augmentation                   | Peak Accuracy     |
| ---------------------------- | ------------------------------------------------ | ----------------- |
| Random Forest                | Raw Pixels (No Augmentation)                     | 70.59%            |
| **SVM (RBF Kernel)**   | **Raw Pixels (No Augmentation)**           | **~86.61%** |
| MLP (Neural Network)         | Horizontal Flipping (10k images)                 | 86.17%            |
| MLP (Neural Network)         | Pixel Shifting (27k images)                      | 88.51%            |
| **PyTorch Custom CNN** | **Pixel Shifting (27k) + 2D Spatial Grid** | **91.95%**  |

---

## Tech Stack

* **Language:** Python
* **Classical ML:** `scikit-learn`
* **Deep Learning:** `PyTorch`
* **Data Processing & Augmentation:** `NumPy`, `SciPy` (`scipy.ndimage.shift`)
* **Visualization:** `Matplotlib`

---

## How to Run

1. Clone the repository.
2. Ensure your environment has the required dependencies: `pip install torch numpy scikit-learn scipy matplotlib`
3. The dataset should be located at `data/combined48.txt`.
4. Run the Jupyter Notebook `RESULTS_COMBINED.ipynb` to step through the classical ML baselines.
5. The final Deep Learning CNN cell will automatically track the highest validation accuracy and save the best model weights.
