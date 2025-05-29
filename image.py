import cv2
import os

# Preprocessing function
def preprocess_image(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error loading image: {image_path}")
        return None, None, None, None  # Skip further processing if the image is not loaded

    # Step 1: Contrast Enhancement (using histogram equalization)
    hist_equalized = cv2.equalizeHist(img)

    # Adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_equalized = clahe.apply(hist_equalized)

    # Step 2: Edge Detection
    sobelx = cv2.Sobel(adaptive_equalized, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(adaptive_equalized, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)

    edges_canny = cv2.Canny(adaptive_equalized, 100, 200)

    # Step 3: Noise Reduction
    median_filtered = cv2.medianBlur(adaptive_equalized, 5)
    gaussian_filtered = cv2.GaussianBlur(median_filtered, (5, 5), 0)

    return adaptive_equalized, sobel_combined, edges_canny, gaussian_filtered

# Function to apply preprocessing to a dataset
def preprocess_dataset(dataset_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for image_name in os.listdir(dataset_path):
        if not image_name.endswith('.jpg'):  # Only process JPG images
            print(f"Skipping non-JPG file: {image_name}")
            continue

        image_path = os.path.join(dataset_path, image_name)

        # Check file size
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            print(f"Skipping empty file: {image_name}")
            continue

        # Preprocess the image
        adaptive_equalized, sobel_combined, edges_canny, gaussian_filtered = preprocess_image(image_path)

        if adaptive_equalized is None:
            print(f"Skipping corrupted file: {image_name}")
            continue

        # Save the processed images
        cv2.imwrite(os.path.join(output_path, "adaptive_" + image_name), adaptive_equalized)
        cv2.imwrite(os.path.join(output_path, "sobel_" + image_name), sobel_combined)
        cv2.imwrite(os.path.join(output_path, "canny_" + image_name), edges_canny)
        cv2.imwrite(os.path.join(output_path, "gaussian_" + image_name), gaussian_filtered)

# Paths to your train and validation datasets
train_path = r"E:\Clothing Wrinkles\Images\train"
valid_path = r"E:\Clothing Wrinkles\Images\valid"

# Paths to store preprocessed images
preprocessed_train_path = r"E:\Clothing Wrinkles\Images\train_preprocessed"
preprocessed_valid_path = r"E:\Clothing Wrinkles\Images\valid_preprocessed"

# Apply preprocessing to both train and validation datasets
preprocess_dataset(train_path, preprocessed_train_path)
preprocess_dataset(valid_path, preprocessed_valid_path)

print("Preprocessing completed for train and validation datasets.")
