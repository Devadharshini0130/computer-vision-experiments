import cv2
import matplotlib.pyplot as plt

# Read the input image
image = cv2.imread("input.jpg")

if image is None:
    print("Error: input.jpg not found.")
    exit()

# Convert BGR to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mean Filter
mean_filter = cv2.blur(image, (5, 5))

# Gaussian Filter
gaussian_filter = cv2.GaussianBlur(image, (5, 5), 0)

# Median Filter
median_filter = cv2.medianBlur(image, 5)

# Bilateral Filter
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

# Convert filtered images to RGB
mean_rgb = cv2.cvtColor(mean_filter, cv2.COLOR_BGR2RGB)
gaussian_rgb = cv2.cvtColor(gaussian_filter, cv2.COLOR_BGR2RGB)
median_rgb = cv2.cvtColor(median_filter, cv2.COLOR_BGR2RGB)
bilateral_rgb = cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB)

# Display all results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(mean_rgb)
plt.title("Mean Filter")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gaussian_rgb)
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(median_rgb)
plt.title("Median Filter")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(bilateral_rgb)
plt.title("Bilateral Filter")
plt.axis("off")

plt.tight_layout()
plt.savefig("output.png", dpi=150, bbox_inches="tight")
plt.show()
