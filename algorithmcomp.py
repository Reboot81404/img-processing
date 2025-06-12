import cv2
import numpy as np
import os 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from skimage import color,data,restoration,io
import time
from scipy.ndimage import gaussian_filter
import psutil

### cpu-1 için
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

p = psutil.Process(os.getpid())
p.cpu_affinity([0]) 
###



image = cv2.imread('imgastronaut.jpg')
image = cv2.resize(image, (512, 512))

# 
def show_image(title, img):
    plt.figure()
    plt.title(title)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# for the save times
times = {}



# --- 1. Canny Edge Detection ---
start = time.time()
canny = cv2.Canny(image, 100, 200)

end = time.time()
times["Canny"] = end - start
plt.imshow(canny, cmap='gray')
plt.title("canny")
plt.show()


##-- Deblur --
start = time.time()

# Assume `image` is already loaded as a BGR image (e.g., via cv2.imread)
image_float = image.astype(np.float32) / 255.0

# Create PSF (Point Spread Function)
psf = np.ones((5, 5)) / 25

# Initialize channels
blurred_channels = []
deconvolved_channels = []

# RGB için 3 farklı işlem
for i in range(3):
    blurred = convolve2d(image_float[:, :, i], psf, mode='same', boundary='wrap')
    blurred_channels.append(blurred)

    deconvolved, _ = restoration.unsupervised_wiener(blurred, psf)
    deconvolved_channels.append(deconvolved)


blurred_color = np.stack(blurred_channels, axis=2)
deconvolved_color = np.stack(deconvolved_channels, axis=2)

#  uint8 for display
blurred_uint8 = np.clip(blurred_color * 255, 0, 255).astype(np.uint8)
deconvolved_uint8 = np.clip(deconvolved_color * 255, 0, 255).astype(np.uint8)

end = time.time()
times["deblur"] = end - start

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title('Blurred')
plt.imshow(cv2.cvtColor(blurred_uint8, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title('Deblurred')
plt.imshow(cv2.cvtColor(deconvolved_uint8, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()



# --- 3. Gaussian Blur ---
start = time.time()
gauss = gaussian_filter(image, sigma=1)
end = time.time()
times["Gaussian"] = end - start
show_image("Gaussian", gauss)

# --- 4. Gradient (Sobel) ---
#start = time.time()
#sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#gradient = cv2.magnitude(sobelx, sobely)
#end = time.time()
#times["Gradient"] = end - start
#show_image("Gradient", np.uint8(gradient))

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

start = time.time()
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
gradient = cv2.magnitude(sobelx, sobely)
end = time.time()

times["Gradient"] = end - start
show_image("Gradient", np.uint8(gradient))


#################################################
# --- 5. K-means Segmentation ---
start = time.time()
pixel_values  =  image.reshape((-1, 3))
pixel_values  = np.float32(pixel_values )

k=4
kmeans = KMeans(n_clusters=k, n_init=10, algorithm='elkan', random_state=42)
labels = kmeans.fit_predict(pixel_values)
centers = np.uint8(kmeans.cluster_centers_)

segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

end = time.time()
times["K-means"] = end - start
show_image("K-means segmentation",segmented_image)
###################################################
# --- 6. Matrix Multiplication --- 
start = time.time()
mat1 = np.random.rand(256, 256).astype(np.float32)
mat2 = np.random.rand(256, 256).astype(np.float32)
mat1_tf = tf.convert_to_tensor(mat1)
mat2_tf = tf.convert_to_tensor(mat2)
result = tf.matmul(mat1_tf, mat2_tf)
matrix_output = result.numpy()
end = time.time() # Show a portion for clarity
times["matrix mul"] = end - start
matrix_vis = cv2.normalize(matrix_output, None, 0, 255, cv2.NORM_MINMAX)
matrix_vis = matrix_vis.astype(np.uint8)

plt.imshow(matrix_vis)
plt.colorbar()
plt.title("Matrix Multiplication Output")
plt.show()



# --- 7. Bicubic Resize ---
start = time.time()
bicubic = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
end = time.time()
times["bicubic"] = end - start
show_image("Resize Bicubic", bicubic)

# --- 8. Bilinear Resize --- 
start = time.time()
bilinear = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
end = time.time()
times["bilinear"] = end - start
show_image("Resize Bilinear", bilinear)

# --- 9. Rotation 45 degrees --- 
start = time.time()
h, w = image.shape[:2]
M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
rotated = cv2.warpAffine(image, M, (w, h))
end = time.time()
times["Rotation"] = end - start
show_image("Rotation", rotated)

# --- 10. 2D Convolution ---
kernel = np.array([[1, 0, -1],
                    [0, 0,0], 
                    [-1, 0, 1]]) ## setted for the edge detection

start = time.time()
conv2d = cv2.filter2D(image, -1, kernel) ## -1 same depth for the input and output
end = time.time()
times["Conv2D"] = end - start
show_image("2D Convolution", conv2d)

# --- 11. PDE (Heat Equation Simulation) ---
def pde_diffusion(image, iterations=50, dt=0.1):
    image_pde = image.astype(np.float32) / 255.0
    for _ in range(iterations):
        laplacian = cv2.Laplacian(image_pde, cv2.CV_32F)
        image_pde += dt * laplacian
    image_pde = np.clip(image_pde * 255, 0, 255).astype(np.uint8)
    return image_pde

start = time.time()
image_pde = pde_diffusion(image, iterations=50, dt=0.05)
end = time.time()

times["PDE"] = end - start
show_image("PDE", image_pde)

# --- Zaman Grafiği ---
plt.figure(figsize=(12, 6))
plt.bar(list(times.keys()), list(times.values()), color='orange')
plt.xlabel("seconds")
plt.title("CPU-1 Compare")
plt.tight_layout()
plt.show()

###########
print("\n--- Execution Time Table ---")
print("Operation           Time (s)")
print("-----------------------------")
for op, t in times.items():
    print(f"{op:<20} {t:.6f}")
