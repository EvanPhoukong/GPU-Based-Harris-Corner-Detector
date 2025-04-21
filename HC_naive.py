from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import argparse, cv2, os
from mycv2 import imread, imwrite
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import math, os, sys, time, argparse, cv2

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--sigma")
    parser.add_argument("--hcwin")
    parser.add_argument("--block")
    args = vars(parser.parse_args())
    image = imread(args["image"])

    #Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    #filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    #print(filename)

    sigma = float(args["sigma"])
    block = int(args["block"])
    hcwin = int(args["hcwin"])
    topfeatures = Corners(image, sigma, hcwin, block, args["image"])
    #img = cv2.imread(filename, 0)
    #directory = os.path.dirname(filename)
    #image = cv2.imread(filename, 1)
    # with open("corners.txt", 'w') as ofile:
    #     for (y, x) in topfeatures:
    #         ofile.write(f"Corner: {(y, x)}\n")
    #for (y, x) in topfeatures:
    #    cv2.putText(image, 'X', (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    #cv2.imshow("initial frame", image)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

def GaussianDerivative(sigma):
    a = int(2.5 * sigma - 0.5)
    w = 2 * a + 1

    sum = 0

    G = np.zeros((w, 1))
    for i in range(0, w):
        G[i] = (-1 * (i - a) * np.exp((-1 * (i - a) * (i - a)) / (2 * sigma * sigma)))
        sum = sum - i * G[i]

    G = G / sum
    return G


def GaussianKernel(sigma):
    a = int(2.5 * sigma - 0.5)
    w = 2 * a + 1

    sum = 0

    G = np.zeros((w, 1))
    for i in range(0, w):
        G[i] = np.exp((-1 * (i - a) * (i - a)) / (2 * sigma * sigma))
        sum = sum + G[i]

    G = G / sum
    return G

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def gpu_convolve(image, kernel, block_Size):
    global conv_dtoh_begin
    global conv_dtoh_end
    global conv_kernel_begin
    global conv_kernel_end
    global conv_htod_begin
    global conv_htod_end
    global image_bytes
    global kernel_bytes
    global output_bytes

    height, width = image.shape
    ker_h, ker_w = kernel.shape
    image_size = height * width 

    flat_image = image.astype(np.float32)
    flat_kernel = kernel.astype(np.float32)
    c_image = np.zeros_like(flat_image).astype(np.float32)
    
    flat_image = np.ndarray.flatten(flat_image)
    flat_kernel = np.ndarray.flatten(flat_kernel)
    c_image = np.ndarray.flatten(c_image)

    image_bytes = flat_image.nbytes
    kernel_bytes = flat_kernel.nbytes
    output_bytes = c_image.nbytes

    flat_image_gpu = drv.mem_alloc(image_bytes)
    flat_kernel_gpu = drv.mem_alloc(kernel_bytes)
    c_image_gpu = drv.mem_alloc(output_bytes)

    conv_htod_begin = time.time()
    drv.memcpy_htod(flat_image_gpu, flat_image)
    drv.memcpy_htod(flat_kernel_gpu, flat_kernel)
    drv.Context.synchronize()
    conv_htod_end = time.time()

    mod = SourceModule("""
    __global__ void naive_convolution(float *image, float *kernel, float *c_image, int ker_h, int ker_w, int image_size, int width) {
        int row, col, offseti, offsetj, temp_index;
        int ker_size = ker_h * ker_w;
        float sum = 0;
        row = threadIdx.x + blockIdx.x * blockDim.x;
        col = threadIdx.y + blockIdx.y * blockDim.y;
        if (row < width && col < width) {
            for (int k = 0; k < ker_h; k++) {   
                for (int m = 0; m < ker_w; m++) {
                    offseti = -1 * ker_h/2 + k;
                    offsetj = -1 * ker_w/2 + m;
                    temp_index = (row + offseti) * width + (col + offsetj);
                    if ((row +offseti >= 0) && (row + offseti < width) && (col +offsetj >= 0) && (col + offsetj < width)) {
                        sum += image[temp_index] * kernel[k * ker_w + m];
                    }
                }
            }
            c_image[row * width + col] = sum; 
        }   
    }
    """)

    naive_convolution = mod.get_function("naive_convolution")

    grid_Size = (width + block_Size - 1) // block_Size
    conv_kernel_begin = time.time()
    naive_convolution(flat_image_gpu, flat_kernel_gpu, c_image_gpu, np.int32(ker_h), np.int32(ker_w), np.int32(image_size),
                      np.int32(width), block = (block_Size, block_Size, 1), grid = (grid_Size, grid_Size, 1))
    drv.Context.synchronize()
    conv_kernel_end = time.time()

    image_gpu_result = np.zeros_like(c_image).astype(np.float32)

    conv_dtoh_begin = time.time()
    drv.memcpy_dtoh(image_gpu_result, c_image_gpu)
    drv.Context.synchronize()
    conv_dtoh_end = time.time()

    image_gpu_result = image_gpu_result.reshape(height, width)

    return image_gpu_result
    
def gpu_covariance(image, hcwin, gX, gY, block_Size):
    global cov_dtoh_begin
    global cov_dtoh_end
    global cov_kernel_begin
    global cov_kernel_end
    global cov_htod_begin
    global cov_htod_end
    global gX_bytes
    global gY_bytes
    global eigen_bytes

    height, width = image.shape
    image_size = height * width 

    gX = gX.astype(np.float32)
    gY = gY.astype(np.float32)
    eigen_vals = np.empty(image_size).astype(np.float32)

    flat_gX = np.ndarray.flatten(gX)
    flat_gY = np.ndarray.flatten(gY)
    flat_eigen_vals = np.ndarray.flatten(eigen_vals)

    gX_bytes = flat_gX.nbytes
    gY_bytes = flat_gY.nbytes
    eigen_bytes = flat_eigen_vals.nbytes

    flat_gX_gpu = drv.mem_alloc(gX_bytes)
    flat_gY_gpu = drv.mem_alloc(gY_bytes)
    eigen_vals_gpu = drv.mem_alloc(eigen_bytes)
    
    cov_htod_begin = time.time()
    drv.memcpy_htod(flat_gX_gpu, flat_gX)
    drv.memcpy_htod(flat_gY_gpu, flat_gY)
    drv.Context.synchronize()
    cov_htod_end = time.time()

    mod = SourceModule("""
    __global__ void covariance(float *gX, float *gY, float *eigen_vals_gpu, int window, int image_size, int width) {
        int row, col, w, temp_index;
        float vert, horiz, iyy, ixiy, ixx;
        iyy = 0.0, ixiy = 0.0, ixx = 0.0, w = window/2;
        row = threadIdx.x + blockIdx.x * blockDim.x;
        col = threadIdx.y + blockIdx.y * blockDim.y;
        if (row < width && col < width){
            for (int offseti = -w; offseti < w + 1; offseti++) {   
                for (int offsetj = -w; offsetj < w + 1; offsetj++) {
                    if ((row +offseti >= 0) && (row + offseti < width) && (col +offsetj >= 0) && (col + offsetj < width)) {
                        temp_index = (row + offseti) * width + (col + offsetj);
                        vert = gX[temp_index];
                        horiz = gY[temp_index];
                        ixx += vert * vert;
                        iyy += horiz * horiz;
                        ixiy += vert * horiz;
                    }
                }
            }
            eigen_vals_gpu[row * width + col] = (ixx * iyy - ixiy * ixiy) - 0.04 * ((ixx + iyy) * (ixx + iyy));
        }
    }
    """)

    naive_covariance= mod.get_function("covariance")

    grid_Size = (width + block_Size - 1) // block_Size

    cov_kernel_begin = time.time()
    naive_covariance(flat_gX_gpu, flat_gY_gpu, eigen_vals_gpu, np.int32(hcwin), np.int32(image_size),
                      np.int32(width), block = (block_Size, block_Size, 1), grid = (grid_Size, grid_Size, 1))
    drv.Context.synchronize()
    cov_kernel_end = time.time()
    
    image_gpu_result = np.zeros_like(flat_eigen_vals).astype(np.float32)

    cov_dtoh_begin = time.time()
    drv.memcpy_dtoh(image_gpu_result, eigen_vals_gpu)
    drv.Context.synchronize()
    cov_dtoh_end = time.time()

    return image_gpu_result.reshape(height, width)

def Corners(image, sigma, hcwin, block, image_name):


    gK = GaussianKernel(sigma)
    gD = (GaussianDerivative(sigma))[::-1]
    gkT = gK.T
    gdT = gD.T

    temp_horizontal_gpu = gpu_convolve(image, gK, block)
    gY_gpu = gpu_convolve(temp_horizontal_gpu, gdT, block)
    temp_vertical_gpu = gpu_convolve(image, gkT, block)
    gX_gpu = gpu_convolve(temp_vertical_gpu, gD, block)

    eigen_vals_gpu = gpu_covariance(image, hcwin, gX_gpu, gY_gpu, block)

    corners = eigen_vals_gpu

    #THE BELOW LINES OF CODE WERE GENERATED USING CHATGPT.

    # --- Step 1: Sort pixels based on cornerness using NumPy ---
    # Flatten the corners matrix and get sorted indices (in descending order)
    flat_sorted_indices = np.argsort(corners, axis=None)[::-1]
    sorted_rows, sorted_cols = np.unravel_index(flat_sorted_indices, corners.shape)

    # --- Step 2: Non-maximum suppression using NumPy vectorized operations ---
    # We will build a list of selected corner points in (j, i) format
    final_corners = []

    min_distance = 15
    for idx in range(len(sorted_rows)):
        # Prepare candidate in (j, i) format as required
        candidate = np.array([sorted_cols[idx], sorted_rows[idx]])
        if final_corners:
            # Convert final_corners to a NumPy array for vectorized distance calculation
            existing = np.array(final_corners)
            # Compute Euclidean distances from candidate to each already selected corner
            distances = np.linalg.norm(existing - candidate, axis=1)
            if np.all(distances >= min_distance):
                final_corners.append((candidate[0], candidate[1]))
        else:
            final_corners.append((candidate[0], candidate[1]))
        if len(final_corners) == 50:
            break

    height, width = image.shape
    image_size = height * width
    convolution_flops = image_size * (3 * 2) #Kernel size * FLOPS
    convolution_bytes = image_size * (8) #2 global mem accesses * 4 bytes
    covariance_flops = image_size * (49 * 6 + 7) # Window size * flops in window + eigen calcs
    covariance_bytes = image_size * (8) #2 global mem accesses * 4 bytes
    conv_htod_bytes = image_bytes + kernel_bytes
    conv_dtoh_bytes = output_bytes
    cov_htod_bytes = gX_bytes +  gY_bytes
    cov_dtoh_bytes = eigen_bytes
    conv_time = (conv_kernel_end - conv_kernel_begin) * 4 #kernel called 4 times
    cov_time = (cov_kernel_end - cov_kernel_begin)
    conv_htod_time = (conv_htod_end - conv_htod_begin)
    cov_htod_time = (cov_htod_end - cov_htod_begin)
    conv_dtoh_time = (conv_dtoh_end - conv_dtoh_begin)
    cov_dtoh_time = (cov_dtoh_end - cov_dtoh_begin)

    with open("metrics.csv", "a") as ofile:
        ofile.write(f"{height}, {conv_time}, {cov_time}, {conv_htod_time}, {cov_htod_time}, {conv_dtoh_time}, {cov_dtoh_time}, {convolution_flops}, {convolution_bytes}, {covariance_flops}, {covariance_bytes}, {conv_htod_bytes}, {cov_htod_bytes}, {conv_dtoh_bytes}, {cov_dtoh_bytes}\n")
    return final_corners

if __name__ == "__main__":
    main()

