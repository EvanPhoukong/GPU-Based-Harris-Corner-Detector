import argparse
from mycv2 import imread, imwrite

import numpy as np
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
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


    sigma = float(args["sigma"])
    block = int(args["block"])
    hcwin = int(args["hcwin"])
    topfeatures = Corners(image, sigma, hcwin, block, args["image"])

    with open("corners.txt", 'w') as ofile:
        for (y, x) in topfeatures:
            ofile.write(f"Corner: {(int(y), int(x))}\n")

# def convolve(image, kernel):
#     height, width = image.shape
#     ker_h, ker_w = kernel.shape
#     #img_array = np.zeros_like((height, width))
#     img_array = np.zeros((height, width))
#     for i in range(height):
#         for j in range(width):
#             sum = 0
#             for k in range(ker_h):
#                 for m in range(ker_w):
#                     offseti = -1 * math.floor(ker_h / 2) + k
#                     offsetj = -1 * math.floor(ker_w / 2) + m
#                     if i + offseti in range(height) and j + offsetj in range(width):
#                         sum += image[i + offseti][j + offsetj] * kernel[k][m]
#             img_array[i][j] = sum
#     return img_array




# def covariance(image, window, gX, gY):
#     height, width = image.shape
#     all_Z = []
#     for i in range(height):
#         for j in range(width):
#             Z, iyy, ixiy, ixx, w = [], 0, 0, 0, math.floor(window/2)
#             for offseti in range(-w, w + 1):
#                 for offsetj in range(-w, w + 1):
#                     if i + offseti in range(height) and j + offsetj in range(width):
#                         vert = gX[i + offseti][j + offsetj]
#                         horiz = gY[i + offseti][j + offsetj]
#                         ixx += vert ** 2
#                         iyy += horiz ** 2
#                         ixiy += vert * horiz
#             Z = [[ixx, ixiy],[ixiy, iyy]]
#             all_Z.append(Z)
#     return np.array(all_Z)


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

def gpu_shared_convolve(image, kernel, block_Size):
    global conv_dtoh_begin
    global conv_dtoh_end
    global conv_kernel_begin
    global conv_kernel_end
    global conv_htod_begin
    global conv_htod_end

    height, width = image.shape
    ker_h, ker_w = kernel.shape
    image_size = height * width 

    flat_image = image.astype(np.float32)
    flat_kernel = kernel.astype(np.float32)
    c_image = np.zeros_like(flat_image).astype(np.float32)
    
    flat_image = np.ndarray.flatten(flat_image)
    flat_kernel = np.ndarray.flatten(flat_kernel)
    c_image = np.ndarray.flatten(c_image)

    flat_image_gpu = drv.mem_alloc(flat_image.nbytes)
    flat_kernel_gpu = drv.mem_alloc(flat_kernel.nbytes)
    c_image_gpu = drv.mem_alloc(c_image.nbytes)

    conv_htod_begin = time.time()
    drv.memcpy_htod(flat_image_gpu, flat_image)
    drv.memcpy_htod(flat_kernel_gpu, flat_kernel)
    conv_htod_end = time.time()

    mod = SourceModule("""
    #define BLOCK 16
    __global__ void shared_convolution(float *image, float *kernel, float *c_image, int ker_h, int ker_w, int image_size, int width) {
        int row, col, offseti, offsetj, temp_index, tx, ty;
        float sum = 0;
        tx = threadIdx.x;
        ty = threadIdx.y;
        row = tx + blockIdx.x * blockDim.x;
        col = ty + blockIdx.y * blockDim.y;
        __shared__ float Ashared[BLOCK][BLOCK];
        if (row < width && col < width) {
            Ashared[tx][ty] = image[row * width + col];
        } else {
            Ashared[tx][ty] = 0;       
        }
        __syncthreads();
        if (row < width && col < width) {
            for (int k = 0; k < ker_h; k++) {   
                for (int m = 0; m < ker_w; m++) {
                    offseti = -1 * ker_h/2 + k;
                    offsetj = -1 * ker_w/2 + m;
                    temp_index = (row + offseti) * width + (col + offsetj);
                    if ((tx + offseti < blockDim.x) && (tx + offseti >= 0) && (ty + offsetj >= 0) && (ty + offsetj < blockDim.y)) {
                        sum += Ashared[tx + offseti][ty + offsetj] * kernel[k * ker_w + m];
                    } else if ((row + offseti >= 0) && (row + offseti < width) && (col + offsetj >= 0) && (col + offsetj < width)) {
                        sum += image[temp_index] * kernel[k * ker_w + m];
                    }
                }
            }
            c_image[row * width + col] = sum; 
        }   
    }
    """)

    shared_convolution = mod.get_function("shared_convolution")

    grid_Size = (width + block_Size - 1) // block_Size
    conv_kernel_begin = time.time()
    shared_convolution(flat_image_gpu, flat_kernel_gpu, c_image_gpu, np.int32(ker_h), np.int32(ker_w), np.int32(image_size),
                      np.int32(width), block = (block_Size, block_Size, 1), grid = (grid_Size, grid_Size, 1))
    conv_kernel_end = time.time()

    image_gpu_result = np.zeros_like(c_image).astype(np.float32)

    conv_dtoh_begin = time.time()
    drv.memcpy_dtoh(image_gpu_result, c_image_gpu)
    conv_dtoh_end = time.time()

    image_gpu_result = image_gpu_result.reshape(height, width)

    return image_gpu_result
    
def gpu_shared_covariance(image, hcwin, gX, gY, block_Size):
    global cov_dtoh_begin
    global cov_dtoh_end
    global cov_kernel_begin
    global cov_kernel_end
    global cov_htod_begin
    global cov_htod_end

    height, width = image.shape
    image_size = height * width 

    gX = gX.astype(np.float32)
    gY = gY.astype(np.float32)
    eigen_vals = np.empty(image_size).astype(np.float32)

    flat_gX = np.ndarray.flatten(gX)
    flat_gY = np.ndarray.flatten(gY)
    flat_eigen_vals = np.ndarray.flatten(eigen_vals)

    flat_gX_gpu = drv.mem_alloc(flat_gX.nbytes)
    flat_gY_gpu = drv.mem_alloc(flat_gY.nbytes)
    eigen_vals_gpu = drv.mem_alloc(flat_eigen_vals.nbytes)

    cov_htod_begin = time.time()
    drv.memcpy_htod(flat_gX_gpu, flat_gX)
    drv.memcpy_htod(flat_gY_gpu, flat_gY)
    cov_htod_end = time.time()

    mod = SourceModule("""
    # define BLOCK 16
    __global__ void shared_covariance(float *gX, float *gY, float *eigen_vals_gpu, int window, int image_size, int width) {
        int row, col, w, temp_index, tx, ty;
        float vert, horiz, iyy, ixiy, ixx;
        iyy = 0.0, ixiy = 0.0, ixx = 0.0, w = window/2;
        tx = threadIdx.x;
        ty = threadIdx.y;
        row = tx + blockIdx.x * blockDim.x;
        col = ty + blockIdx.y * blockDim.y;
        __shared__ float Ashared[BLOCK][BLOCK];
        __shared__ float Bshared[BLOCK][BLOCK];
        if (row < width && col < width) {
            Ashared[tx][ty] = gX[row * width + col];
            Bshared[tx][ty] = gY[row * width + col];
        } else {
            Ashared[tx][ty] = 0;
            Bshared[tx][ty] = 0;       
        }
        __syncthreads();
        if (row < width && col < width){
            for (int offseti = -w; offseti < w + 1; offseti++) {   
                for (int offsetj = -w; offsetj < w + 1; offsetj++) {
                    temp_index = (row + offseti) * width + (col + offsetj);
                    if ((tx + offseti < blockDim.x) && (tx + offseti >= 0) && (ty + offsetj >= 0) && (ty + offsetj < blockDim.y)) {
                        vert = Ashared[tx + offseti][ty + offsetj];
                        horiz = Bshared[tx + offseti][ty + offsetj];
                        ixx += vert * vert;
                        iyy += horiz * horiz;
                        ixiy += vert * horiz;
                    } else if ((row + offseti >= 0) && (row + offseti < width) && (col +offsetj >= 0) && (col + offsetj < width)) {
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

    shared_covariance= mod.get_function("shared_covariance")

    grid_Size = (width + block_Size - 1) // block_Size

    cov_kernel_begin = time.time()
    shared_covariance(flat_gX_gpu, flat_gY_gpu, eigen_vals_gpu, np.int32(hcwin), np.int32(image_size),
                      np.int32(width), block = (block_Size, block_Size, 1), grid = (grid_Size, grid_Size, 1))
    cov_kernel_end = time.time()
    
    image_gpu_result = np.zeros_like(flat_eigen_vals).astype(np.float32)

    cov_dtoh_begin = time.time()
    drv.memcpy_dtoh(image_gpu_result, eigen_vals_gpu)
    cov_dtoh_end = time.time()

    return image_gpu_result.reshape(height, width)

def Corners(image, sigma, hcwin, block, image_name):


    gK = GaussianKernel(sigma)
    gD = (GaussianDerivative(sigma))[::-1]
    gkT = gK.T
    gdT = gD.T
    # cpu_conv_begin = time.time()
    # temp_horizontal_cpu = convolve(image, gK)
    # gY_cpu = convolve(temp_horizontal_cpu, gdT)
    # temp_vertical_cpu = convolve(image, gkT)
    # gX_cpu = convolve(temp_vertical_cpu, gD)
    # cpu_conv_end = time.time()

    temp_horizontal_gpu = gpu_shared_convolve(image, gK, block)
    gY_gpu = gpu_shared_convolve(temp_horizontal_gpu, gdT, block)
    temp_vertical_gpu = gpu_shared_convolve(image, gkT, block)
    gX_gpu = gpu_shared_convolve(temp_vertical_gpu, gD, block)

    # if np.allclose(temp_horizontal_cpu, temp_horizontal_gpu, 1e-3, 1e-5):
    #     pass
    # else:
    #     print("NOT THE SAME temp_H")

    # if np.allclose(temp_vertical_cpu, temp_vertical_gpu, 1e-3, 1e-5):
    #     pass
    # else:
    #     print("NOT THE SAME temp_V")

    # if np.allclose(gY_cpu, gY_gpu, 1e-3, 1e-5):
    #     pass
    # else:
    #     print("NOT THE SAME GY")

    # if np.allclose(gX_cpu, gX_gpu, 1e-3, 1e-5):
    #     pass
    # else:
    #     print("NOT THE SAME GX")

    # cpu_cov_begin = time.time()
    # covar_matrices_cpu = covariance(image, hcwin, gX_cpu, gY_cpu)
    # cpu_cov_end = time.time()
    eigen_vals_gpu = gpu_shared_covariance(image, hcwin, gX_gpu, gY_gpu, block)

    # cpu_eig_begin = time.time()
    # corners = np.zeros(image.shape).astype(np.float32)
    # height, width = image.shape
    # index = 0
    # for i in range(height):
    #     for j in range(width):
    #         eigval, eigvec = np.linalg.eig(covar_matrices_cpu[index])
    #         lambda1, lambda2 = sorted(eigval)
    #         corners[i, j] = lambda1 * lambda2 - 0.04 * ((lambda1+lambda2) ** 2)
    #         index += 1
    # cpu_eig_end = time.time()

    # corners = corners.astype(np.float32)
    # if np.allclose(corners, eigen_vals_gpu, 1e-1, 1e-1):
    #     pass
    # else:
    #     print("NOT THE SAME eigens")

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

    kernel_time = (conv_kernel_end - conv_kernel_begin) + (cov_kernel_end - cov_kernel_begin)
    htod_time = (conv_htod_end - conv_htod_begin) + (cov_htod_end - cov_htod_begin)
    dtoh_time = (conv_dtoh_end - conv_dtoh_begin) + (cov_dtoh_end - cov_dtoh_begin)
    # cpu_time = (cpu_eig_end - cpu_eig_begin) + (cpu_conv_end - cpu_conv_begin) + (cpu_cov_end - cpu_cov_begin)

    print(f"Image: {image_name}, Sigma: {sigma}, HC Window Size: {hcwin}, GPU Kernel Time: {kernel_time}, Host to Device Time: {htod_time}, Device to Host Time: {dtoh_time}, Total GPU Time: {kernel_time + htod_time + dtoh_time}")

    return final_corners



if __name__ == "__main__":
    main()

