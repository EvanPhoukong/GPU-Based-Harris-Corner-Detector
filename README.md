# GPU-Based-Harris-Corner-Detector
CUDA-accelerated Harris Corner Detector for real-time corner detection on high-resolution images
• Integrated custom GPU kernels in C for convolution (CGMA: 6) and covariance (CGMA: 301), leveraging shared
memory and parallelism to achieve runtimes under 0.1 seconds on 4500x4500 images.
• Benchmarked GPU performance by analyzing kernel execution times and data transfer durations (host-to-device and
device-to-host), achieving 500x faster execution times on average compared to CPU-based execution
• Implemented non-maximum suppression to filter out closely located corner points, ensuring only the most prominent
corners are retained, using NumPy for efficient vectorized operations
