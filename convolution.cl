__kernel void convolution(
    __global float *input,
    __global float *output,
    const int width,
    const int height,
    __constant float *filter,
    const int filterSize)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int halfSize = filterSize / 2;
    float sum = 0.0f;
   
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            int nx = x + i;
            int ny = y + j;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx] * filter[(i + halfSize) * filterSize + (j + halfSize)];
            }
        }
    }
   
    output[y * width + x] = sum;

    if (x == 0 && y == 0) {
        printf("sum: %f\n", sum);
    }
}

