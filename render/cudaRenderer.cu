#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define THREADS_PER_BLOCK 256
////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    //logic to calculate bounding box, and outside in box, then return 
    //else do furtehr calculation 
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    //can you store the square of the radius so you don't need to compute everytime 
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb; //idk what this is 
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void cudaUpsweep(long long N, long long two_d, long long two_dplus_1, int* data, long long roundedN) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x; // Thread index
    long long i = idx * two_dplus_1; // Actual element index

    if(i + two_dplus_1 - 1 >= roundedN) return;

    data[i + two_dplus_1 - 1] += data[i + two_d - 1];
}

__global__ void cudaDownsweep(long long N, long long two_d, long long two_dplus_1, int* data, long long pow2N) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x; // Thread index
    long long i = idx * two_dplus_1; // Actual element index

    if(i + two_dplus_1 - 1 >= pow2N) return;

    int t = data[i + two_d - 1];
    data[i + two_d - 1] = data[i + two_dplus_1 - 1];
    data[i + two_dplus_1 - 1] += t;
}

__global__ void cudaMiddleStep(int* data, long long pow2N) {
    if(blockIdx.x * blockDim.x + threadIdx.x == 0) {
        data[pow2N - 1] = 0;
    }
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result

//FIND ME EXCLUSIVE
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.

    // Upsweep
    long long roundedN = nextPow2(N);

    for(int two_d = 1; two_d <= roundedN / 2; two_d *= 2) {
	    int two_dplus1 = 2 * two_d;

        int ops = roundedN / two_dplus1;
        int numBlocks = (ops + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	    cudaUpsweep<<<numBlocks, THREADS_PER_BLOCK>>>(N, two_d, two_dplus1, result, roundedN);
        cudaDeviceSynchronize();
    }

    cudaMiddleStep<<<1, THREADS_PER_BLOCK>>>(result, roundedN);
    cudaDeviceSynchronize();

    // Downsweep
    for(int two_d = roundedN / 2; two_d >= 1; two_d /= 2) {
	    int two_dplus1 = 2 * two_d;
        
        int ops = roundedN / two_dplus1;
        int numBlocks = (ops + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaDownsweep<<<numBlocks, THREADS_PER_BLOCK>>>(N, two_d, two_dplus1, result, roundedN);
        cudaDeviceSynchronize();
    }
}

//FIND ME REPEATS
__global__ void cuda_identify_transition_points(int* prefix_sum_array, int* result, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Unique thread index

    if (idx >= length - 1) return; // Ensure we stay within bounds for idx + 1

    // Check if we have a transition point
    if (prefix_sum_array[idx + 1] != prefix_sum_array[idx] + 1) {
        result[prefix_sum_array[idx]] = idx;
    }
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_not_repeats(int* device_input, int length, int* device_output, int *positions_mask) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    // Step 1: For each Bucket, call Exclusive Sum. Sync. 
    // Step 2: Go though each excluive_scan[num_buckets] to find the max. Non paralell 
    // Step 3: allocate array of size max * num_buckets. Memset it to -1 
    // Step 4: check not_equal on each item in the exclusive scan for each bucket, store into the array made in step 3 

    int numBlocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    pair_and_compare<<<numBlocks, THREADS_PER_BLOCK>>>(device_input, positions_mask, length);
    cudaDeviceSynchronize();

    // Step 2: Perform an exclusive scan on positions_mask (keeping it on the GPU)
    exclusive_scan(positions_mask, length, positions_mask); // Result remains in positions_mask
    // cudaDeviceSynchronize(); //do we need this? 

    // Step 3: Copy the last element of positions_mask to the CPU to get the count of repeats
    int result;
    cudaMemcpy(&result, &positions_mask[length - 1], sizeof(int), cudaMemcpyDeviceToHost);
    //possible need to synchronise? 

    // Step 4: Identify transition points using cuda_identify_transition_points
    cuda_identify_transition_points<<<numBlocks, THREADS_PER_BLOCK>>>(positions_mask, device_output, length);
    cudaDeviceSynchronize();

    return result; 
}

//
// Each thread processes a circle. 
// There are 4 arrays, one for each quadrant 
// The thread draws a bounding box, and assigns it to the quadrant
// Where do I CUDA memcopy? 

__global__ void kernelBucketCircles(int* mask_ptr, int dim_buckets, short bucket_size_x, short bucket_size_y) {
    int numCircles = cuConstRendererParams.numCircles;

    //TODO for Ishita: Understand 1D indexing 
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;
    // printf("I'm calculating for cricle %d\n", index);

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    //coordinates: (screenMinX, screenMinY), (screenMinX, screenMaxY), (screenMaxX,screenMinY), (screenMaxX,screenMaxY)
    // if(
    //     screenMinX < 0 || screenMinX > imageWidth ||
    //     screenMinX < 0 || screenMinX > imageWidth ||
    // )

    //assign buckets, for all the coordinates 
    int bucket_xidx_min = screenMinX/ bucket_size_x;
    int bucket_xidx_max = screenMaxX/ bucket_size_x;
    int bucket_yidx_min = screenMinY/ bucket_size_y;
    int bucket_yidx_max = screenMaxY/ bucket_size_y;

    // //set buckets, this ensures all the possible buckets are set 
    // //flattened_index=bucket_xidx ×(num_buckets×num_circles)+bucket_yidx×num_circles+index
    // mask_ptr[bucket_xidx_min * (dim_buckets * numCircles) + (bucket_yidx_min * numCircles) + index] = 1;
    // mask_ptr[bucket_xidx_min * (dim_buckets * numCircles) + (bucket_yidx_max * numCircles) + index] = 1;
    // mask_ptr[bucket_xidx_max * (dim_buckets * numCircles) + (bucket_yidx_min * numCircles) + index] = 1;
    // mask_ptr[bucket_xidx_max * (dim_buckets * numCircles) + (bucket_yidx_max * numCircles) + index] = 1;

    //need a for loop, as circles can span many boxes!!!
    for (int bx = bucket_xidx_min; bx <= bucket_xidx_max; bx++) {
        for (int by = bucket_yidx_min; by <= bucket_yidx_max; by++) {
            // Calculate the flattened index for the mask array
            int bucket_index = bx * (dim_buckets * numCircles) + by * numCircles + index;
            printf("Adding circle %d with bounding box  x:(%d, %d) y:(%d, %d) to bucket location (%d, %d)\n", index, screenMinX, screenMaxX, screenMinY, screenMaxY, bx, by);
            // Set the mask to indicate this circle is in this bucket
            mask_ptr[bucket_index] = 1;
        }
}
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a pixel. This should order and atomicity
__global__ void kernelRenderCircles(int* mask_ptr, int dim_buckets, short bucket_size_x, short bucket_size_y) {

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    int numCircles = cuConstRendererParams.numCircles;

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    // Divide the screen into the same grid of buckets you used in kernelBucketCircles
    int pixelBucketX = xIndex / bucket_size_x;
    int pixelBucketY = yIndex / bucket_size_y;

    // Find the circles in the Pixels Bucket 
    int startIndex = pixelBucketX * (dim_buckets * cuConstRendererParams.numCircles) + pixelBucketY * numCircles;

    if (xIndex >= imageWidth || yIndex >= imageHeight)
        return;
    
    for(int index = 0; index < numCircles; index++) { //range max
        if (mask_ptr[startIndex + index] == 1) { //check each of our indexes 
            int index3 = 3 * index;
            // read position and radius
            float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
            float  rad = cuConstRendererParams.radius[index];

            // compute the bounding box of the circle. The bound is in integer
            // screen coordinates, so it's clamped to the edges of the screen.
            short minX = static_cast<short>(imageWidth * (p.x - rad));
            short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
            short minY = static_cast<short>(imageHeight * (p.y - rad));
            short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

            // a bunch of clamps.  Is there a CUDA built-in for this?
            short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
            short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
            short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
            short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

            // Check if pixel is in circles bounding box
            if(screenMinX <= xIndex && xIndex <= screenMaxX && screenMinY <= yIndex && yIndex <= screenMaxY) {
                // Check if actual intersect the circle
                float invWidth = 1.f / imageWidth;
                float invHeight = 1.f / imageHeight;

                float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(xIndex) + 0.5f),
                                                invHeight * (static_cast<float>(yIndex) + 0.5f));

                float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (yIndex * imageWidth + xIndex)]);
                
                shadePixel(index, pixelCenterNorm, p, imgPtr);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");

    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {

    // Get image dimensions from the host-side `Image` object
    int imageWidth = image->width;
    int imageHeight = image->height;
    printf("Image Width = %d\n", imageWidth);
    printf("Image Height = %d\n", imageHeight);

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);

    //num of blocks
    //dim3 gridDim(4,4);

    // Define the number of buckets and bucket sizes
    int dim_buckets = 5; // MODIFY IF WANTED

    short bucket_size_x = ((imageWidth + dim_buckets - 1) / dim_buckets) + 1;
    //printf("")
    short bucket_size_y = ((imageHeight + dim_buckets - 1)) / dim_buckets + 1;

    int num_buckets = dim_buckets * dim_buckets; // Square number of buckets

    // Allocate mask array on CUDA device and set to zeros 
    int* mask_ptr;
    cudaMalloc(&mask_ptr, sizeof(int) * num_buckets * numCircles);
    cudaMemset(mask_ptr, 0, sizeof(int) * num_buckets * numCircles);

    dim3 gridDim(
        (imageWidth + blockDim.x - 1) / blockDim.x,
        (imageHeight + blockDim.y - 1) / blockDim.y
    );

    // Define block and grid dimensions for `kernelBucketCircles`
    int blockSize = 256; // Number of threads per block
    dim3 blockDimBucket(blockSize, 1, 1);
    dim3 gridDimBucket((numCircles + blockSize - 1) / blockSize);

    // Launch kernelBucketCircles with grid and block dimensions for circles
    kernelBucketCircles<<<gridDimBucket, blockDimBucket>>>(mask_ptr, dim_buckets, bucket_size_x, bucket_size_y);
    cudaDeviceSynchronize();

    //FIND ME
    //Mask Ptr is now populated 
    // Step 1: For each Bucket, call Exclusive Sum. Sync. Pass in copies of pointer because its gonna currput the data otherwise 
    prefix_sum_result_ptr = cuudamalloc(size = (numCircles + 1) * num_buckets)
    for (i in range numbuckets){
        bucket_start_ptr = mask_ptr + numCircles * i 
        result_start_ptr = prefix_sum_result_ptr + (numCircles + 1)*i
        excluive_scan(bucket_start_ptr,numCircles, result_start_ptr )
    }
    cudaDeviceSynchronize();

    // Step 2: Go though each excluive_scan[num_buckets] to find the max. Non paralell 
    bucket_num_circles[num_buckets]; 
    for (i in range num_buckets){
        bucket_num_circles[i] = prefix_sum_result_ptr[num_circles]
    }

    // Step 3: allocate array of size max * num_buckets. Memset it to -1 
    max = max(bucket_num_circles)
    not_equal_result_ptr = cudamalloc(max * num_buckets)
    cudamemset(-1)

    // Step 4: check not_equal on each item in the exclusive scan for each bucket, store into the array made in step 3 
    for (i in range num_buckets){
        prefix_sum_ptr = prefix_sum_result_ptr + i (num_circles + 1)
        result_ptr = not_equal_result_ptr + max * i; 
        cuda_identify_transition_points(prefix_sum_ptr, result_ptr, num_circles + 1 ) //is it correct to do num_circles + 1?
    }
    cudaDeviceSynchronize();


        // Allocate host memory for copying mask_ptr back
    int* host_mask_ptr = new int[num_buckets * numCircles];

    // // Copy mask_ptr from device to host
    // cudaMemcpy(host_mask_ptr, mask_ptr, sizeof(int) * num_buckets * numCircles, cudaMemcpyDeviceToHost);

    // // Print the contents of mask_ptr for debugging
    // printf("Contents of mask_ptr:\n");
    // for (int i = 0; i < num_buckets; i++) {
    //     printf("Bucket %d: ", i);
    //     for (int j = 0; j < numCircles; j++) {
    //         if (host_mask_ptr[i * numCircles + j] == 1) { // Only print if a circle is present
    //             printf("Circle %d ", j);
    //         }
    //     }
    //     printf("\n");
    // }


    kernelRenderCircles<<<gridDim, blockDim>>>(mask_ptr, dim_buckets, bucket_size_x, bucket_size_y);

    cudaDeviceSynchronize();
}
