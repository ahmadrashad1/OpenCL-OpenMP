#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <SDL2/SDL.h>

// Define the number of points for the circle and the radius
#define N_POINTS 1000
#define RADIUS 200
#define J 300  // Origin x-coordinate
#define K 300  // Origin y-coordinate
#define PI 3.14159265358979323846

// Taylor series approximation for sin(x)
double sin_taylor(double t) {
    double term = t;  // First term: t
    double sum = term;
    for (int n = 1; n < 10; ++n) {
        term *= -t * t / ((2 * n) * (2 * n + 1));  // Next term
        sum += term;
    }
    return sum;
}

// Taylor series approximation for cos(x)
double cos_taylor(double t) {
    double term = 1.0;  // First term: 1
    double sum = term;
    for (int n = 1; n < 10; ++n) {
        term *= -t * t / ((2 * n - 1) * (2 * n));  // Next term
        sum += term;
    }
    return sum;
}

void draw_circle(SDL_Renderer* renderer) {
    int i;
    double x, y;
    double t;
    double sin_t, cos_t;
    double final_sin_t = 0.0, final_cos_t = 0.0;

    // Use OpenMP to parallelize the loop for calculating (x, y)
    #pragma omp parallel for private(x, y, sin_t, cos_t, t) shared(renderer) reduction(+:final_sin_t, final_cos_t)
    for (i = 0; i < N_POINTS; ++i) {
        t = 2 * PI * i / N_POINTS;  // Angle from 0 to 2*pi
        sin_t = sin_taylor(t);
        cos_t = cos_taylor(t);

        // Accumulate the results for final output
        final_sin_t += sin_t;
        final_cos_t += cos_t;

        // Parametric equations for the circle
        x = RADIUS * cos_t + J;
        y = RADIUS * sin_t + K;

        // Render the point (x, y)
        // Use a critical section to ensure that the drawing is not interrupted by other threads
        #pragma omp critical
        SDL_RenderDrawPoint(renderer, (int)x, (int)y);
    }

    // Synchronize threads to ensure the final result is printed only once
    #pragma omp barrier

    // Print the final Taylor series results for sin(t) and cos(t)
    printf("Final Taylor series Result = sin(t) = %.5f, cos(t) = %.5f\n", final_sin_t, final_cos_t);
}

int main(int argc, char* argv[]) {
    // Initialize SDL2
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // Create a window
    SDL_Window* window = SDL_CreateWindow("Circle with OpenMP", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 600, 600, SDL_WINDOW_SHOWN);
    if (!window) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // Create a renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // Set renderer color to black (background color)
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    // Start measuring time
    double start_time = omp_get_wtime();

    // Set the draw color to white for drawing the circle
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

    // Draw the circle
    draw_circle(renderer);

    // End measuring time
    double end_time = omp_get_wtime();
    printf("Total Execution Time = %.6f seconds\n", end_time - start_time);

    // Present the renderer (show the drawn points on the screen)
    SDL_RenderPresent(renderer);

    // Wait for 5 seconds to display the circle
    SDL_Delay(5000);

    // Cleanup SDL resources
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

