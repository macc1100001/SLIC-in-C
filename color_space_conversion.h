#ifndef COLOR_SPACE_CONVERSION_IMPLEMENTATION
#define COLOR_SPACE_CONVERSION_IMPLEMENTATION

#include<math.h>
#include<stdlib.h>

/*
 * Reference for colorspace conversions
 * http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
 *
 */

// Reference white point for D65 illuminant
#define Xn 0.95047
#define Yn 1.00000
#define Zn 1.08883

float gamma_correction(float);

float f(float);

float* rgb_to_cielab(unsigned char*, int, int, int);

float srgb_companding(float);

unsigned char* cielab_to_rgb(float*, int, int, int);

#endif
