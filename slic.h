#ifndef _SLIC_IMPLEMENTATION_HH
#define	_SLIC_IMPLEMENTATION_HH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include"color_space_conversion.h"

struct cluster{
	float l;
	float a;
	float b;
	int x;
	int y;
};

#ifdef DEBUG
void paintPixel(unsigned char* rgb_img, int w, int h, int c, int x, int y, int r, int g, int b);
#endif

float gradient(float* img_cielab, int w, int h, int c, int x, int y);

float Distance_D(struct cluster* center_k, float* img_cielab, int xi, int yi, int S, float m);

float* SLIC(float* cielab_img, int w, int h, int c, int K, float m, struct cluster** centers, int* sizeC, int max_iter);
#endif

