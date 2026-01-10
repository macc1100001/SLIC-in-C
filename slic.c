#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Reference white point for D65 illuminant
#define Xn 0.95047
#define Yn 1.00000
#define Zn 1.08883


struct cluster{
	float l;
	float a;
	float b;
	int x;
	int y;
};

float gamma_correction(float val){
	if(val > 0.04045)
		return powf((val+0.055) / 1.055, 2.4); 
	else
		return val / 12.92;
}

float f(float t){
	if(t > 0.008856)
		return powf(t, 1.0/3.0);
	else
		return 7.787 * t + (16.0/116.0);
}

/*
 * Returned cielab is interleaved, L a b. In that order
 *
 */

float* rgb_to_cielab(unsigned char* rgb, int w, int h, int c){
	float* cielab_color = malloc(w*h*3*sizeof(float));
	for(int i = 0; i < w; ++i){
		for(int j = 0; j < h*c; j+=c){
			unsigned char r_o = *(rgb+i*h*c + j);
			unsigned char g_o = *(rgb+i*h*c + j+1);
			unsigned char b_o = *(rgb+i*h*c + j+2);

			float* L = (cielab_color+i*h*c + j);
			float* a = (cielab_color+i*h*c + j+1);
			float* b = (cielab_color+i*h*c +j+2);

			// Normalize RGB values
			float r_n = r_o / 255.0;	// R'
			float g_n = g_o / 255.0;	// G'
			float b_n = b_o / 255.0;	// B'

			// Apply gamma correction
			r_n = gamma_correction(r_n);
			g_n = gamma_correction(g_n);
			b_n = gamma_correction(b_n);

			// Convert to XYZ color space
			float X = r_n * 0.4124564 + g_n * 0.3575761 + b_n * 0.1804375;
			float Y = r_n * 0.2126729 + g_n * 0.7151522 + b_n * 0.0721750;
			float Z = r_n * 0.0193339 + g_n * 0.1191920 + b_n * 0.9503041;

			*L = 116 * f(Y/Yn) - 16;
			*a = 500 * (f(X/Xn) - f(Y/Yn));
			*b = 200 * (f(Y/Yn) - f(Z/Zn));

		}
	}

	return cielab_color;
}

void paintPixel(unsigned char* rgb_img, int w, int h, int c, int x, int y, int r, int g, int b){
	// Make pixel bigger by painting not a single pixel, but an entire 4x4 block representing it
	int blockSize = 5;
	for(int i = -blockSize; i <= blockSize; ++i){
		for(int j = -blockSize; j <= blockSize; ++j){
			*(rgb_img+(x+i)*h*c + (y+j)*c) = r;
			*(rgb_img+(x+i)*h*c + (y+j)*c+1) = g;
			*(rgb_img+(x+i)*h*c + (y+j)*c+2) = b;
		}
	}
}

float gradient(float* img_cielab, int w, int h, int c, int x, int y, float* grad_x, float* grad_y){
	//float grad_x, grad_y;
	*grad_x = 0.0;
	*grad_y = 0.0;
	for(int i = 0; i < 3; ++i){
		float term_x = *(img_cielab+(x+1)*h*c + y*c+i) - *(img_cielab+(x-1)*h*c + y*c+i);
		float term_y = *(img_cielab+x*h*c + (y+1)*c+i) - *(img_cielab+x*h*c + (y-1)*c+i);
		*grad_x += term_x*term_x;
		*grad_y += term_y*term_y;
	}
	return *grad_x + *grad_y; 
}

//float Distance_D(unsigned char* rgb_img, int w, int h, int c, int blockSize, struct cluster* centers, int sizeC){
float Distance_D(float* img_cielab, int w, int h, int c, int k, int i, int S, int m)
	float dlab, dxy, Ds;
	for(int i = 0; i < 3; ++i){
		float l_diff = *(img_cielab+k*h*c + )
	}
	/*for(int k = 0; k < sizeC; ++k){
		for(int i = -blockSize; i <= blockSize; ++i){
			for(int j = -blockSize; j <= blockSize; ++j){
				dlab = 
			}
		}
	}*/
	return 0.0;
}

float* SLIC(float* cielab_img, int w, int h, int c, int K, int m, struct cluster* cluster_centers, int* sizeC){
	int S = round(sqrt((w*h)/K));
	int sizeOfS = round((w*h)/K);
	//struct cluster* cluster_centers = malloc(K * sizeof(struct cluster));
	int* labels = malloc(w*h*sizeof(int));
	float* distances = malloc(w*h*sizeof(float));
	for(int i = 0; i < w*h; ++i){labels[i] = -1;}
	for(int i = 0; i < w*h; ++i){distances[i] = INFINITY;}
	for(int i = 0; i < 10; ++i){printf("label(%d) = %d, dist(%d) = %f\n", i, labels[i], i, distances[i]);}
	int k = 0;
	for(int i = S; i < w; i+=S){
		for(int j = c*S; j < h*c; j+=c*S){
			(cluster_centers+k)->l = *(cielab_img+i*h*c + j);
			(cluster_centers+k)->a = *(cielab_img+i*h*c + j+1);
			(cluster_centers+k)->b = *(cielab_img+i*h*c + j+2);
			(cluster_centers+k)->x = i;
			(cluster_centers+k)->y = j/c;
			++k;
		}
	}
	printf("Superpixel size: %d\n", sizeOfS);
	printf("S = %d\n", S);
	printf("Requested K = %d, real k = %d\n", K, k);
	/*for(int i = 0; i < k; ++i){
		printf("center (%d,%d) => (%f,%f,%f)\n", 
				(cluster_centers+i)->x,
				(cluster_centers+i)->y,
				(cluster_centers+i)->l,
				(cluster_centers+i)->a,
				(cluster_centers+i)->b);
	}*/
	// Gradient descent on cluster centers
	float grad_x, grad_y;
	float grad = INFINITY;
	float umbral = 0.01;
	//int iter = 0;
	for(int i = 0; i < k; ++i){
		grad = gradient(cielab_img, w, h, c, (cluster_centers+i)->x, (cluster_centers+i)->y, &grad_x, &grad_y);
		//printf("Start--center %d, grad = %f\n", i+1, grad);
		do{
			//++iter;
			//grad = gradient(cielab_img, w, h, c, (cluster_centers+i)->x, (cluster_centers+i)->y, &grad_x, &grad_y);
			float alpha = 0.01;
			(cluster_centers+i)->x -= alpha*grad_x;
			(cluster_centers+i)->y -= alpha*grad_y;
			grad = gradient(cielab_img, w, h, c, (cluster_centers+i)->x, (cluster_centers+i)->y, &grad_x, &grad_y);
			//printf("grad_x = %f, grad_y = %f\n", grad_x, grad_y);
			//printf("grad = %f\n", grad);
			//printf("iters = %d\n", iter);
		}while(grad > umbral);
		//printf("iters = %d, grad = %f\n", iter, grad);
		//iter = 0;
		//grad = gradient(cielab_img, w, h, c, (cluster_centers+i)->x, (cluster_centers+i)->y, &grad_x, &grad_y);
		//printf("End--center %d, grad = %f\n", i+1, grad);
	}

	// Compute distances in a 2S x 2S grid around cluster centers
	float error = INFINITY;
	umbral = 1e-4;
	do{
		for(int i = 0; i < k; ++i){
			// Assign the best matching pixels from a 2Sx2S square neighborhood around
			// the cluster center according to the distance measure
			for(){
				for(){

				}
			}
		}
		// Compute new cluster centers and residual error(L1 distance between previous
		// centers and recomputed centers)

	}while(error <= umbral);
	// TODO: Enforce connectivity

	*sizeC = k;
	free(labels);
	labels = NULL;
	free(distances);
	distances = NULL;
	//free(cluster_centers);
	//cluster_centers = NULL;
	return NULL;
}

int main(int argc, char** argv){
	
	int opt, k = 1500;
	char* image_path;
	while((opt = getopt(argc, argv, "k:")) != -1){
		switch(opt){
			case 'k':
				k = atoi(optarg);
			default:
				fprintf(stderr, "Usage: %s [-k number of Superpixels] imagepath\n", argv[0]);
				exit(EXIT_FAILURE);
		}
	}
	if(optind >= argc){
		fprintf(stderr, "Expected arguments after options\n");
		fprintf(stderr, "Usage: %s [-k number of Superpixels] imagepath\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	image_path = argv[optind];
	
	int x_dim, y_dim, n_channels;
	unsigned char* image_data = stbi_load(image_path, &x_dim, &y_dim, &n_channels, 3);

	if(image_data == NULL){
		fprintf(stderr, "Error loading the image provided\n");
		exit(EXIT_FAILURE);
	}

	float* cielab = rgb_to_cielab(image_data, x_dim, y_dim, n_channels);
	printf("Image size: %d x %d %d\n", x_dim, y_dim, n_channels);
	printf("rgb values at pixel (0,0) => (%d,%d,%d)\n", image_data[0], image_data[1], image_data[2]);
	printf("CIELab values at pixel (0,0) => (%f,%f,%f)\n", cielab[0], cielab[1], cielab[2]);

	// Allocate inside SLIC
	struct cluster* centers = malloc(1444 * sizeof(struct cluster));
	int sizeC;

	float* segmented = SLIC(cielab, x_dim, y_dim, n_channels, k, 10, centers, &sizeC);
	if(centers == NULL){
		fprintf(stderr, "Error allocating memory\n");
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < sizeC; ++i){
		paintPixel(image_data, x_dim, y_dim, n_channels, (centers+i)->x, (centers+i)->y, 255, 0, 0);
	}
	int result = stbi_write_jpg("gato_mod.jpeg", x_dim, y_dim, n_channels, image_data, 100);

	free(centers);
	centers = NULL;


	/*
	for(int i = 0; i < x_dim; ++i){
		for(int j = 0; j < y_dim*n_channels; j+=n_channels){
			unsigned char* r = (image_data+i*y_dim*n_channels + j);
			unsigned char* g = (image_data+i*y_dim*n_channels + j+1);
			unsigned char* b = (image_data+i*y_dim*n_channels + j+2);
			*g = 80;
			*b =20;
		}
	}
	*/
	//for(int i = 0; i < x_dim; ++i){
	//	for(int j = 0; j < y_dim; ++j){
	//		if(image_data[i*y_dim + j] > 150)
	//			image_data[i*y_dim + j] = (int)fmin((image_data[i*y_dim + j] + 90), 255);
	//		if(image_data[i*y_dim + j] < 128)
	//			image_data[i*y_dim + j] = (int)fmax((image_data[i*y_dim + j] - 90), 0);
	//	}
	//}
    //int result = stbi_write_jpg("gato_mod.jpeg", x_dim, y_dim, n_channels, image_data, 100);

	stbi_image_free(image_data);
	free(cielab);
	cielab = NULL;
	image_data = NULL;

	exit(EXIT_SUCCESS);
}
