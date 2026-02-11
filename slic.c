#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


#include"color_space_conversion.h"

struct cluster{
	float l;
	float a;
	float b;
	int x;
	int y;
};

#ifdef DEBUG
void paintPixel(unsigned char* rgb_img, int w, int h, int c, int x, int y, int r, int g, int b){
	// Make pixel bigger by painting not a single pixel, but an entire 4x4 block representing it
	int blockSize = 5;
	for(int i = -blockSize; i <= blockSize; ++i){
		for(int j = -blockSize; j <= blockSize; ++j){
			*(rgb_img+(x+i)*w*c + (y+j)*c) = r;
			*(rgb_img+(x+i)*w*c + (y+j)*c+1) = g;
			*(rgb_img+(x+i)*w*c + (y+j)*c+2) = b;
		}
	}
}
#endif

float gradient(float* img_cielab, int w, int h, int c, int x, int y){
	float grad_norm = 0.0;
	for(int i = 0; i < 3; ++i){
		float term_x = (*(img_cielab+(x+1)*w*c + y*c+i) - *(img_cielab+(x-1)*w*c + y*c+i))/2;
		float term_y = (*(img_cielab+x*w*c + (y+1)*c+i) - *(img_cielab+x*w*c + (y-1)*c+i))/2;
		grad_norm += term_x*term_x + term_y*term_y;
	}
	return grad_norm; 
}

float Distance_D(struct cluster* center_k, float* img_cielab, int xi, int yi, int S, float m){
	float dlab, dxy, Ds;
	float diff_li = (center_k)->l - *(img_cielab);
	float diff_ai = (center_k)->a - *(img_cielab+1);
	float diff_bi = (center_k)->b - *(img_cielab+2);
	float dx = (center_k)->x - xi;
	float dy = (center_k)->y - yi;
	dlab = sqrt((diff_li*diff_li) + (diff_ai*diff_ai) + (diff_bi*diff_bi));
	dxy = sqrt((dx*dx) + (dy*dy));
	Ds = dlab + (m/S)*dxy;
	return Ds;
}

float* SLIC(float* cielab_img, int w, int h, int c, int K, float m, struct cluster** cluster_centers, int* sizeC){
	int S = round(sqrt(w*h/K));
	int sizeOfS = round(w*h/K);
	int neighborhood = 1;
	*cluster_centers = malloc(K * sizeof(struct cluster));
	int* labels = malloc(w*h*sizeof(int));
	float* distances = malloc(w*h*sizeof(float));
	if(*cluster_centers == NULL || labels == NULL || distances == NULL){
		free(*cluster_centers);
		free(labels);
		free(distances);
		*cluster_centers = NULL;
		labels = NULL;
		distances = NULL;
		return NULL;
	}
	for(int i = 0; i < w*h; ++i){labels[i] = -1;} 
	for(int i = 0; i < w*h; ++i){distances[i] = INFINITY;}
	int k = 0; // Number of real cluster centers
	for(int i = S; i < h; i+=S){
		for(int j = c*S; j < w*c; j+=c*S){
			(*cluster_centers+k)->l = *(cielab_img+i*w*c + j); // c*S
			(*cluster_centers+k)->a = *(cielab_img+i*w*c + j+1); // c*S
			(*cluster_centers+k)->b = *(cielab_img+i*w*c + j+2); //c*S
			(*cluster_centers+k)->x = i;
			(*cluster_centers+k)->y = j/c;
			++k;
		}
	}
	printf("Superpixel size: %d\n", sizeOfS);
	printf("S = %d\n", S);
	printf("Requested K = %d, real k = %d\n", K, k);
	*cluster_centers = reallocarray(*cluster_centers, k, sizeof(struct cluster));
	// Gradient descent on cluster centers
	float grad;
	float new_grad;
	for(int i = 0; i < k; ++i){
		grad = gradient(cielab_img, w, h, c, (*cluster_centers+i)->x, (*cluster_centers+i)->y);
		int new_x, new_y;

		for(int dh = -neighborhood; dh <= neighborhood; ++dh){
			for(int dw = -neighborhood; dw <= neighborhood; ++dw){
				new_x = (*cluster_centers+i)->x + dw;
				new_y = (*cluster_centers+i)->y + dh;

				new_grad = gradient(cielab_img, w, h, c, new_x, new_y);
				if(new_grad < grad){
					(*cluster_centers+i)->l = *(cielab_img+new_x*w*c + new_y*c);
					(*cluster_centers+i)->a	= *(cielab_img+new_x*w*c + new_y*c+1);
					(*cluster_centers+i)->b = *(cielab_img+new_x*w*c + new_y*c+2);
					(*cluster_centers+i)->x = new_x;
					(*cluster_centers+i)->y = new_y;
					grad = new_grad;
				}
			}
		}
	}

	// Compute distances in a 2S x 2S grid around cluster centers
	float error = 0;
	float umbral = 1e-4;
	struct cluster* previous_centers = malloc(k * sizeof(struct cluster));
	previous_centers = memcpy(previous_centers, *cluster_centers, k*sizeof(struct cluster));
	int pixel_count[k];
	int new_x[k];
	int new_y[k];
	do{
		for(int i = 0; i < k; ++i){
			// Assign the best matching pixels from a 2Sx2S square neighborhood around
			// the cluster center according to the distance measure
			for(int j = -S; j <= S; ++j){
				for(int l = -S; l <= S; ++l){
					// compute D between Ck and i
					int x_pix = (*cluster_centers+i)->x+j;
					int y_pix = (*cluster_centers+i)->y+l;
					if(x_pix < 0 || x_pix >= h)continue;
					if(y_pix < 0 || y_pix >= w)continue;
					float D = Distance_D(*cluster_centers+i, cielab_img+(((x_pix)*w*c) + y_pix*c), x_pix, y_pix, S, m);
					if(D < *(distances+(x_pix*w + y_pix))){
						*(distances+(x_pix*w) + y_pix) = D;
						*(labels+(x_pix*w) + y_pix) = i;
					}
				}
			}
		}
		// Compute new cluster centers and residual error(L1 distance between previous
		// centers and recomputed centers)
		memset(pixel_count, 0, k*sizeof(int));
		memset(new_x, 0, k*sizeof(int));
		memset(new_y, 0, k*sizeof(int));

		for(int i = 0; i < h; ++i){
			for(int j = 0; j < w; ++j){
				int label_idx = *(labels+i*w + j);
				if(label_idx == -1) continue;
				new_x[label_idx] += i;
				new_y[label_idx] += j;
				pixel_count[label_idx] += 1;
			}
		}
		for(int i = 0; i < k; ++i){
			if(pixel_count[i] == 0) continue;
			new_x[i] /= pixel_count[i];
			new_y[i] /= pixel_count[i];
			(*cluster_centers+i)->l = *(cielab_img+new_x[i]*w*c + new_y[i]*c);
			(*cluster_centers+i)->a = *(cielab_img+new_x[i]*w*c + new_y[i]*c+1);
			(*cluster_centers+i)->b = *(cielab_img+new_x[i]*w*c + new_y[i]*c+2);
			(*cluster_centers+i)->x = new_x[i];
			(*cluster_centers+i)->y = new_y[i];
			error += abs((*cluster_centers+i)->x - (previous_centers+i)->x) + abs((*cluster_centers+i)->y - (previous_centers+i)->y);
		}
		error /= k;
		previous_centers = memcpy(previous_centers, *cluster_centers, k*sizeof(struct cluster));
		
		printf("error: %f\n", error);

	}while(error > umbral);
	// TODO: Enforce connectivity
	
	// Generate final image
	float* out_img = malloc(w*h*c*sizeof(float));
	if(out_img != NULL){
		for(int i = 0; i < h; ++i){
			for(int j = 0; j < w; ++j){
				int label_idx = *(labels+i*w + j);
				*(out_img+i*w*c + j*c)		= (*cluster_centers+label_idx)->l; 
				*(out_img+i*w*c + j*c+1)	= (*cluster_centers+label_idx)->a; 
				*(out_img+i*w*c + j*c+2)	= (*cluster_centers+label_idx)->b; 
			}
		}
	}
	
	
	*sizeC = k;
	free(labels);
	labels = NULL;
	free(distances);
	distances = NULL;
	free(previous_centers);
	previous_centers = NULL;
	return out_img;
}

void usage(char* program, int k){
	fprintf(stderr, "Usage: %s [-k number of Superpixels] imagepath\nDefault k = %d\n", program, k);
}

int main(int argc, char** argv){
	
	int opt, k = 500;
	char* image_path;
	while((opt = getopt(argc, argv, "k:")) != -1){
		switch(opt){
			case 'k':
				k = atoi(optarg);
				break;
			default:
				usage(argv[0], k);
				exit(EXIT_FAILURE);
		}
	}
	if(optind >= argc){
		fprintf(stderr, "Expected arguments after options\n");
		usage(argv[0], k);
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
	printf("Image dimensions: %d x %d %d\n", x_dim, y_dim, n_channels);

	int sizeC;
	struct cluster* centers;

	float* segmented = SLIC(cielab, x_dim, y_dim, n_channels, k, 10.0, &centers, &sizeC);
	if(segmented == NULL){
		fprintf(stderr, "Error allocating memory\n");
		exit(EXIT_FAILURE);
	}

	unsigned char* segmented_rgb = cielab_to_rgb(segmented, x_dim, y_dim, n_channels);

	int result_seg = stbi_write_jpg("segmented.jpeg", x_dim, y_dim, n_channels, segmented_rgb, 100);

#ifdef DEBUG	
	for(int i = 0; i < sizeC; ++i){
		paintPixel(image_data, x_dim, y_dim, n_channels, (centers+i)->x, (centers+i)->y, 255, 0, 0);
	}
	int result = stbi_write_jpg("segmented_mod.jpeg", x_dim, y_dim, n_channels, image_data, 100);
#endif
	
	

	free(centers);
	centers = NULL;
	free(segmented);
	segmented = NULL;
	free(segmented_rgb);
	segmented_rgb = NULL;

	stbi_image_free(image_data);
	free(cielab);
	cielab = NULL;
	image_data = NULL;

	exit(EXIT_SUCCESS);
}
