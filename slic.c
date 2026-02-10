#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

/*
 * Reference for colorspace conversions
 * http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
 *
 */

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
		return (903.3 * t + 16.0)/116.0;
		//return 7.787 * t + (16.0/116.0);
}

/*
 * Returned cielab is interleaved, L a b. In that order
 *
 */

float* rgb_to_cielab(unsigned char* rgb, int w, int h, int c){
	float* cielab_color = malloc(w*h*3*sizeof(float));
	if(cielab_color == NULL) return NULL;
	for(int i = 0; i < h; ++i){
		for(int j = 0; j < w*c; j+=c){
			unsigned char r_o = *(rgb+i*w*c + j);
			unsigned char g_o = *(rgb+i*w*c + j+1);
			unsigned char b_o = *(rgb+i*w*c + j+2);

			float* L = (cielab_color+i*w*c + j);
			float* a = (cielab_color+i*w*c + j+1);
			float* b = (cielab_color+i*w*c + j+2);

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

float srgb_companding(float val){
	if(val <= 0.0031308)
		return 12.92*val;
	else
		return 1.055*powf(val, 1.0/2.4) - 0.055;
}


unsigned char* cielab_to_rgb(float* cielab_img, int w, int h, int c){
	unsigned char* rgb_img = malloc(w*h*3*sizeof(unsigned char));
	if(rgb_img == NULL) return NULL;
	for(int i = 0; i < h; ++i){
		for(int j = 0; j < w*c; j+=c){
			float l_o = *(cielab_img+i*w*c + j);
			float a_o = *(cielab_img+i*w*c + j+1);
			float b_o = *(cielab_img+i*w*c + j+2);

			unsigned char* r = (rgb_img+i*w*c + j);
			unsigned char* g = (rgb_img+i*w*c + j+1);
			unsigned char* b = (rgb_img+i*w*c + j+2);

			// Convert to XYZ color space
			float fy = (l_o + 16)/116.0;
			float fz = fy - (b_o/200.0);
			float fx = (a_o/500.0) + fy;

			float xr, yr, zr;
			if((fx*fx*fx) > 0.008856){
				xr = fx*fx*fx;
			}
			else{
				xr = (116.0*fx - 16)/903.3;
			}

			if(l_o > (903.3*0.008856)){
				yr = ((l_o + 16)/116.0);
				yr = yr * yr * yr;
			}
			else{
				yr = l_o/903.3;
			}

			if((fz*fz*fz) > 0.008856){
				zr = fz*fz*fz;
			}
			else{
				zr = (116.0*fz - 16)/903.3;
			}
			
			float X = xr*Xn;
			float Y = yr*Yn;
			float Z = zr*Zn;

			// Convert to RGB
			*r = ceil(255 * srgb_companding(X * 3.2404542 + Y * -1.5371385 + Z * -0.4985314));
			*g = ceil(255 * srgb_companding(X * -0.9692660 + Y * 1.8760108 + Z * 0.0415560));
			*b = ceil(255 * srgb_companding(X * 0.0556434 + Y * -0.2040259 + Z * 1.0572252));


		}
	}

	return rgb_img;
}


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
	int S = round(sqrt((w*h)/K));
	int sizeOfS = round((w*h)/K);
	int neighborhood = 1;
	*cluster_centers = malloc(K * sizeof(struct cluster));
	int* labels = malloc(w*h*sizeof(int));
	float* distances = malloc(w*h*sizeof(float));
	if(*cluster_centers == NULL || labels == NULL || distances == NULL){
		return NULL;
	}
	for(int i = 0; i < w*h; ++i){labels[i] = -1;} 
	//memset(labels, -1, w*h);
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
	float grad = INFINITY;
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
		// cnters and recomputed centers)

		int pixel_count[k];
		memset(pixel_count, 0, k*sizeof(int));
		int new_x[k];
		int new_y[k];
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
		//error = 0;
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
	//out_img = memcpy(out_img, cielab_img, w*h*c*sizeof(float));
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
	//printf("rgb values at pixel (0,0) => (%d,%d,%d)\n", image_data[0], image_data[1], image_data[2]);
	//printf("CIELab values at pixel (0,0) => (%f,%f,%f)\n", cielab[0], cielab[1], cielab[2]);

	/*printf("testing cielab to rgb function...\n");
	unsigned char* img_test = cielab_to_rgb(cielab, x_dim, y_dim, n_channels);
	int res_test = stbi_write_jpg("gato_recuperado.jpeg", x_dim, y_dim, n_channels, img_test, 100);
	free(img_test);
	img_test = NULL;
	*/

	// Allocate inside SLIC
	//struct cluster* centers = malloc(1444 * sizeof(struct cluster));
	int sizeC;
	struct cluster* centers;

	float* segmented = SLIC(cielab, x_dim, y_dim, n_channels, k, 10.0, &centers, &sizeC);
	if(segmented == NULL){
		fprintf(stderr, "Error allocating memory\n");
		exit(EXIT_FAILURE);
	}
	//printf("Segmented CIELab values at pixel (0,0) => (%f,%f,%f)\n", segmented[0], segmented[1], segmented[2]);

	unsigned char* segmented_rgb = cielab_to_rgb(segmented, x_dim, y_dim, n_channels);
	//printf("Segmented RGB values at pixel (0,0) => (%d,%d,%d)\n", segmented_rgb[0], segmented_rgb[1], segmented_rgb[2]);

	int result_seg = stbi_write_jpg("segmentado.jpeg", x_dim, y_dim, n_channels, segmented_rgb, 100);

	
	/*for(int i = 0; i < sizeC; ++i){
		paintPixel(image_data, x_dim, y_dim, n_channels, (centers+i)->x, (centers+i)->y, 255, 0, 0);
	}
	int result = stbi_write_jpg("gato_mod.jpeg", x_dim, y_dim, n_channels, image_data, 100);
	*/
	

	free(centers);
	centers = NULL;
	free(segmented);
	segmented = NULL;
	free(segmented_rgb);
	segmented_rgb = NULL;


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
