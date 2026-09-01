#include <stdio.h>
#include <unistd.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


#include "slic.h"

void usage(char* program, int k, int i){
	fprintf(stderr, "Usage: %s [-k number of Superpixels] [-i maximum number of iterations] imagepath\nDefault k = %d\nDefault i = %d\n", program, k, i);
}

int main(int argc, char** argv){
	
	int opt, k = 500;
	char* image_path;
    int max_iters = 50;
	while((opt = getopt(argc, argv, "i:k:")) != -1){
		switch(opt){
			case 'k':
				k = atoi(optarg);
				break;
            case 'i':
                max_iters = atoi(optarg);
                break;
			default:
				usage(argv[0], k, max_iters);
				exit(EXIT_FAILURE);
		}
	}
	if(optind >= argc){
		fprintf(stderr, "Expected arguments after options\n");
		usage(argv[0], k, max_iters);
		exit(EXIT_FAILURE);
	}
    if(k == 0 || max_iters == 0){
       fprintf(stderr, "Error reading arguments\n");
       usage(argv[0], k, max_iters);
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

	float* segmented = SLIC(cielab, x_dim, y_dim, n_channels, k, 10.0, &centers, &sizeC, max_iters);
	if(segmented == NULL){
		fprintf(stderr, "Error allocating memory\n");
		exit(EXIT_FAILURE);
	}

	unsigned char* segmented_rgb = cielab_to_rgb(segmented, x_dim, y_dim, n_channels);

#ifndef DEBUG
	int result_seg = stbi_write_jpg("segmented.jpeg", x_dim, y_dim, n_channels, segmented_rgb, 100);
#endif
    printf("Done...\n");

#ifdef DEBUG	
	for(int i = 0; i < sizeC; ++i){
		paintPixel(segmented_rgb, x_dim, y_dim, n_channels, (centers+i)->x, (centers+i)->y, 255, 0, 0);
	}
	int result = stbi_write_jpg("segmented_mod.jpeg", x_dim, y_dim, n_channels, segmented_rgb, 100);
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

