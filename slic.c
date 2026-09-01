#include "slic.h"

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
		float term_y = (*(img_cielab+(y+1)*w*c + x*c+i) - *(img_cielab+(y-1)*w*c + x*c+i));
		float term_x = (*(img_cielab+y*w*c + (x+1)*c+i) - *(img_cielab+y*w*c + (x-1)*c+i));
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
	Ds = dlab + (m*dxy)/S;
	return Ds;
}

float* SLIC(float* cielab_img, int w, int h, int c, int K, float m, struct cluster** centers, int* sizeC, int max_iters){
    if(K > (w*h))
        return NULL;
	int S = sqrt(w*h/K);
	int sizeOfS = w*h/K;
	int neighborhood = 1;
	*centers = malloc(1.5*K * sizeof(struct cluster));
    struct cluster* cluster_centers = *centers;
	int* labels = malloc(w*h*sizeof(int));
	float* distances = malloc(w*h*sizeof(float));
	if(cluster_centers == NULL || labels == NULL || distances == NULL){
		free(cluster_centers);
		free(labels);
		free(distances);
		cluster_centers = NULL;
		labels = NULL;
		distances = NULL;
		return NULL;
	}
	for(int i = 0; i < w*h; ++i){labels[i] = -1;} 
	for(int i = 0; i < w*h; ++i){distances[i] = INFINITY;}
	int k = 0; // Number of real cluster centers
	for(int i = S; i < h; i+=S){
		for(int j = c*S; j < w*c; j+=c*S){
			(cluster_centers+k)->l = *(cielab_img+i*w*c + j); // c*S
			(cluster_centers+k)->a = *(cielab_img+i*w*c + j+1); // c*S
			(cluster_centers+k)->b = *(cielab_img+i*w*c + j+2); //c*S
			(cluster_centers+k)->y = i;
			(cluster_centers+k)->x = j/c;
			++k;
		}
	}
	printf("Superpixel size: %d\n", sizeOfS);
	printf("S (grid separation) = %d\n", S);
	printf("Requested K = %d, real k = %d\n", K, k);
	struct cluster* tmp = reallocarray(cluster_centers, k, sizeof(struct cluster));
    if(!tmp)cluster_centers = tmp;
	// Gradient descent on cluster centers
	float grad;
	float new_grad;
	for(int i = 0; i < k; ++i){
		grad = gradient(cielab_img, w, h, c, (cluster_centers+i)->x, (cluster_centers+i)->y);
		int new_x, new_y;

		for(int dh = -neighborhood; dh <= neighborhood; ++dh){
            new_y = (cluster_centers+i)->y + dh;
			for(int dw = -neighborhood; dw <= neighborhood; ++dw){
				new_x = (cluster_centers+i)->x + dw;

				new_grad = gradient(cielab_img, w, h, c, new_x, new_y);
				if(new_grad < grad){
					(cluster_centers+i)->l = *(cielab_img+new_y*w*c + new_x*c);
					(cluster_centers+i)->a = *(cielab_img+new_y*w*c + new_x*c+1);
					(cluster_centers+i)->b = *(cielab_img+new_y*w*c + new_x*c+2);
					(cluster_centers+i)->x = new_x;
					(cluster_centers+i)->y = new_y;
					grad = new_grad;
                }
			}
		}
	}

	// Compute distances in a 2S x 2S grid around cluster centers
	float error = 0;
	float threshold = 1e-4;
	//struct cluster* previous_centers = malloc(k * sizeof(struct cluster));
	//previous_centers = memcpy(previous_centers, cluster_centers, k*sizeof(struct cluster));
	int pixel_count[k];
	int new_x[k];
	int new_y[k];
	int iter = 0;
	do{
		for(int i = 0; i < k; ++i){
			// Assign the best matching pixels from a 2Sx2S square neighborhood around
			// the cluster center according to the distance measure
			for(int j = -S; j <= S; ++j){
				for(int l = -S; l <= S; ++l){
					// compute D between Ck and i
					int x_pix = (cluster_centers+i)->x+l;
					int y_pix = (cluster_centers+i)->y+j;
					if(x_pix < 0 || x_pix >= w)continue;
					if(y_pix < 0 || y_pix >= h)continue;
					float D = Distance_D(cluster_centers+i, cielab_img+(((y_pix)*w*c) + x_pix*c), x_pix, y_pix, S, m);
					if(D < *(distances+(y_pix*w + x_pix))){
						*(distances+(y_pix*w) + x_pix) = D;
						*(labels+(y_pix*w) + x_pix) = i;
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
				if(label_idx == -1){
                    printf("Ophaned pixel at x= %d, y = %d\n", j, i);
                    continue;
                } 
				new_x[label_idx] += j;
				new_y[label_idx] += i;
				pixel_count[label_idx] += 1;
			}
		}
		for(int i = 0; i < k; ++i){
			if(pixel_count[i] == 0) 
                continue;
            
			new_x[i] /= pixel_count[i];
			new_y[i] /= pixel_count[i];
			(cluster_centers+i)->l = *(cielab_img+new_y[i]*w*c + new_x[i]*c);
			(cluster_centers+i)->a = *(cielab_img+new_y[i]*w*c + new_x[i]*c+1);
			(cluster_centers+i)->b = *(cielab_img+new_y[i]*w*c + new_x[i]*c+2);
            
            int errx = (cluster_centers+i)->x - new_x[i];
            int erry = (cluster_centers+i)->y - new_y[i];
            error += sqrt(errx*errx + erry*erry);
            
            //error += abs((cluster_centers+i)->x - new_x[i]) + abs((cluster_centers+i)->y - new_y[i]);
			(cluster_centers+i)->x = new_x[i];
			(cluster_centers+i)->y = new_y[i];
			//error += abs((cluster_centers+i)->x - (previous_centers+i)->x) + abs((cluster_centers+i)->y - (previous_centers+i)->y);
		}
		error /= k;
		//previous_centers = memcpy(previous_centers, cluster_centers, k*sizeof(struct cluster));
		
		printf("Iteration: %d, error: %f\n", ++iter, error);

	}while(error > threshold && iter < max_iters);
	// TODO: Enforce connectivity
	
	// Generate final image
	float* out_img = malloc(w*h*c*sizeof(float));
	if(out_img != NULL){
		for(int i = 0; i < h; ++i){
			for(int j = 0; j < w; ++j){
				int label_idx = *(labels+i*w + j);
				*(out_img+i*w*c + j*c)		= (cluster_centers+label_idx)->l; 
				*(out_img+i*w*c + j*c+1)	= (cluster_centers+label_idx)->a; 
				*(out_img+i*w*c + j*c+2)	= (cluster_centers+label_idx)->b; 
			}
		}
	}

	
	*sizeC = k;
	free(labels);
	labels = NULL;
	free(distances);
	distances = NULL;
	//free(previous_centers);
	//previous_centers = NULL;
	return out_img;
}

