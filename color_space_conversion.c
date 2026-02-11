#include"color_space_conversion.h"

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

/*
 *	Returned RGB is interleaved, R G B. In that order.
 *
 */

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
			*r = round(255 * srgb_companding(X * 3.2404542 + Y * -1.5371385 + Z * -0.4985314));
			*g = round(255 * srgb_companding(X * -0.9692660 + Y * 1.8760108 + Z * 0.0415560));
			*b = round(255 * srgb_companding(X * 0.0556434 + Y * -0.2040259 + Z * 1.0572252));

		}
	}

	return rgb_img;
}

