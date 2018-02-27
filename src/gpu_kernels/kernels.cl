#define TILE_SIZE 32

__kernel void hello(__global char* string)
{
string[0] = 'H';
string[1] = 'e';
string[2] = 'l';
string[3] = 'l';
string[4] = 'o';
string[5] = ',';
string[6] = ' ';
string[7] = 'W';
string[8] = 'o';
string[9] = 'r';
string[10] = 'l';
string[11] = 'd';
string[12] = '!';
string[13] = '\0';
}

__kernel void test(const int width,
	 	  		   const int height,
				   __global float * input_image,
			   	   __global float * output_image)	{

	int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));

	if (pixelcoord.x < width && pixelcoord.y < height)
	{
	    float z = input_image[pixelcoord.x * height + pixelcoord.y];
		output_image[pixelcoord.x * height + pixelcoord.y] = z;
	}
} 

__kernel void predict2(const int width,
		 	  		   const int height,
					   __global unsigned int * left_child_arr,
		 	 	  	   __global unsigned int * right_child_arr,
				  	   __global float * offset1_x_arr,
				  	   __global float * offset1_y_arr,
				  	   __global float * offset2_x_arr,
				  	   __global float * offset2_y_arr,
				  	   __global float * theshold_arr,
					   __global float * probability_arr,
				  	   __global unsigned char * is_unary_arr,
					   __global unsigned char * is_leaf_arr,
				  	   __global unsigned char * label_arr,
				  	   __global float * input_image,
			   	  	   __global unsigned char * output_image)	{

	int col = get_global_id(0);
	int row = get_global_id(1);

	if (col < width && row < height)
	{
	    float src_z = input_image[row * width + col];
		
		// check that the pixel is foreground (10 meters is background)
		if (src_z == 10.0f) {
			output_image[row * width + col] = 0;
			return;
		}
		int idx = 0;		

		while(true) {		   

		    float z = src_z;
			
		    // evaluate feature
			///////////////////////////////////////////////////
			float value;
			int u = row + (int)(offset1_x_arr[idx] / z);
			int v = col + (int)(offset1_y_arr[idx] / z);

			if (u < 0 || v < 0 || u >= height || v >= width) {
			    value = 10.0;
			}
			else {
				value = input_image[u * width + v];
			}

			if (!is_unary_arr[idx]) {
			    u = row + (int)(offset2_x_arr[idx] / z);
			    v = col + (int)(offset2_y_arr[idx] / z);	
				if (u < 0 || v < 0 || u >= height || v >= width) {
			       	z = 10.0;
				}
				else {
					z = input_image[u * width + v];
			    }
			}

			value -= z;
			//////////////////////////////////////////////////			

			if (value < theshold_arr[idx]) { // falls to left
			    idx = left_child_arr[idx];
			}
			else { // falls to right
			    idx = right_child_arr[idx];
			}

			if (is_leaf_arr[idx] == 1) {
				if (probability_arr[idx] > 0.6)
				{
					output_image[row * width + col] = label_arr[idx];
				}
				else {
					output_image[row * width + col] = 0;
				}
				return;
			}
		}
	}

}

__kernel void predict(__global unsigned int * left_child_arr,
		 	  		  __global unsigned int * right_child_arr,
					  __global short * offset1_x_arr,
					  __global short * offset1_y_arr,
  					  __global short * offset2_x_arr,
					  __global short * offset2_y_arr,
					  __global float * theshold_arr,
					  __global unsigned char * is_unary_arr,
					  __global unsigned char * label_arr,
					  __read_only image2d_t input_image,
					  __write_only image2d_t output_image)
{

	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));
	int width = get_image_width(input_image);
	int height = get_image_height(input_image);

	if (pixelcoord.x < width && pixelcoord.y < height)
	{

		// uint4 color = ((uint)255, (uint)0, (uint)0, (uint)255);				
  		// write_imageui(output_image, pixelcoord, color);
		// return;

		float4 pixel = read_imagef(input_image, sampler, (int2)(pixelcoord.x, pixelcoord.y));
   		float z = pixel.x;

		uint4 color = ((uint)z, (uint)0, (uint)0, (uint)255);
		write_imageui(output_image, pixelcoord, color);
		return;

 		// check that the pixel is foreground (10 meters is background)
		if (z == 10.0f) return;
		int idx = 0;
		
		while(true) {		    

			if (idx == 0) { // is leaf node
			   	unsigned char label = label_arr[idx];
				uint4 color = ((uint)label, (uint)label, (uint)label, (uint)label);				
				write_imageui(output_image, pixelcoord, color);
				return;
			}
			
		    // evaluate feature
			///////////////////////////////////////////////////
			float value;
			int u = pixelcoord.x + offset1_x_arr[idx]/z;
			int v = pixelcoord.y + offset1_y_arr[idx]/z;

			if (u >= width || v >= height) {
			    value = 10.0;
			}
			else {
				pixel = read_imagef(input_image, sampler, (int2)(u, v));
				value = pixel.x;
			}

			if (!is_unary_arr[idx]) {
				float value2;	
			    u = pixelcoord.x + offset2_x_arr[idx]/z;
			    v = pixelcoord.y + offset2_y_arr[idx]/z;	
				if (u >= width || v >= height) {
			       z = 10.0;
				}
				else {
				   pixel = read_imagef(input_image, sampler, (int2)(u, v));
				   z = pixel.x;
			    }
			}

			value -= z;
			//////////////////////////////////////////////////

			if (value < theshold_arr[idx]) { // falls to left
			    idx = left_child_arr[idx];
			}
			else { // falls to right
			    idx = right_child_arr[idx];
			}
		}	
		
	}
}
