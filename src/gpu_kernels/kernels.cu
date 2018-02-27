#define TILE_SIZE 32

__kernel void predict(__global uint32_t * left_child_arr,
		 	  		  __global uint32_t * right_child_arr,
					  __global int16_t * offset1_x_arr,
					  __global int16_t * offset1_y_arr,
  					  __global int16_t * offset2_x_arr,
					  __global int16_t * offset2_y_arr,
					  __global float * theshold_arr,
					  __global uint8_t * is_unary_arr,
					  __global uint8_t * label_arr,
					  __read_only image2d_t input_image,
					  __write_only image2d_t output_image)
{

	// const sampler_t sampler=CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	// int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));
	// int width = get_image_width(input_image);
	// int height = get_image_height(input_image);

	// if (pixelcoord.x < width && pixelcoord.y < height) {

	// 	float z = read_imagef(input_image, sampler, (int2)(pixelcoord.x, pixelcoord.y));

 	// 	// check that the pixel is foreground (10 meters is background)
	// 	if (z == 10.0f) return;
	// 	int idx = 1;
		
	// 	while(true) {		    

	// 		if (idx == 0) { // is leaf node
	// 		   	uint8_t label = label_arr[idx];
	// 			write_imagef(output_image, pixelcoord, label);
	// 			return;
	// 		}
			
	// 	    // evaluate feature
	// 		///////////////////////////////////////////////////
	// 		float value;
	// 		int u = pixelcoord.x + int(offset1_x_arr[idx]/z);
	// 		int v = pixelcoord.y + int(offset1_y_arr[idx]/z);	

	// 		if (u >= width || v >= height) {
	// 		    value = 10.0;
	// 		}
	// 		else {
	// 			value = read_imagef(input_image, sampler, (int2)(u, v));
	// 		}

	// 		if (!is_unary_arr[idx]) {
	// 			float value2;	
	// 		    u = pixelcoord.x + int(offset2_x_arr[idx]/z);
	// 		    v = pixelcoord.y + int(offset2_y_arr[idx]/z);	
	// 			if (u >= width || v >= height) {
	// 		       z = 10.0;
	// 			}
	// 			else {
	// 			   z = read_imagef(input_image, sampler, (int2)(u, v));				   
	// 		    }
	// 		}

	// 		value -= z;
	// 		//////////////////////////////////////////////////

	// 		if (value < theshold_arr[idx]) { // falls to left
	// 		    idx = left_child_arr[idx];
	// 		}
	// 		else { // falls to right
	// 		    idx = right_child_arr[idx];
	// 		}
	// 	}		
	// }

}
