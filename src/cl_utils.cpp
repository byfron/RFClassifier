/*
 * Copyright © MindMaze Holding SA 2017 - All Rights Reserved.
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * CONFIDENTIAL: This project is proprietary and confidential. It cannot be
 * copied and/or distributed without the express permission of MindMaze
 * Holding SA.
 */
#include "cl_utils.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>

#define MAX_GROUPS      (64)
#define MAX_WORK_ITEMS  (64)

char *file_contents(const char *filename, int *length)
{
	FILE *f = fopen(filename, "r");
	void *buffer;
	if (!f) {
		fprintf(stderr, "Unable to open %s for reading\n", filename);
		return NULL;
	}
	fseek(f, 0, SEEK_END);
	*length = ftell(f);
	fseek(f, 0, SEEK_SET);
	buffer = malloc(*length+1);
	*length = fread(buffer, 1, *length, f);
	fclose(f);
	((char*)buffer)[*length] = '\0';
	return (char*)buffer;
}


// Helper function to get error string
// *********************************************************************
const char* oclErrorString(cl_int error)
{
	static const char* errorString[] = {
		"CL_SUCCESS",
		"CL_DEVICE_NOT_FOUND",
		"CL_DEVICE_NOT_AVAILABLE",
		"CL_COMPILER_NOT_AVAILABLE",
		"CL_MEM_OBJECT_ALLOCATION_FAILURE",
		"CL_OUT_OF_RESOURCES",
		"CL_OUT_OF_HOST_MEMORY",
		"CL_PROFILING_INFO_NOT_AVAILABLE",
		"CL_MEM_COPY_OVERLAP",
		"CL_IMAGE_FORMAT_MISMATCH",
		"CL_IMAGE_FORMAT_NOT_SUPPORTED",
		"CL_BUILD_PROGRAM_FAILURE",
		"CL_MAP_FAILURE",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"CL_INVALID_VALUE",
		"CL_INVALID_DEVICE_TYPE",
		"CL_INVALID_PLATFORM",
		"CL_INVALID_DEVICE",
		"CL_INVALID_CONTEXT",
		"CL_INVALID_QUEUE_PROPERTIES",
		"CL_INVALID_COMMAND_QUEUE",
		"CL_INVALID_HOST_PTR",
		"CL_INVALID_MEM_OBJECT",
		"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
		"CL_INVALID_IMAGE_SIZE",
		"CL_INVALID_SAMPLER",
		"CL_INVALID_BINARY",
		"CL_INVALID_BUILD_OPTIONS",
		"CL_INVALID_PROGRAM",
		"CL_INVALID_PROGRAM_EXECUTABLE",
		"CL_INVALID_KERNEL_NAME",
		"CL_INVALID_KERNEL_DEFINITION",
		"CL_INVALID_KERNEL",
		"CL_INVALID_ARG_INDEX",
		"CL_INVALID_ARG_VALUE",
		"CL_INVALID_ARG_SIZE",
		"CL_INVALID_KERNEL_ARGS",
		"CL_INVALID_WORK_DIMENSION",
		"CL_INVALID_WORK_GROUP_SIZE",
		"CL_INVALID_WORK_ITEM_SIZE",
		"CL_INVALID_GLOBAL_OFFSET",
		"CL_INVALID_EVENT_WAIT_LIST",
		"CL_INVALID_EVENT",
		"CL_INVALID_OPERATION",
		"CL_INVALID_GL_OBJECT",
		"CL_INVALID_BUFFER_SIZE",
		"CL_INVALID_MIP_LEVEL",
		"CL_INVALID_GLOBAL_WORK_SIZE",
	};
	const int errorCount = sizeof(errorString) / sizeof(errorString[0]);
	const int index = -error;
	return (index >= 0 && index < errorCount) ? errorString[index] : "";
}


void create_reduction_pass_counts(
	int count, 
	int max_group_size,    
	int max_groups,
	int max_work_items, 
	int *pass_count, 
	size_t **group_counts, 
	size_t **work_item_counts,
	int **operation_counts,
	int **entry_counts)
{
	int work_items = (count < max_work_items * 2) ? count / 2 : max_work_items;
	if(count < 1)
		work_items = 1;
        
	int groups = count / (work_items * 2);
	groups = max_groups < groups ? max_groups : groups;

	int max_levels = 1;
	int s = groups;

	while(s > 1) 
	{
		int work_items = (s < max_work_items * 2) ? s / 2 : max_work_items;
		s = s / (work_items*2);
		max_levels++;
	}
 
	*group_counts = (size_t*)malloc(max_levels * sizeof(size_t));
	*work_item_counts = (size_t*)malloc(max_levels * sizeof(size_t));
	*operation_counts = (int*)malloc(max_levels * sizeof(int));
	*entry_counts = (int*)malloc(max_levels * sizeof(int));

	(*pass_count) = max_levels;
	(*group_counts)[0] = groups;
	(*work_item_counts)[0] = work_items;
	(*operation_counts)[0] = 1;
	(*entry_counts)[0] = count;
	if(max_group_size < work_items)
	{
		(*operation_counts)[0] = work_items;
		(*work_item_counts)[0] = max_group_size;
	}
    
	s = groups;
	int level = 1;
   
	while(s > 1) 
	{
		int work_items = (s < max_work_items * 2) ? s / 2 : max_work_items;
		int groups = s / (work_items * 2);
		groups = (max_groups < groups) ? max_groups : groups;

		(*group_counts)[level] = groups;
		(*work_item_counts)[level] = work_items;
		(*operation_counts)[level] = 1;
		(*entry_counts)[level] = s;
		if(max_group_size < work_items)
		{
			(*operation_counts)[level] = work_items;
			(*work_item_counts)[level] = max_group_size;
		}
        
		s = s / (work_items*2);
		level++;
	}
}
