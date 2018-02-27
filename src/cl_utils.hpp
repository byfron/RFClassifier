/*
 * Copyright © MindMaze Holding SA 2017 - All Rights Reserved.
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * CONFIDENTIAL: This project is proprietary and confidential. It cannot be
 * copied and/or distributed without the express permission of MindMaze
 * Holding SA.
 */
#pragma once
#include <CL/cl.hpp>

char *file_contents(const char *filename, int *length);
const char* oclErrorString(cl_int error);
