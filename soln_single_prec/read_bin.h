#ifndef READ_BIN_H
#define READ_BIN_H

#pragma once
  
#include <stdio.h>
#include <stdlib.h>

void writeBinFile(const char * file_name, float * bin_data, size_t nmemb);
size_t readBinFile(const char * file_name, float ** bin_data);

#endif