
extern "C"{
	#include <stdio.h>
	#include <stdlib.h>

	void writeBinFile(const char * file_name, float * bin_data, size_t nmemb){
		FILE * fp = fopen(file_name, "w");
		fwrite(bin_data, sizeof(*bin_data), nmemb, fp);
		fclose(fp);
	}
	
	size_t readBinFile(const char * file_name, float ** bin_data){
		FILE * fp = fopen(file_name,"r");
		fseek(fp, 0, SEEK_END);
		size_t sz = ftell(fp);
		size_t nmemb = sz/sizeof(**bin_data);
		//printf("file is %lu elements in %lu bytes\n", nmemb, sz);
		rewind(fp);
		*bin_data = malloc(sz);
		fread(*bin_data, sizeof(**bin_data), nmemb, fp);
		fclose(fp);
		return nmemb;
	}
	
	int main(int argc, char ** argv){
		if(argc < 2){
			printf("need two names of binary files, one to read, one to write");
			return 1;
		}
		float * bin_data;
		size_t nmemb = readBinFile(argv[1], &bin_data);
		writeBinFile(argv[2], bin_data, nmemb);
		free(bin_data);
		return 0;
	}
}