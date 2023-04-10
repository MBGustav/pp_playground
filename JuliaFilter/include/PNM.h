
#include <iostream>
#include <fstream>

void write_pnm(const char* filename, unsigned char* image_data, int dim) {
    // Open file for writing in binary mode
    std::ofstream pnm_file(filename,std::ios_base::binary);

    if (!pnm_file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }

    // PNM file header
    pnm_file << "P6\n" << dim << " " << dim << "\n255\n";

    pnm_file.write(reinterpret_cast<const char*>(image_data), dim*dim*3);
    // Close file
    pnm_file.close();

    std::cout << "Image saved: " << filename << std::endl;
}