#pragma once
#include <iostream>
#include <fstream>
#include <vector>

std::vector<float> read_fvecs(const std::string& filename, int& dimension, int& num_vectors, int a = 1, int b = -1) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // First: peek the dim from the first vector
    int d = 0;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));
    if (d <= 0 || d > 10000) throw std::runtime_error("Invalid dim in header");
    dimension = d;
    file.seekg(0, std::ios::beg);
    const int vec_size = sizeof(int) + d * sizeof(float);
    std::cout << "Reading from vector " << a << " to " << b << std::endl;
    std::cout << "Seekg position: " << (a - 1) * vec_size << " bytes" << std::endl;
    std::cout << "Each vector size: " << vec_size << " bytes (dim=" << d << ")" << std::endl;
    // Determine total number of vectors
    file.seekg(0, std::ios::end);
    int total_vectors = file.tellg() / vec_size;
    if (b == -1) b = total_vectors;
    if (a < 1) a = 1;
    if (b > total_vectors) b = total_vectors;
    if (b < a) return {};

    int n = b - a + 1;
    num_vectors = n;

    // Seek to aligned position
    file.seekg(static_cast<std::streampos>((a - 1) * vec_size), std::ios::beg);

    std::vector<float> result(static_cast<size_t>(n) * d);
    std::cout << "Reading " << n << " vectors" << std::endl;
    for (int i = 0; i < n; ++i) {
        int current_dim;
        file.read(reinterpret_cast<char*>(&current_dim), sizeof(int));
        if (current_dim != d) {
            std::cerr << "Mismatch at vector " << (i + a)
              << ", expected dim: " << d << ", but got: " << current_dim
              << ", file tellg: " << file.tellg() << std::endl; 
            throw std::runtime_error("Dimension mismatch in vector " + std::to_string(i + a));
        }
        file.read(reinterpret_cast<char*>(result.data() + static_cast<size_t>(i) * d), sizeof(float) * d); 
        if (!file) throw std::runtime_error("Unexpected EOF at vector " + std::to_string(i + a));
    }

    return result; 
}   


std::vector<int> read_ivecs(const std::string& filename, int& dimension, int& num_vectors, int a = 1, int b = -1) {
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("I/O error: Unable to open the file " + filename);
    }

    // Read the dimension 'd' from the first vector
    int d = 0;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));
    if (file.eof() || d <= 0) {
        throw std::runtime_error("File is empty or corrupted.");
    }

    // Each vector size in bytes
    const int vec_sizeof = sizeof(int) + d * sizeof(int);

    // Get the total number of vectors
    file.seekg(0, std::ios::end);
    const std::streampos file_size = file.tellg();
    const int bmax = static_cast<int>(file_size / vec_sizeof);
    file.seekg(0, std::ios::beg);

    // Default values for 'a' and 'b'
    if (b == -1) {
        b = bmax;
    }
    if (a < 1) {
        a = 1;
    }
    if (b > bmax) {
        b = bmax;
    }
    if (b < a) {
        file.close();
        dimension = d;
        num_vectors = 0;
        return std::vector<int>();  // Return empty vector
    }

    // Move to the starting position
    const std::streampos start_pos = static_cast<std::streampos>((a - 1) * vec_sizeof);
    file.seekg(start_pos, std::ios::beg);

    // Number of vectors to read
    const int n = b - a + 1;

    // Prepare the output vector
    std::vector<int> v(n * d);
    std::cout << "Reading " << n << " vectors" << std::endl;
    // Read the vectors
    for (int i = 0; i < n; ++i) {
        int dim = 0;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (dim != d) {
            throw std::runtime_error("Dimension mismatch in vector " + std::to_string(i + a));
        }

        file.read(reinterpret_cast<char*>(v.data() + i * d), d * sizeof(int));

        if (file.eof()) {
            throw std::runtime_error("Unexpected end of file while reading vector " + std::to_string(i + a));
        }
    }

    file.close();
    dimension = d;
    num_vectors = n;
    return v;
}

// Function to read a range of vectors from a binary file
// std::vector<float> read_bvecs(const std::string& filename, int& dimension, int& num_vectors, int a = 1, int b = -1) {
//     std::cout << "Reading bvecs file: " << filename << std::endl;
//     std::ifstream file(filename, std::ios::binary);
//     if (!file.is_open()) {
//         throw std::runtime_error("I/O error: Unable to open file " + filename);
//     }

//     // Read dimension from first vector
//     int d = 0;
//     file.read(reinterpret_cast<char*>(&d), sizeof(int));
//     if (file.eof() || d <= 0) {
//         throw std::runtime_error("File is empty or corrupted.");
//     }
//     std::cout << "Dimension: " << d << std::endl;
//     const int vec_sizeof = sizeof(int) + d * sizeof(uint8_t);
//     file.seekg(0, std::ios::end);
//     const std::streampos file_size = file.tellg();
//     const int bmax = static_cast<int>(file_size / vec_sizeof);
//     file.seekg(0, std::ios::beg);
//     std::cout << "File size: " << file_size << std::endl;
//     if (b == -1) b = bmax;
//     if (a < 1) a = 1;
//     if (b > bmax) b = bmax;
//     if (b < a) {
//         file.close();
//         dimension = d;
//         num_vectors = 0;
//         return {};
//     }

//     file.seekg((a - 1) * vec_sizeof, std::ios::beg);
//     const int n = b - a + 1;
//     std::cout << "Number of vectors: " << n << std::endl;
//     std::vector<float> v(n * d);
//     std::vector<uint8_t> buffer(d);

//     for (int i = 0; i < n; ++i) {
//         int dim = 0;
//         file.read(reinterpret_cast<char*>(&dim), sizeof(int));
//         if (dim != d) {
//             throw std::runtime_error("Dimension mismatch in vector " + std::to_string(i + a));
//         }

//         file.read(reinterpret_cast<char*>(buffer.data()), d);
//         for (int j = 0; j < d; ++j) {
//             v[i * d + j] = static_cast<float>(buffer[j]);
//         }
//     }
//     std::cout << "Read vectors successfully!" << std::endl;
//     file.close();
//     dimension = d;
//     num_vectors = n;
//     return v;
// }
std::vector<float> read_bvecs(const std::string& filename, int& dimension, int& num_vectors, int a = 1, int b = -1) {
    std::cout << "Reading bvecs file: " << filename << std::endl;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("I/O error: Unable to open file " + filename);
    }
    
    // Read dimension from first vector
    int d = 0;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));
    if (file.eof() || d <= 0) {
        throw std::runtime_error("File is empty or corrupted.");
    }
    std::cout << "Dimension: " << d << std::endl;
    
    const int vec_sizeof = sizeof(int) + d * sizeof(uint8_t);
    file.seekg(0, std::ios::end);
    const std::streampos file_size = file.tellg();
    const int bmax = static_cast<int>(file_size / vec_sizeof);
    file.seekg(0, std::ios::beg);
    std::cout << "File size: " << file_size << std::endl;
    
    if (b == -1) b = bmax;
    if (a < 1) a = 1;
    if (b > bmax) b = bmax;
    if (b < a) {
        file.close();
        dimension = d;
        num_vectors = 0;
        return {};
    }
    
    file.seekg((a - 1) * vec_sizeof, std::ios::beg);
    const int n = b - a + 1;
    std::cout << "Number of vectors: " << n << std::endl;
    
    // 关键修复：使用 size_t 计算总元素数
    std::cout << "Allocating vector for " << n << " vectors..." << std::endl;
    
    // 使用 size_t 避免溢出
    const size_t total_elements = static_cast<size_t>(n) * static_cast<size_t>(d);
    std::cout << "Total elements: " << total_elements << std::endl;
    std::cout << "Memory needed: " << (total_elements * sizeof(float)) / (1024*1024*1024) << " GB" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 关键修复：使用 size_t 构造向量
    std::vector<float> v(total_elements);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Vector allocation successful in " << duration.count() << " ms" << std::endl;
    
    std::vector<uint8_t> buffer(d);
    
    std::cout << "Reading data..." << std::endl;
    for (int i = 0; i < n; ++i) {
        int dim = 0;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (dim != d) {
            throw std::runtime_error("Dimension mismatch in vector " + std::to_string(i + a));
        }
        file.read(reinterpret_cast<char*>(buffer.data()), d);
        
        // 使用 size_t 计算索引
        for (int j = 0; j < d; ++j) {
            v[static_cast<size_t>(i) * d + j] = static_cast<float>(buffer[j]);
        }
        
        // 每100万个向量报告一次进度
        if (i > 0 && i % 1000000 == 0) {
            std::cout << "Processed " << i << "/" << n << " vectors" << std::endl;
        }
    }
    
    std::cout << "Read vectors successfully!" << std::endl;
    file.close();
    dimension = d;
    num_vectors = n;
    return v;
}