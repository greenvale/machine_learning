#pragma once

/* MNIST dataset */

#include <iostream>
#include <fstream>
#include <bitset>
#include <vector>
#include <tuple>

// prints vectors
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << "[";
    for (int i = 0; i < vec.size(); ++i)
    {
        if (i > 0)
            os << ", ";
        os << vec[i];
    }
    os << "]";
    return os;
}

// reverses ordering of bytes in word (needed for IDX file formatting)
template <typename T>
T swap(T word)
{
    T swapped;

    for (int i = 0; i < sizeof(T); ++i)
    {
        swapped = swapped << 8;
        swapped = swapped | (word & 0xff);
        word = word >> 8;
    }

    return swapped;
}

// import data from IDX file
template <typename T>
std::tuple<T*, std::vector<uint32_t>, uint32_t, uint32_t> read_IDX(const std::string& path)
{
    std::ifstream buf;
    uint32_t magic_num;
    uint32_t dtype_code;
    uint32_t num_dims;
    uint32_t num_elems = 1;
    std::vector<uint32_t> sizes;
    
    T* data;

    uint32_t tmp;

    buf.open(path.c_str(), std::ios::in);

    buf.read((char*)(&magic_num), 4);
    num_dims = (swap(magic_num)) & 0xff;
    dtype_code = (swap(magic_num) >> 8) & 0xff;

    std::cout << "IDX IMPORT: " << path << "\nDtype code: " << dtype_code << " | Number of dims: " << num_dims << " | Size: (";
    
    sizes = std::vector<uint32_t>(num_dims);

    for (int i = 0; i < num_dims; ++i)
    {
        if (i > 0)
            std::cout << ", ";
        
        buf.read((char*)(&tmp), 4);
        tmp = swap(tmp);
        sizes[i] = (uint32_t)tmp;
        num_elems *= tmp;
        std::cout << tmp;
    }
    std::cout << ") | Number of elems: " << num_elems << "\n";

    data = new T[num_elems];
    buf.read((char*)data, num_elems * sizeof(T));

    buf.close();

    return { data, sizes, num_elems, dtype_code };
}

/******************************************************************************************************************/

class MNIST
{

public:
    std::vector<std::vector<double>> m_train_x;
    std::vector<std::vector<double>> m_train_y;
    std::vector<std::vector<double>> m_test_x;
    std::vector<std::vector<double>> m_test_y;

public:
    MNIST() = delete;
    MNIST(const std::string& p_train_imgs, 
        const std::string& p_train_labels, 
        const std::string& p_test_imgs, 
        const std::string& p_test_labels);

    void print_char(const std::string& set, const int& n) const;

};

// ctor
MNIST::MNIST(const std::string& p_train_imgs, 
    const std::string& p_train_labels, 
    const std::string& p_test_imgs, 
    const std::string& p_test_labels)
{
    std::tuple<uint8_t*, std::vector<uint32_t>, uint32_t, uint32_t> train_imgs_tup;
    std::tuple<uint8_t*, std::vector<uint32_t>, uint32_t, uint32_t> train_labels_tup;
    std::tuple<uint8_t*, std::vector<uint32_t>, uint32_t, uint32_t> test_imgs_tup;
    std::tuple<uint8_t*, std::vector<uint32_t>, uint32_t, uint32_t> test_labels_tup;

    train_imgs_tup   = read_IDX<uint8_t>(p_train_imgs);
    train_labels_tup = read_IDX<uint8_t>(p_train_labels);
    test_imgs_tup    = read_IDX<uint8_t>(p_test_imgs);
    test_labels_tup  = read_IDX<uint8_t>(p_test_labels);

    uint8_t* train_imgs_arr   = std::get<0>(train_imgs_tup);
    uint8_t* train_labels_arr = std::get<0>(train_labels_tup);
    uint8_t* test_imgs_arr    = std::get<0>(test_imgs_tup);
    uint8_t* test_labels_arr  = std::get<0>(test_labels_tup);

    // allocate vectors
    m_train_x = std::vector<std::vector<double>>(60000);
    m_train_y = std::vector<std::vector<double>>(60000);
    m_test_x = std::vector<std::vector<double>>(10000);
    m_test_y = std::vector<std::vector<double>>(10000);

    for (int i = 0; i < 60000; ++i)
    {
        m_train_x[i] = std::vector<double>(28*28);
        m_train_y[i] = std::vector<double>(10, 0.0);
    }
    for (int i = 0; i < 10000; ++i)
    {
        m_test_x[i] = std::vector<double>(28*28);
        m_test_y[i] = std::vector<double>(10, 0.0);
    }

    // assign data in vectors
    for (int i = 0; i < 60000; ++i)
    {
        for (int j = 0; j < 28*28; ++j)
            m_train_x[i][j] = ((double)train_imgs_arr[28*28*i + j]) / 255.0;
        m_train_y[i][train_labels_arr[i]] = 1.0;
    }
    for (int i = 0; i < 10000; ++i)
    {
        for (int j = 0; j < 28*28; ++j)
            m_test_x[i][j] = ((double)test_imgs_arr[28*28*i + j]) / 255.0;
        m_test_y[i][test_labels_arr[i]] = 1.0;
    }
}

// draw MNIST character and label to output
void MNIST::print_char(const std::string& set, const int& n) const
{
    std::cout << "\n";
    for (int i = 0; i < 28; ++i)
    {
        std::cout << "  ";
        for (int j = 0; j < 28; ++j)
        {
            if (set == "train")
            {
                if (m_train_x[n][28*i + j] < 0.3)
                    if (i == 0 || j == 0 || i == 27 || j == 27)
                        std::cout << ". ";
                    else
                        std::cout << "  ";
                else if (m_train_x[n][28*i + j] < 0.6)
                    std::cout << "x ";
                else
                    std::cout << "# ";
            }
            else if (set == "test")
            {
                if (m_test_x[n][28*i + j] < 0.3)
                    if (i == 0 || j == 0 || i == 27 || j == 27)
                        std::cout << ". ";
                    else
                        std::cout << "  ";
                else if (m_test_x[n][28*i + j] < 0.6)
                    std::cout << "x ";
                else
                    std::cout << "# ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n    ";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << i << " ";
    }
    std::cout << "\n    ";
    for (int i = 0; i < 10; ++i)
    {
        if (set == "train" && m_train_y[n][i] == 1.0)
            std::cout << "^ ";
        else if (set == "test" && m_test_y[n][i] == 1.0)
            std::cout << "^ ";
        else
            std::cout << "  ";
    }
    std::cout << "\n  ";
}