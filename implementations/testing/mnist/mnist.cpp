#include <iostream>
#include <fstream>
#include <bitset>
#include <vector>
#include <tuple>

#include "../../neural_network.hpp"

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

private:
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

    void train();

    void draw(const int& set, const int& n);

};

MNIST::MNIST(const std::string& p_train_imgs, 
    const std::string& p_train_labels, 
    const std::string& p_test_imgs, 
    const std::string& p_test_labels)
{
}

/******************************************************************************************************************/


int main()
{   
    std::tuple<uint8_t*, std::vector<uint32_t>, uint32_t, uint32_t> import_dat = read_IDX<uint8_t>("./data/train-images.idx3-ubyte");
}