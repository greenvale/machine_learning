#pragma once

/* MNIST CSV LOADER
*/

#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <array>
#include <string>
#include <assert.h>
#include <sstream>

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

struct MNIST
{
    std::vector<std::vector<double>> trainX;
    std::vector<std::vector<double>> trainY;
    std::vector<std::vector<double>> testX;
    std::vector<std::vector<double>> testY;    
};

// Loads X, Y values prenormalisation from CSV file (either train or test) with N (60,000 or 10,000)
std::pair<std::vector<std::vector<uint8_t>>, std::vector<std::vector<uint8_t>>> MNIST_Load_CSV(std::string path, int N)
{
    std::vector<std::vector<uint8_t>> X;
    std::vector<std::vector<uint8_t>> Y;

    std::ifstream is;
    is.open(path);
    for (int i = 0; i < N; ++i)
    {
        std::vector<uint8_t> x;
        std::vector<uint8_t> y(10, 0);

        std::string line;
        std::getline(is, line);

        std::stringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        int itoken = std::stoi(token);
        assert(itoken >= 0 && itoken < 10);
        y[itoken] = 1;

        while (std::getline(ss, token, ','))
        {
            int itoken = std::stoi(token);
            assert(itoken >= 0 && itoken <= 255);
            x.push_back((uint8_t)itoken);
        }

        X.push_back(x);
        Y.push_back(y);
    }
    is.close();

    std::cout << "Loaded MNIST CSV file" << std::endl;

    return {X, Y};
}

MNIST MNIST_From_Int(const std::vector<std::vector<uint8_t>>& trainXInt, const std::vector<std::vector<uint8_t>>& trainYInt,
    const std::vector<std::vector<uint8_t>>& testXInt, const std::vector<std::vector<uint8_t>>& testYInt)
{
    MNIST mnist;

    for (int i = 0; i < trainXInt.size(); ++i)
    {
        std::vector<double> x;
        std::vector<double> y;

        for (int j = 0; j < 28*28; ++j)
            x.push_back((double)trainXInt[i][j]/255.0);
        for (int j = 0; j < 10; ++j)
            y.push_back((double)trainYInt[i][j]);
        
        mnist.trainX.push_back(x);
        mnist.trainY.push_back(y);
    }

    for (int i = 0; i < testXInt.size(); ++i)
    {
        std::vector<double> x;
        std::vector<double> y;

        for (int j = 0; j < 28*28; ++j)
            x.push_back((double)testXInt[i][j]/255.0);
        for (int j = 0; j < 10; ++j)
            y.push_back((double)testYInt[i][j]);
        
        mnist.testX.push_back(x);
        mnist.testY.push_back(y);
    }

    std::cout << "Loaded MNIST from Int" << std::endl;

    return mnist;
}

void MNIST_Print_Char(std::vector<double> img)
{
    std::cout << "\n";
    for (int i = 0; i < 28; ++i)
    {
        std::cout << "  ";
        for (int j = 0; j < 28; ++j)
        {
            if (img[28*i + j] < 0.3)
                if (i == 0 || j == 0 || i == 27 || j == 27)
                    std::cout << ". ";
                else
                    std::cout << "  ";
            else if (img[28*i + j] < 0.6)
                std::cout << "x ";
            else
                std::cout << "# ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}