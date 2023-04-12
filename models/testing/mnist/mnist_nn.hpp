#pragma once

/* Neural network for MNIST dataset
*/

#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>
#include "mnist_csv.hpp"
#include <math.h>
#include <bitset>
#include <array>

namespace gv
{

class MNIST_NN
{
public:
    std::array<double, 28*28*30>    W1;
    std::array<double, 30*10>       W2;
    std::array<double, 30>          B1;
    std::array<double, 10>          B2;

    std::array<double, 28*28>       X;
    std::array<double, 28*28>       Y;

    std::array<double, 30>          Z1;
    std::array<double, 10>          Z2;
    std::array<double, 30>          H1;
    std::array<double, 10>          H2;

    std::array<double, 28*28*30>    DW1;
    std::array<double, 30*10>       DW2;
    std::array<double, 30>          DB1;
    std::array<double, 10>          DB2;
    std::array<double, 30>          DZ1;
    std::array<double, 10>          DZ2;

    double loss;

    MNIST_NN();

    void setX(std::vector<double> x);
    void setY(std::vector<double> y);

    std::vector<double> getOutput();
    double getLoss();

    void forwardPass();
    void backprop();
    void gradDescentInc(double alpha);
};

MNIST_NN::MNIST_NN()
{
    std::transform(W1.cbegin(), W1.cend(), W1.begin(), [](double x){return 2*((double)rand()/(double)RAND_MAX) - 1;});
    std::transform(B1.cbegin(), B1.cend(), B1.begin(), [](double x){return 2*((double)rand()/(double)RAND_MAX) - 1;});
    std::transform(W2.cbegin(), W2.cend(), W2.begin(), [](double x){return 2*((double)rand()/(double)RAND_MAX) - 1;});
    std::transform(B2.cbegin(), B2.cend(), B2.begin(), [](double x){return 2*((double)rand()/(double)RAND_MAX) - 1;});
}

void MNIST_NN::setX(std::vector<double> x)
{
    std::copy(x.cbegin(), x.cend(), X.begin());
}

void MNIST_NN::setY(std::vector<double> y)
{
    std::copy(y.cbegin(), y.cend(), Y.begin());
}

std::vector<double> MNIST_NN::getOutput()
{
    std::vector<double> otpt(H2.cbegin(), H2.cend());
    return otpt;
}

double MNIST_NN::getLoss()
{
    return loss;
}

void MNIST_NN::forwardPass()
{
    // first layer
    std::copy(B1.cbegin(), B1.cend(), Z1.begin());
    for (int i = 0; i < 30; ++i)
        for (int j = 0; j < 28*28; ++j)
            Z1[i] += W1[i*28*28 + j] * X[j];
    std::transform(Z1.cbegin(), Z1.cend(), H1.begin(), [](double x){return x > 0 ? x : 0;});

    // second layer
    std::copy(B2.cbegin(), B2.cend(), Z2.begin());
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 30; ++j)
            Z2[i] += W2[i*30 + j] * H1[j];
    std::transform(Z2.cbegin(), Z2.cend(), H2.begin(), [](double x){return exp(x);});
    double sum = 0;
    for (int i = 0; i < 10; ++i)
        sum += H2[i];
    std::transform(H2.cbegin(), H2.cend(), H2.begin(), [&](double x){return x / sum;});

    // loss
    loss = 0;
    for (int i = 0; i < 10; ++i)
        loss -= Y[i] * log(H2[i]);
        
}

void MNIST_NN::backprop()
{
    // gradient layer 2
    std::transform(H2.cbegin(), H2.cend(), Y.cbegin(), DZ2.begin(), [](double h, double y){return h - y;});
    std::copy(DZ2.cbegin(), DZ2.cend(), DB2.begin());
    std::fill(DW2.begin(), DW2.end(), 0);
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 30; ++j)
            DW2[i*30 + j] = DZ2[i] * H1[j];

    // gradient layer 1
    std::fill(DZ1.begin(), DZ1.end(), 0);
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 30; ++j)
        {
            double dhdz = Z1[j] > 0 ? 1 : 0;
            DZ1[j] += DZ2[i] * W2[i*30 + j] * dhdz;
        }
    std::copy(DZ1.cbegin(), DZ1.cend(), DB1.begin());
    std::fill(DW1.begin(), DW1.end(), 0);
    for (int i = 0; i < 30; ++i)
        for (int j = 0; j < 28*28; ++j)
            DW1[i*28*28 + j] = DZ1[i] * X[j];
}


void MNIST_NN::gradDescentInc(double alpha)
{
    std::transform(W1.cbegin(), W1.cend(), DW1.cbegin(), W1.begin(), [alpha](double w, double d){return w - alpha*d;});
    std::transform(W2.cbegin(), W2.cend(), DW2.cbegin(), W2.begin(), [alpha](double w, double d){return w - alpha*d;});
    std::transform(B1.cbegin(), B1.cend(), DB1.cbegin(), B1.begin(), [alpha](double b, double d){return b - alpha*d;});
    std::transform(B2.cbegin(), B2.cend(), DB2.cbegin(), B2.begin(), [alpha](double b, double d){return b - alpha*d;});
}

}