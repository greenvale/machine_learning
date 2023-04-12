
#include "mnist_csv.hpp"
#include <string>
#include <fstream>
#include "math.h"

#include "mnist_nn.hpp"
#include "../../../computational_graph/computational_graph.hpp"

/*
    MNIST Test for neural network implementations
*/

// Test with computational graph
void TEST_1(const MNIST& mnist)
{
    double alpha = 0.01;

    gv::cg::MatVecMul   M1(1, 28*28, 30);
    gv::cg::MatVecMul   M2(1, 30, 10);
    gv::cg::VecSum      V1(2, 1, 30);
    gv::cg::VecSum      V2(2, 1, 10);
    gv::cg::ReLU        A1(1, 30);
    gv::cg::Classifier  L (10);

    std::vector<double> Xbuf(28*28);
    std::vector<double> Ybuf(10);
    std::vector<double> W1(28*28*30);
    std::vector<double> W2(30*10);
    std::vector<double> B1(30); 
    std::vector<double> B2(10);

    gv::cg::bind({&M1, 0}, {&V1, 0});
    gv::cg::bind({&V1, 0}, {&A1, 0});
    gv::cg::bind({&A1, 0}, {&M2, 1});
    gv::cg::bind({&M2, 0}, {&V2, 0});
    gv::cg::bind({&V2, 0}, {&L,  0});

    M1.inputVec(&W1, 0);
    M2.inputVec(&W2, 0);
    V1.inputVec(&B1, 1);
    V2.inputVec(&B2, 1);
    M1.inputVec(&Xbuf, 1);
    L.inputVec(&Ybuf, 1);

    // initialise random vectors for weights and biases
    std::transform(W1.cbegin(), W1.cend(), W1.begin(), [](double x){return 2*((double)rand()/(double)RAND_MAX) - 1;});
    std::transform(B1.cbegin(), B1.cend(), B1.begin(), [](double x){return 2*((double)rand()/(double)RAND_MAX) - 1;});
    std::transform(W2.cbegin(), W2.cend(), W2.begin(), [](double x){return 2*((double)rand()/(double)RAND_MAX) - 1;});
    std::transform(B2.cbegin(), B2.cend(), B2.begin(), [](double x){return 2*((double)rand()/(double)RAND_MAX) - 1;});

    for (int i = 0; i < mnist.trainX.size(); ++i)
    {
        std::copy(mnist.trainX[i].cbegin(), mnist.trainX[i].cend(), Xbuf.begin());
        std::copy(mnist.trainY[i].cbegin(), mnist.trainY[i].cend(), Ybuf.begin());

        M1.comp();
        V1.comp();
        A1.comp();
        M2.comp();
        V2.comp();
        L.comp();

        if (i % 1000 == 0)
        {
            std::cout << i << " | Loss: " << L.output()[10] << std::endl; 
        }

        L.grad();
        V2.grad();
        M2.grad();
        A1.grad();
        V1.grad();
        M1.grad();

        M1.gradDescent(0, alpha);
        V1.gradDescent(1, alpha);
        M2.gradDescent(0, alpha);
        V2.gradDescent(1, alpha);
    }

    for (int i = 0; i < 20; ++i)
    {
        std::copy(mnist.trainX[i].cbegin(), mnist.trainX[i].cend(), Xbuf.begin());
        std::copy(mnist.trainY[i].cbegin(), mnist.trainY[i].cend(), Ybuf.begin());

        M1.comp();
        V1.comp();
        A1.comp();
        M2.comp();
        V2.comp();
        L.comp();

        std::cout << L.output() << std::endl;
        std::cout << Ybuf << std::endl;
        std::cout << "\n\n";
    }
}

// Test with std::array implementation
void TEST_2(const MNIST& mnist)
{
    double alpha = 0.01;

    gv::MNIST_NN nn;

    for (int i = 0; i < mnist.trainX.size(); ++i)
    {
        nn.setX(mnist.trainX[i]);
        nn.setY(mnist.trainY[i]);
        nn.forwardPass();
        nn.backprop();
        nn.gradDescentInc(alpha);

        if (i % 1000 == 0)
        {
            std::cout << i << " | Loss: " << nn.getLoss() << std::endl; 
        }

    }

    for (int i = 0; i < 20; ++i)
    {
        nn.setX(mnist.testX[i]);
        nn.setY(mnist.testY[i]);
        nn.forwardPass();
        std::cout << nn.getOutput() << std::endl;
        std::cout << mnist.testY[i] << std::endl;
        std::cout << "Loss: " << nn.getLoss() << std::endl;
        std::cout << "\n\n";
    }
}

int main()
{
    auto trainPair = MNIST_Load_CSV("./data/mnist_train.csv", 60000);
    auto testPair =  MNIST_Load_CSV("./data/mnist_test.csv", 10000);
    
    MNIST mnist = MNIST_From_Int(trainPair.first, trainPair.second, testPair.first, testPair.second);

    TEST_1(mnist);

    TEST_2(mnist);
}