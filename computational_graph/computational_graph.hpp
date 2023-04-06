#pragma once

/*  Computational Graph
    William Denny (greenvale)
*/

#include <vector>
#include <assert.h>
#include <tuple>
#include <array>
#include <string>
#include <iostream>
#include <algorithm>
#include <functional>
#include <numeric>
#include <math.h>

namespace gv
{

namespace cg
{

/* Operation node base class
* Takes arbitrary number (numIn) of inputs as ptrs to vectors (pIn)
* Computes one output vector (out)
* Arbitrary number (numChild) of child operation nodes are bound to this node
* Takes child gradient (dJ/dy) inputs as ptrs to vectors (pGradIn)
* Computes gradient vectors (grad) for each input vector
*/
class Operation
{
protected:
    int m_numIn, m_numChild;
    std::vector<std::vector<double>*> m_pIn;
    std::vector<std::vector<double>> m_grad;
    std::vector<double> m_out;
    std::vector<const std::vector<double>*> m_pGradIn;

protected:
    std::vector<double> totalGradIn();

public:
    Operation() = delete;
    Operation(const int& numIn, const int& numChild);

    virtual void comp() = 0;
    virtual void grad() = 0;

    void inputVec(std::vector<double>* pVec, const int& ind);
    std::vector<double> output();
    void gradDescent(const int& ind, const double& alpha);

    friend void bind(const std::pair<Operation*, int>& par, const std::pair<Operation*, int>& child);
};

// parameter ctor taking number of inputs & number of child operations
// initialises containers for inputs, gradient inputs and operation gradients 
// output doesn't need initialisation
Operation::Operation(const int& numIn, const int& numChild) : m_numIn(numIn), m_numChild(numChild)
{
    assert(numIn > 0 && numChild >= 0);

    // initialise grad container
    m_grad = std::vector<std::vector<double>>(m_numIn);

    // initialise input ptr container, grad input ptr container
    m_pIn = std::vector<std::vector<double>*>(m_numIn);
    m_pGradIn = std::vector<const std::vector<double>*>(m_numChild);
}

// calculates total derivative for DJ/Dy 
// DJ/Dy = sum_i{ dJ/d(y_i) }
std::vector<double> Operation::totalGradIn()
{
    for (auto p : m_pGradIn)
        assert(p != nullptr && p->size() == m_pGradIn[0]->size());

    // initialise Dy vector as zero vector
    std::vector<double> Dy(m_pGradIn[0]->size(), 0);

    // accumulate each d(y_i) into Dy vector
    for (auto p : m_pGradIn)
        std::transform(Dy.cbegin(), Dy.cend(), p->cbegin(), Dy.begin(), std::plus<>());

    return Dy;
}

// bind input vector by ptr to operation node given input index
void Operation::inputVec(std::vector<double>* pVec, const int& ind)
{
    // assert input index is valid
    assert(ind >= 0 && ind < m_numIn);

    m_pIn[ind] = pVec;
}

// returns output
std::vector<double> Operation::output()
{
    return m_out;
}

// iteration of gradient descent on input element given input gradient and alpha
void Operation::gradDescent(const int& ind, const double& alpha)
{
    std::transform(m_pIn[ind]->cbegin(), m_pIn[ind]->cend(), m_grad[ind].cbegin(), m_pIn[ind]->begin(),
        [&](double xi, double dxi) {return xi - alpha * dxi; });
}

// binds 2 operations together in series in parent-child relationship
// parent output -> child input
// takes output index for parent and input index for child
void bind(const std::pair<Operation*, int>& par, const std::pair<Operation*, int>& child)
{
    // parent and child operation pointers are not nullptrs
    assert(par.first != nullptr && child.first != nullptr);

    // parent y index in range of nchild and child x index in range of npar
    assert(par.second >= 0 && par.second < par.first->m_numChild);
    assert(child.second >= 0 && child.second < child.first->m_numIn);

    // parent dy ptr = child dx ptr
    par.first->m_pGradIn[par.second] = &(child.first->m_grad[child.second]);

    // child x ptr = parent y ptr
    child.first->m_pIn[child.second] = &(par.first->m_out);
}

//**************************************************************************************************************************

/* Unary operation given by lambda function
* Input  : x (1 vector)
* Output : y (vector)
* y_i = lambda(x_i)
* (dJ/dx)_i = lambda'(x_i, y_i)
* Lambda function is unary so node has 1 input. Lambda derivative is binary taking input vector and output vector.
*/
class Unary : public Operation
{
private:
    int m_size;
public:
    std::function<double(double)> m_fcn;
    std::function<double(double, double)> m_gradFcn;

    Unary() = delete;
    Unary(const int& numChild, const int& size, std::function<double(double)> fcn,
        std::function<double(double, double)> gradFcn);

    void comp() override;
    void grad() override;
};

// parameter ctor
Unary::Unary(const int& numChild, const int& size, std::function<double(double)> fcn,
    std::function<double(double, double)> gradFcn) : 
    Operation(1, numChild), m_size(size), m_fcn(fcn), m_gradFcn(gradFcn)
{
    assert(size > 0);

    // initialise out
    m_out = std::vector<double>(m_size);

    // initialise grad
    m_grad[0] = std::vector<double>(m_size);
}

// node computation
void Unary::comp()
{
    assert(m_pIn[0] != nullptr);

    std::transform(m_pIn[0]->cbegin(), m_pIn[0]->cend(), m_out.begin(), m_fcn);
}

// node gradient
void Unary::grad()
{
    for (auto p : m_pGradIn)
        assert(p != nullptr && p->size() == m_size);

    // get total derivative for DJ/Dy
    std::vector<double> Dy = totalGradIn();

    // get dy/dx for unary operation using gradient function taking both inputs and outputs
    std::transform(m_pIn[0]->cbegin(), m_pIn[0]->cend(), m_out.cbegin(), m_grad[0].begin(), m_gradFcn);

    // multiply dy/dx with DJ/Dy to get dJ/dx
    std::transform(m_grad[0].cbegin(), m_grad[0].cend(), Dy.cbegin(), m_grad[0].begin(),
        [](double dx_i, double Dy_i) {return Dy_i * dx_i; });
}

//**************************************************************************************************************************

/* Summation of n vectors (n >= 2)
* Input : { x_1, ... , x_n } (>= 2 vectors)
* Output : y (vector)
* y = x_1 + ... + x_n
* dJ/d(x_i) = sum_j{ dJ/d(y_j) } * dy/dx = DJ/Dy * dy/dx = DJ/Dy
*/
class VecSum: public Operation
{
private:
    int m_size;
public:
    VecSum() = delete;
    VecSum(const int& numIn, const int& numChild, const int& size);

    void comp() override;
    void grad() override;
};

// parameter ctor
VecSum::VecSum(const int& numIn, const int& numChild, const int& size) : 
    Operation(numIn, numChild), m_size(size)
{
    assert(size > 0);

    // initialise out
    m_out = std::vector<double>(m_size);

    // initialise grad
    std::fill(m_grad.begin(), m_grad.end(), std::vector<double>(m_size));
}

// node computation
void VecSum::comp()
{
    for (auto p : m_pIn)
        assert(p != nullptr && p->size() == m_size);

    // set y to zero vector
    std::fill(m_out.begin(), m_out.end(), 0);

    // accumulate x_i in vector y for all i
    for (auto p : m_pIn)
        std::transform(m_out.cbegin(), m_out.cend(), p->cbegin(), m_out.begin(), std::plus<>());
}

// node gradients
void VecSum::grad()
{
    for (auto p : m_pGradIn)
        assert(p != nullptr && p->size() == m_size);
    
    // get total derivative DJ/Dy
    std::vector<double> Dy = totalGradIn();

    // set d(x_i) to Dy for all i
    for (int i = 0; i < m_grad.size(); ++i)
        std::copy(Dy.cbegin(), Dy.cend(), m_grad[i].begin());
}

//**************************************************************************************************************************

/*  Matrix-vector multiplication
* Input  : W (matrix) , x (vector)
* Output : y (vector)
* y = W * x
* dJ/dW = sum_j{ dJ/d(y_j) } * dy/dW
*       = sum_j{ dJ/d(y_j) } * 3d_array( dy_<ROW> / dW_<LANE><COL> )
* (dJ/dW)_ij = (DJ/Dy)_i * x_j
* dJ/dx = sum_j{ dJ/d(y_j) } * dy/dx
*       = sum_j{ dJ/d(y_j) } * W
* (dJ/dx)_i = sum_j{ (DJ/Dy)_j * W_ji }
*/
class MatVecMul : public Operation
{
private:
    int m_inSize, m_outSize;
public:
    MatVecMul() = delete;
    MatVecMul(const int& numChild, const int& inSize, const int& outSize);

    void comp() override;
    void grad() override;
};

MatVecMul::MatVecMul(const int& numChild, const int& inSize, const int& outSize) : 
    Operation(2, numChild), m_inSize(inSize), m_outSize(outSize)
{
    assert(inSize > 0 && outSize > 0);

    // initialise out
    m_out = std::vector<double>(m_outSize);

    // initialise grad
    m_grad[0] = std::vector<double>(m_inSize * m_outSize);
    m_grad[1] = std::vector<double>(m_inSize);
}

// compute node
void MatVecMul::comp()
{
    assert(m_pIn[0] != nullptr && m_pIn[1] != nullptr);
    assert(m_pIn[0]->size() == m_inSize * m_outSize && m_pIn[1]->size() == m_inSize);

    // set y to zero vector
    std::fill(m_out.begin(), m_out.end(), 0);

    for (int i = 0; i < m_outSize; ++i)
        for (int j = 0; j < m_inSize; ++j)
            m_out[i] += m_pIn[0]->at(m_inSize * i + j) * m_pIn[1]->at(j);
}

// compute node gradients
void MatVecMul::grad()
{
    for (auto p : m_pGradIn)
        assert(p != nullptr && p->size() == m_outSize);

    // get total derivative for DJ/Dy
    std::vector<double> Dy = totalGradIn();

    // initialise vector for dJ/dW
    std::fill(m_grad[0].begin(), m_grad[0].end(), 0);

    // initialise vector for dJ/dx
    std::fill(m_grad[1].begin(), m_grad[1].end(), 0);

    // calculate dJ/dW
    for (int i = 0; i < m_outSize; ++i)        // iterate rows of y
        for (int j = 0; j < m_inSize; ++j)    // iterate rows of x
            m_grad[0][m_inSize * i + j] = Dy[i] * m_pIn[1]->at(j);

    // calculate dJ/dx
    for (int i = 0; i < m_inSize; ++i)        // iterate rows of x
        for (int j = 0; j < m_outSize; ++j)    // iterate rows of y
            m_grad[1][i] += Dy[j] * m_grad[0][m_inSize * j + i];
}

//**************************************************************************************************************************

// Loss functions

class RMSE : public Operation
{
private:
    int m_size;
public:
    RMSE() = delete;
    RMSE(const int& size);

    void comp() override;
    void grad() override;
};

RMSE::RMSE(const int& size) : Operation(2, 0), m_size(size)
{
    assert(size > 0);

    // initialise out 
    m_out = std::vector<double>(1);

    // initialise grad
    m_grad[0] = std::vector<double>(m_size);
    m_grad[1] = std::vector<double>(m_size);
}

void RMSE::comp()
{
    assert(m_pIn[0] != nullptr && m_pIn[0]->size() == m_size);
    assert(m_pIn[1] != nullptr && m_pIn[1]->size() == m_size);

    m_out[0] = 0;

    for (int i = 0; i < m_size; ++i)
        m_out[0] += (m_pIn[0]->at(i) - m_pIn[1]->at(i)) * (m_pIn[0]->at(i) - m_pIn[1]->at(i));
}

void RMSE::grad()
{
    // dJ/dx0
    for (int i = 0; i < m_size; ++i)
        m_grad[0][i] = m_pIn[0]->at(i) - m_pIn[1]->at(i);

    // dJ/dx1
    std::transform(m_grad[0].cbegin(), m_grad[0].cend(), m_grad[1].begin(), [](double x) { return -x; });
}

//**************************************************************************************************************************

// Activation functions

// ReLU
class ReLU : public Unary
{
public:
    ReLU(const int& numChild, const int& size);
};

ReLU::ReLU(const int& numChild, const int& size) : Unary(numChild, size,
    [](double x) { return x > 0 ? x : 0; },
    [](double x, double y) { return x > 0 ? 1 : 0; })
{
}

// Sigmoid function
class Sigmoid : public Unary
{
public:
    Sigmoid(const int& numChild, const int& size);
};

Sigmoid::Sigmoid(const int& numChild, const int& size) : Unary(numChild, size,
    [](double x) {return 1.0 / (1.0 + exp(-1.0 * x)); },
    [](double x, double y) {return y * (1.0 - y); })
{
}

// Softmax layer (not unary operation)
class Softmax : public Operation
{
private:
    int m_size;
public:
    Softmax(const int& numChild, const int& size);

    void comp() override;
    void grad() override;
};

Softmax::Softmax(const int& numChild, const int& size) : Operation(1, numChild), m_size(size)
{
    assert(size > 0);

    // initialise out
    m_out = std::vector<double>(m_size);

    // initialise grad
    m_grad[0] = std::vector<double>(m_size);
}

void Softmax::comp()
{
    assert(m_pIn[0] != nullptr && m_pIn[0]->size() == m_size);

    std::transform(m_pIn[0]->cbegin(), m_pIn[0]->cend(), m_out.begin(), [](double x) {return exp(x); });
    double tmp = std::accumulate(m_out.cbegin(), m_out.cend(), 0);
    tmp = 1.0 / tmp;
    std::transform(m_out.cbegin(), m_out.cend(), m_out.begin(), [&](double x) {return x * tmp; });
}

void Softmax::grad()
{
    for (auto p : m_pGradIn)
        assert(p != nullptr && p->size() == m_size);

    // get total derivative DJ/Dy
    std::vector<double> Dy = totalGradIn();

    // get dy/dx
    std::transform(m_out.cbegin(), m_out.cend(), m_grad[0].begin(), [](double x) {return x * (1.0 - x); });
    
    // multiply dy/dx by DJ/Dy to get dJ/dx
    std::transform(m_grad[0].cbegin(), m_grad[0].cend(), Dy.cbegin(), m_grad[0].begin(),
        [](double dx_i, double Dy_i) {return Dy_i * dx_i; });
}

} // namespace cg

} // namespace gv