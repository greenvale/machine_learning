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

namespace gv
{

namespace cg
{

/* Operation node base class
*/
class Operation
{
protected:
    int m_numIn, m_numChild;
    std::vector<const std::vector<double>*> m_pIn;
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

    void inputVec(const std::vector<double>* pVec, const int& ind);
    std::vector<double> output();

    friend void bind(const std::pair<Operation*, int>& par, const std::pair<Operation*, int>& child);
};

// parameter ctor taking number of inputs & number of child operations
// initialises containers for inputs, gradient inputs and operation gradients 
// output doesn't need initialisation
Operation::Operation(const int& numIn, const int& numChild) : m_numIn(numIn), m_numChild(numChild)
{
    assert(numIn > 0 && numChild >= 0);

    m_grad = std::vector<std::vector<double>>(m_numIn);
    m_pIn = std::vector<const std::vector<double>*>(m_numIn);
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
void Operation::inputVec(const std::vector<double>* pVec, const int& ind)
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
class unary : public Operation
{
public:
    int m_size;
    std::function<double(double)> m_fcn;
    std::function<double(double, double)> m_gradFcn;

    unary() = delete;
    unary(const int& numChild, const int& size, std::function<double(double)> fcn,
        std::function<double(double, double)> gradFcn);

    void comp() override;
    void grad() override;
};

// parameter ctor
unary::unary(const int& numChild, const int& size, std::function<double(double)> fcn,
    std::function<double(double, double)> gradFcn) : 
    Operation(1, numChild), m_size(size), m_fcn(fcn), m_gradFcn(gradFcn)
{
    assert(size > 0);

    m_out = std::vector<double>(m_size);
    m_grad[0] = std::vector<double>(m_size);
}

// node computation
void unary::comp()
{
    assert(m_pIn[0] != nullptr);

    std::transform(m_pIn[0]->cbegin(), m_pIn[0]->cend(), m_out.begin(), m_fcn);
}

// node gradient
void unary::grad()
{
    for (auto p : m_pGradIn)
        assert(p != nullptr && p->size() == m_size);

    std::vector<double> Dy = totalGradIn();
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
class NVecSum: public Operation
{
public:
    int m_size;

    NVecSum() = delete;
    NVecSum(const int& numIn, const int& numChild, const int& size);

    void comp() override;
    void grad() override;
};

// parameter ctor
NVecSum::NVecSum(const int& numIn, const int& numChild, const int& size) : 
    Operation(numIn, numChild), m_size(size)
{
    assert(size > 0);

    m_out = std::vector<double>(m_size);
    std::fill(m_grad.begin(), m_grad.end(), std::vector<double>(m_size));
}

// node computation
void NVecSum::comp()
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
void NVecSum::grad()
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
public:
    int m_inSize, m_outSize;

    MatVecMul(const int& nchild, const int& xsize, const int& ysize);

    void comp() override = 0;
    void grad() override = 0;
};

MatVecMul::MatVecMul(const int& nchild, const int& inSize, const int& outSize) : 
    Operation(2, nchild), m_inSize(inSize), m_outSize(outSize)
{
    assert(inSize > 0 && outSize > 0);

    m_out = std::vector<double>(m_outSize);
    m_grad[0] = std::vector<double>(m_inSize * m_outSize);
    m_grad[1] = std::vector<double>(m_inSize);
}

} // namespace cg

} // namespace gv