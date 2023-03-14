#pragma once

#include "math.h"
#include <iostream>
#include <vector>
#include "assert.h"
#include <iomanip>
#include <string>
#include <algorithm>

/*
    SYMBOLS / TERMINOLOGY
    =====================

    Neural network has (N + 1) layers.

    l[0], ... , l[N]        ---         Network shape               ---  

    x                       ---         Input layer                 ---         len(x) == l[0]

    z[0], ... , z[N-1]      ---         Pre-activated layers        ---         len(z[i]) == l[i]

    a[0], ... , a[N-1]      ---         Activated layers            ---         len(a[i]) == l[i]

    W[0], ... , W[N-1]      ---         Weight matrices             ---         size(W[i]) == l[i-1] * l[i]         row-major ordering -- W[i](j,k) = W[i][(j * l[i+1]) + k]
    
    b[0], ... , b[N-1]      ---         Biases                      ---         len(b[i]) == l[i]

    y                       ---         True output                 ---         len(y) == l[N]


*/

class neural_network
{
private:
    int m_N; // num hidden layers + 1
    std::vector<int> m_l;
    std::vector<std::string> m_activate_types;
    std::string m_loss_type;
    std::vector<std::vector<double>> m_a;
    std::vector<std::vector<double>> m_z;
    std::vector<std::vector<double>> m_W;
    std::vector<std::vector<double>> m_b;

    double m_loss;
    std::vector<std::vector<double>> m_dJ_da;		
    std::vector<std::vector<double>> m_da_dz;
    std::vector<std::vector<double>> m_dJ_dW;
    std::vector<std::vector<double>> m_dJ_db;

public:
    neural_network() = delete;
    neural_network(const std::vector<int>& shape,
        const std::vector<std::string>& activate_types,
        const std::string& loss_type);

    void set_input(const std::vector<double>& data);
    std::vector<double> get_output();

    void fwd_pass(const std::vector<double>& x);
    void backprop(const std::vector<double>& y);

    void print();

};

// ctor
neural_network::neural_network(const std::vector<int>& shape,
    const std::vector<std::string>& activate_types,
    const std::string& loss_type) :
    m_l(shape), m_activate_types(activate_types), m_loss_type(loss_type)
{
    m_N = m_l.size() - 1;

    // initialise vectors
    for (int i = 0; i < m_N; ++i)
    {
        m_z.push_back(std::vector<double>(m_l[i+1], 0.0));
        m_a.push_back(std::vector<double>(m_l[i+1], 0.0));
        m_W.push_back(std::vector<double>(m_l[i]*m_l[i+1], 0.5));
        m_b.push_back(std::vector<double>(m_l[i+1], 0.5));
        
        m_dJ_da.push_back(std::vector<double>(m_l[i+1]));
        m_da_dz.push_back(std::vector<double>(m_l[i+1]));
        m_dJ_dW.push_back(std::vector<double>(m_l[i]*m_l[i+1]));
        m_dJ_db.push_back(std::vector<double>(m_l[i+1]));
    }
}

// get output
std::vector<double> neural_network::get_output()
{
    return m_a[m_N - 1];
}

void neural_network::fwd_pass(const std::vector<double>& x)
{
    for (int i = 0; i < m_N; ++i)
    {
        // compute z for this layer
        for (int j = 0; j < m_l[i+1]; ++j)
        {
            m_z[i][j] = m_b[i][j];

            for (int k = 0; k < m_l[i]; ++k)
            {
                if (i > 0)
                    m_z[i][j] += m_W[i][k*m_l[i+1] + j] * m_a[i-1][k];
                else
                    m_z[i][j] += m_W[i][k*m_l[i+1] + j] * x[k];
            }
        }

        // compute a for this layer
        if (m_activate_types[i] == "relu")
        {
            for (int j = 0; j < m_l[i+1]; ++j)
            {
                m_a[i][j] = (m_z[i][j] > 0.0) ? m_z[i][j] : 0.0;
            }
        }
        else if (m_activate_types[i] == "softmax")
        {
            double sum = 0.0;
            for (int j = 0; j < m_l[i+1]; ++j)
            {
                sum += exp(m_z[i][j]);
            }
            for (int j = 0; j < m_l[i+1]; ++j)
            {
                m_a[i][j] = exp(m_z[i][j]) / sum;
            }
        }
    }
}

void neural_network::backprop(const std::vector<double>& y)
{
    // compute loss
    m_loss = 0.0;
    if (m_loss_type == "mse")
    {
        for (int j = 0; j < m_l[m_N]; ++j)
        {
            m_loss += 0.5 * (m_a[m_N - 1][j] - y[j]) * (m_a[m_N - 1][j] - y[j]);
        }
    }
    else if (m_loss_type == "crossentropy")
    {
        for (int j = 0; j < m_l[m_N]; ++j)
        {
            m_loss += -1.0 * (y[j]*log(m_a[m_N - 1][j]) + (1.0 - y[j])*log(1.0 - m_a[m_N - 1][j]));
        }
    }
    //std::cout << "Loss: " << m_loss << std::endl;

    // compute gradients 
    // da/dz
    for (int i = 0; i < m_N; ++i)
    {
        if (m_activate_types[i] == "relu")
        {
            for (int j = 0; j < m_l[i+1]; ++j)
            {
                m_da_dz[i][j] = (m_z[i][j] > 0.0) ? 1.0 : 0.0;
            }
        }
        else if (m_activate_types[i] == "softmax")
        {
            for (int j = 0; j < m_l[i+1]; ++j)
            {
                m_da_dz[i][j] = m_a[i][j] * (1.0 - m_a[i][j]);
            }
        }
    }

    // dJ/da (for output layer)
    if (m_loss_type == "mse")
    {
        for (int j = 0; j < m_l[m_N]; ++j)
        {
            m_dJ_da[m_N - 1][j] = (m_a[m_N - 1][j] - y[j]);
        }
    }
    else if (m_loss_type == "crossentropy")
    {
        for (int j = 0; j < m_l[m_N]; ++j)
        {
            m_dJ_da[m_N - 1][j] = (m_a[m_N - 1][j] - y[j]);
            m_dJ_da[m_N - 1][j] /= m_a[m_N - 1][j] * (1.0 - m_a[m_N - 1][j]);
        }
    }

    // dJ/da (for hidden layers)
    for (int i = m_N - 2; i >= 0; i--)
    {
        for (int j = 0; j < m_l[i+1]; ++j)
        {
            m_dJ_da[i][j] = 0.0;
            for (int k = 0; k < m_l[i+2]; ++k)
            {
                m_dJ_da[i][j] += m_dJ_da[i+1][k] * m_da_dz[i+1][k] * m_W[i+1][j*m_l[i+2] + k];
            }
        }
    }

    // dJ/dW
    for (int i = 0; i < m_N; ++i)
    {
        for (int j = 0; j < m_l[i]; ++j)
        {
            for (int k = 0; k < m_l[i+1]; ++k)
            {
                m_dJ_dW[i][j*m_l[i+1] + k] = m_dJ_da[i][k] * m_da_dz[i][k] * m_a[i][j];
            }
        }
    }

    // dJ/db
    for (int i = 0; i < m_N; ++i)
    {
        for (int j = 0; j < m_l[i+1]; ++j)
        {
            m_dJ_db[i][j] = m_dJ_da[i][j] * m_da_dz[i][j];
        }
    }

    // adjust weights
    for (int i = 0; i < m_N; ++i)
    {
        for (int j = 0; j < m_l[i]; ++j)
        {
            for (int k = 0; k < m_l[i+1]; ++k)
            {
                m_W[i][j*m_l[i+1] + k] += -0.1 * m_dJ_dW[i][j*m_l[i+1] + k];
            }
        }
        for (int j = 0; j < m_l[i+1]; ++j)
        {
            m_b[i][j] += -0.1 * m_dJ_db[i][j];
        }
    }
}

void neural_network::print()
{
    for (int i = 0; i < m_N; ++i)
    {
        std::cout << "=== LAYER " << i+1 << " =====================\n\n";
        std::cout << "z\t" << "a\t" << "dJ/da\t" << "da/dz\n";
        for (int j = 0; j < m_l[i+1]; ++j)
        {
            std::cout.precision(2);
            std::cout << m_z[i][j] << "\t" << m_a[i][j] << "\t" << m_dJ_da[i][j] << "\t" << m_da_dz[i][j] << "\n";
        }
        std::cout << "\nW";
        for (int j = 0; j < m_l[i+1]; ++j)
        {
            std::cout << "\t";
        }
        std::cout << "dJ/dW\n";
        for (int j = 0; j < m_l[i]; ++j)
        {
            for (int k = 0; k < m_l[i+1]; ++k)
            {
                std::cout.precision(2);
                std::cout << m_W[i][j*m_l[i+1] + k] << "\t";
            }
            for (int k = 0; k < m_l[i+1]; ++k)
            {
                std::cout.precision(2);
                std::cout << m_dJ_dW[i][j*m_l[i+1] + k] << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "\nb";
        for (int j = 0; j < m_l[i+1]; ++j)
        {
            std::cout << "\t";
        }
        std::cout << "dJ/db\n";
        for (int j = 0; j < m_l[i+1]; ++j)
        {
            std::cout.precision(2);
            std::cout << m_b[i][j] << "\t";
        }
        for (int j = 0; j < m_l[i+1]; ++j)
        {
            std::cout.precision(2);
            std::cout << m_dJ_db[i][j] << "\t";
        }
        std::cout << "\n\n";
    }
}