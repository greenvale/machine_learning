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
        class operation
        {
        public:
            std::vector<const std::vector<double>*> vp_x;
            std::vector<std::vector<double>> m_dx;
            std::vector<double> m_y;
            std::vector<const std::vector<double>*> vp_dy;

            operation();
            operation(const int& npar, const int& nchild);

            virtual void comp() = 0;
            virtual void grad() = 0;
            std::vector<double> get_Dy();
            void input_vec(const std::vector<double>* pvec, const int& ind);
        };

        // default ctor
        operation::operation()
        {
        }

        // parameter ctor taking number of parent nodes & number of child nodes
        // initialises containers for inputs, outputs, gradients
        operation::operation(const int& npar, const int& nchild)
        {
            assert(npar > 0 && nchild >= 0);

            m_dx = std::vector<std::vector<double>>(npar);
            vp_x = std::vector<const std::vector<double>*>(npar);
            vp_dy = std::vector<const std::vector<double>*>(nchild);
        }

        // calculates total derivative for DJ/Dy 
        // DJ/Dy = sum_i{ dJ/d(y_i) }
        std::vector<double> operation::get_Dy()
        {
            // assert that dy container not empty and each element has an equal size > 0
            assert(vp_dy.size() > 0 && vp_dy[0]->size() > 0);
            for (auto p : vp_dy)
                assert(p != nullptr && p->size() == vp_dy[0]->size());

            // initialise Dy vector as zero vector
            std::vector<double> Dy(vp_dy[0]->size(), 0);

            // accumulate each d(y_i) into Dy vector
            for (auto p : vp_dy)
                std::transform(Dy.cbegin(), Dy.cend(), p->cbegin(), Dy.begin(), std::plus<>());

            return Dy;
        }

        // bind input vector by ptr to operation node given input index
        void operation::input_vec(const std::vector<double>* pvec, const int& ind)
        {
            // assert input index is valid
            assert(ind >= 0 && ind < vp_x.size());

            vp_x[ind] = pvec;
        }

        // binds 2 operations together in series in parent-child relationship
        // parent output (y) -> child input (x)
        // takes output index for parent and input index for child
        void bind(const std::pair<operation*, int>& par_y, const std::pair<operation*, int>& child_x)
        {
            // assert that parent and child ptrs point to valid operation with > 0 inputs/outputs and valid indexes
            assert(par_y.first != nullptr && child_x.first != nullptr);
            assert(par_y.first->vp_dy.size() > 0 && child_x.first->vp_x.size() > 0);
            assert(par_y.second >= 0 && par_y.second < par_y.first->vp_dy.size());
            assert(child_x.second >= 0 && child_x.second < child_x.first->vp_x.size());

            // parent dy ptr = child dx ptr
            par_y.first->vp_dy[par_y.second] = &(child_x.first->m_dx[child_x.second]);

            // child x ptr = parent y ptr
            child_x.first->vp_x[child_x.second] = &(par_y.first->m_y);
        }

        //**************************************************************************************************************************

        /* Unary operation given by lambda function
        * Input : x (vector)
        * Output : y (vector)
        * y_i = lambda(x_i)
        * (dJ/dx)_i = lambda'(x_i, y_i)
        * Lambda function is unary so node has 1 input parent. Lambda derivative is binary taking input vector and output vector.
        */
        class unary : public operation
        {
        public:
            int m_size;
            std::function<double(double)> m_fcn;
            std::function<double(double, double)> m_fcn_grad;

            unary(const int& nchild, const int& size, std::function<double(double)> fcn,
                std::function<double(double, double)> fcn_grad);

            void comp() override;
            void grad() override;
        };

        // parameter ctor
        unary::unary(const int& nchild, const int& size, std::function<double(double)> fcn,
            std::function<double(double, double)> fcn_grad) : operation(1, nchild)
        {
            assert(size > 0);

            m_size = size;
            m_fcn = fcn;
            m_fcn_grad = fcn_grad;

            m_y = std::vector<double>(m_size);
            m_dx[0] = std::vector<double>(m_size);
        }

        // node computation
        void unary::comp()
        {
            assert(vp_x[0] != nullptr);

            std::transform(vp_x[0]->cbegin(), vp_x[0]->cend(), m_y.begin(), m_fcn);
        }

        // node gradient
        void unary::grad()
        {
            assert(vp_dy.size() > 0);
            for (auto p : vp_dy)
                assert(p != nullptr && p->size() == m_size);

            std::vector<double> Dy = get_Dy();
            std::transform(m_dx[0].cbegin(), m_dx[0].cend(), Dy.cbegin(), m_dx[0].begin(),
                [](double dx_i, double Dy_i) {return Dy_i * dx_i; });
        }

    } // namespace cg

} // namespace gv