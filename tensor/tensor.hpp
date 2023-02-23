#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <assert.h>

/**********************************************************************************************************************************************************************************/

class Tensor
{
private:
	std::vector<int> m_shape;
	std::vector<int> m_increment; // increment steps for indexing dimensions

	int m_num_dims;
	int m_num_elems;

	double* m_data;

public:
	Tensor();
	Tensor(std::vector<int> shape, std::vector<double> data);
	Tensor(std::vector<int> shape, double val);
	~Tensor();

	// function for initialising tensor, used by ctors
	void initialise(std::vector<int> shape);

	// operator overloads
	double& operator[](int index);
	double& operator[](std::vector<int> sub);
	
	int get_num_dims();
	std::vector<int> get_shape();

	// functions for running lambdas on elemens given their index/sub
	void lambda_by_index(std::function<void(Tensor*, const int&)> lambda);
	void lambda_by_sub(std::function<void(Tensor*, const std::vector<int>&)> lambda);
	
	void display();

private:

	double& get_elem_by_sub(std::vector<int> sub);

};

/**********************************************************************************************************************************************************************************/

Tensor::Tensor()
{

}

Tensor::Tensor(std::vector<int> shape, std::vector<double> data)
{
	assert(shape.size() > 0);

	// set m_shape, m_num_elems, and allocate data memory
	this->initialise(shape);

	assert(data.size() == this->m_num_elems);

	// set vals from data vector - assume same ordering
	for (int i = 0; i < this->m_num_elems; ++i)
	{
		this->m_data[i] = data[i];
	}
}

Tensor::Tensor(std::vector<int> shape, double val)
{
	assert(shape.size() > 0);

	// set m_shape, m_num_elems, and allocate data memory
	this->initialise(shape);

	for (int i = 0; i < this->m_num_elems; ++i)
	{
		this->m_data[i] = val;
	}
}

Tensor::~Tensor()
{
	delete[] this->m_data;
}

void Tensor::initialise(std::vector<int> shape)
{
	this->m_num_dims = shape.size();
	this->m_shape = shape;
	this->m_increment = std::vector<int>(this->m_num_dims);
	
	int num_elems = 1;
	for (int i = 0; i < this->m_num_dims; ++i)
	{
		assert(shape[i] > 0); // must be positive integer

		this->m_increment[this->m_num_dims - 1 - i] = num_elems;
		num_elems *= shape[this->m_num_dims - 1 - i];
	}
	this->m_num_elems = num_elems;

	this->m_data = new double[num_elems]; // allocate memory for data
}

/**********************************************************************************************************************************************************************************/

double& Tensor::operator[](int index)
{
	return this->m_data[index];
}

double& Tensor::operator[](std::vector<int> sub)
{
	assert(sub.size() == this->m_num_dims);
	return this->get_elem_by_sub(sub);
}

/**********************************************************************************************************************************************************************************/

int Tensor::get_num_dims()
{
	return this->m_num_dims;
}

std::vector<int> Tensor::get_shape()
{
	return this->m_shape;
}

double& Tensor::get_elem_by_sub(std::vector<int> sub)
{
	int index = 0;
	for (int i = 0; i < this->m_num_dims; ++i)
	{
		index += sub[i] * this->m_increment[i];
	}
	return this->m_data[index];
}

/**********************************************************************************************************************************************************************************/

// performs lambda function on all elemens in tensor, taking the element index as argument with void return type
void Tensor::lambda_by_index(std::function<void(Tensor*, const int&)> lambda)
{
	if (this->m_num_dims > 0)
	{
		int index = 0;

		std::function<void(int)> loop = [&](int dim)
		{
			for (int i = 0; i < this->m_shape[dim]; ++i)
			{
				if (dim < this->m_num_dims - 1)
				{
					loop(dim + 1);
				}
				else
				{
					lambda(this, index);
					index++;
				}
			}
		};
		loop(0);
	}
}

// performs lambda function on all elemens in tensor, taking the ptr to the tensor and the element index as arguments with void return type
void Tensor::lambda_by_sub(std::function<void(Tensor*, const std::vector<int>&)> lambda)
{
	if (this->m_num_dims > 0)
	{
		std::vector<int> sub(this->m_num_dims, 0);

		std::function<void(int)> loop = [&](int dim)
		{
			for (int i = 0; i < this->m_shape[dim]; ++i)
			{
				if (dim < this->m_num_dims - 1)
				{
					sub[dim + 1] = 0;
					loop(dim + 1);
				}
				else
				{
					lambda(this, sub);
				}
				sub[dim]++;
			}
		};
		loop(0);
	}
}

/**********************************************************************************************************************************************************************************/

// displays the tensor
void Tensor::display()
{
	int index = 0;

	// a recursive function that iterates through each dimension in a depth-first search style
	std::function<void(int)> loop = [&](int dim)
	{
		std::cout << "[ ";
		for (int i = 0; i < this->m_shape[dim]; ++i)
		{
			if (dim < this->m_num_dims - 1)
			{
				loop(dim + 1);
			}
			else
			{
				std::cout << this->m_data[index];
				index++;
			}
			if (i < this->m_shape[dim] - 1)
			{
				std::cout << ", ";
			}
		}
		std::cout << " ]";
	};
	loop(0);
	std::cout << std::endl;
}