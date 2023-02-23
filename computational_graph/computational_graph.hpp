#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <assert.h>

#include "../../mathlib/linear_algebra.hpp"

// Operation

template <class T, class U>
class Operation
{
protected:
	int m_num_inputs;
	int m_num_outputs;
	T** m_inputs;
	U** m_outputs;

public:
	Operation();
	Operation(std::vector<T*> inputs, std::vector<U*> outputs);
	~Operation();

	int get_num_inputs();
	int get_num_outputs();

	virtual void compute();
};

// default ctor for operation
template <class T, class U>
Operation<T, U>::Operation()
{
}

// param ctor for operation
template <class T, class U>
Operation<T, U>::Operation(std::vector<T*> inputs, std::vector<U*> outputs)
{
	assert(inputs.size() > 0);
	assert(outputs.size() > 0);

	this->m_num_inputs = inputs.size();
	this->m_num_outputs = outputs.size();
	this->m_inputs = new T*[this->m_num_inputs]; // allocate memory for input ptrs
	this->m_outputs = new U*[this->m_num_outputs]; // allocate memory for output ptrs
	for (int i = 0; i < this->m_num_inputs; ++i)
	{
		assert(inputs[i] != nullptr);
		this->m_inputs[i] = inputs[i];
	}
	for (int i = 0; i < this->m_num_outputs; ++i)
	{
		assert(outputs[i] != nullptr);
		this->m_outputs[i] = outputs[i];
	}
}

// dtor for operation
template <class T, class U>
Operation<T, U>::~Operation()
{
	delete[] this->m_inputs;
	delete[] this->m_outputs;
}

// returns number of inputs
template <class T, class U>
int Operation<T, U>::get_num_inputs()
{
	return this->m_num_inputs;
}

// returns number of outputs
template <class T, class U>
int Operation<T, U>::get_num_outputs()
{
	return this->m_num_outputs;
}

// compute operation - must override in derived class
template <class T, class U>
void Operation<T, U>::compute()
{
}


/**********************************************************************************************************************************************************************************/
// Differentiable operation

template <class T>
class Differentiable_Operation : public Operation<T, T>
{
protected:
	T** m_grads;

public:
	Differentiable_Operation();
	Differentiable_Operation(std::vector<T*> inputs, std::vector<T*> outputs, std::vector<T*> grads);
	~Differentiable_Operation();

	virtual void compute_grad();
};

// default ctor for differentiable operation
template <class T>
Differentiable_Operation<T>::Differentiable_Operation() : Operation<T, T>()
{
}

// param ctor for differentiable operation
template <class T>
Differentiable_Operation<T>::Differentiable_Operation(std::vector<T*> inputs, std::vector<T*> outputs, std::vector<T*> grads) : Operation<T, T>(inputs, outputs)
{
	assert(inputs.size() == grads.size());

	this->m_grads = new T*[this->m_num_inputs]; // allocate memory for gradient ptrs
	for (int i = 0; i < this->m_num_inputs; ++i)
	{
		assert(grads[i] != nullptr);
		this->m_grads[i] = grads[i];
	}
}

// dtor for differentiable operation
template <class T>
Differentiable_Operation<T>::~Differentiable_Operation()
{
	delete[] this->m_grads;
}

// compute gradient operation - must override in derived class
template <class T>
void Differentiable_Operation<T>::compute_grad()
{
}

/**********************************************************************************************************************************************************************************/

class Double_Summation : public Differentiable_Operation<double>
{
public:
	Double_Summation();
	Double_Summation(std::vector<double*> inputs, std::vector<double*> outputs, std::vector<double*> grads);

	void compute();
	void compute_grad();
};

// default ctor
Double_Summation::Double_Summation() : Differentiable_Operation<double>()
{
}

// param ctor
Double_Summation::Double_Summation(std::vector<double*> inputs, std::vector<double*> outputs, std::vector<double*> grads) : Differentiable_Operation<double>(inputs, outputs, grads)
{
}

// compute operation
void Double_Summation::compute()
{
	double result = 0.0;
	for (int i = 0; i < this->m_num_inputs; ++i)
	{
		result += *(this->m_inputs[i]);
	}
	for (int i = 0; i < this->m_num_outputs; ++i)
	{
		*(this->m_outputs[i]) = result;
	}
}

// compute gradients operation
void Double_Summation::compute_grad()
{
	for (int i = 0; i < this->m_num_inputs; ++i)
	{
		*(this->m_grads[i]) = 1.0;
	}
}

/**********************************************************************************************************************************************************************************/

class Double_Product : public Differentiable_Operation<double>
{
public:
	Double_Product();
	Double_Product(std::vector<double*> inputs, std::vector<double*> outputs, std::vector<double*> grads);

	void compute();
	void compute_grad();
};

// default ctor
Double_Product::Double_Product() : Differentiable_Operation<double>()
{
}

// param ctor
Double_Product::Double_Product(std::vector<double*> inputs, std::vector<double*> outputs, std::vector<double*> grads) : Differentiable_Operation<double>(inputs, outputs, grads)
{
}

// compute operation
void Double_Product::compute()
{
	double result = 1.0;
	for (int i = 0; i < this->m_num_inputs; ++i)
	{
		result *= *(this->m_inputs[i]);
	}
	for (int i = 0; i < this->m_num_outputs; ++i)
	{
		*(this->m_outputs[i]) = result;
	}
}

// compute gradients operation
void Double_Product::compute_grad()
{
	for (int i = 0; i < this->m_num_inputs; ++i)
	{
		double result = 1.0;
		for (int j = 0; j < this->m_num_inputs; ++j)
		{
			if (i != j)
			{
				result *= *(this->m_inputs[i]);
			}
			*(this->m_grads[i]) = result;
		}
	}
}

/**********************************************************************************************************************************************************************************/

class Double_Square : public Differentiable_Operation<double>
{
public:
	Double_Square();
	Double_Square(std::vector<double*> inputs, std::vector<double*> outputs, std::vector<double*> grads);

	void compute();
	void compute_grad();

};

Double_Square::Double_Square() : Differentiable_Operation<double>()
{
}

Double_Square::Double_Square(std::vector<double*> inputs, std::vector<double*> outputs, std::vector<double*> grads) : Differentiable_Operation<double>(inputs, outputs, grads)
{
	assert(inputs.size() == 1); // square function only takes one input
}

void Double_Square::compute()
{
	double result = *(this->m_inputs[0]);
	result *= result;
	for (int i = 0; i < this->m_num_outputs; ++i)
	{
		*(this->m_outputs[i]) = result; 
	}
}

void Double_Square::compute_grad()
{
	*(this->m_grads[0]) = *(this->m_inputs[0]) * 2.0;
}

/**********************************************************************************************************************************************************************************/

class Double_Sigmoid : public Differentiable_Operation<double>
{
public:
	Double_Sigmoid();
	Double_Sigmoid(std::vector<double*> inputs, std::vector<double*> outputs, std::vector<double*> grads);

	void compute();
	void compute_grad();

};

Double_Sigmoid::Double_Sigmoid() : Differentiable_Operation<double>()
{
}

Double_Sigmoid::Double_Sigmoid(std::vector<double*> inputs, std::vector<double*> outputs, std::vector<double*> grads) : Differentiable_Operation<double>(inputs, outputs, grads)
{
	assert(inputs.size() == 1); // sigmoid function only takes one input
}

void Double_Sigmoid::compute()
{
	double result = *(this->m_inputs[0]);
	result *= result;
	for (int i = 0; i < this->m_num_outputs; ++i)
	{
		*(this->m_outputs[i]) = result; 
	}
}

void Double_Sigmoid::compute_grad()
{
	*(this->m_grads[0]) = *(this->m_inputs[0]) * 2.0;
}

/**********************************************************************************************************************************************************************************/

/*
class Matrix_Summation : public Differentiable_Operation<mathlib::Matrix>
{
public:
	Matrix_Summation();
	Matrix_Summation(std::vector<mathlib::Matrix*> inputs, std::vector<mathlib::Matrix*> outputs, std::vector<mathlib::Matrix*> grads);

	void compute();
	void compute_grad();
};

Matrix_Summation::Matrix_Summation() : Differentiable_Operation<mathlib::Matrix>()
{
}

Matrix_Summation::Matrix_Summation(std::vector<mathlib::Matrix*> inputs, std::vector<mathlib::Matrix*> outputs, std::vector<mathlib::Matrix*> grads) : Differentiable_Operation<mathlib::Matrix>(inputs, outputs, grads)
{

}

void Matrix_Summation::compute()
{

}

void Matrix_Summation::compute_grad()
{

}
*/