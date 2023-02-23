
#include "computational_graph.hpp"

#include "tensor.hpp"

int main()
{
    Tensor tensor({2, 4, 3}, 0.5);

    
    tensor[{0, 1, 2}] = 2.0;
    tensor[{1,0,0}] = 3.0;
    tensor.display();

    std::cout << std::endl;

    tensor.lambda_by_sub([](Tensor* tensor, const std::vector<int>& sub)
    {
        std::cout << "Sub: ";
        
        for (int i = 0; i < sub.size(); ++i)
        {
            std::cout << sub[i] << ", ";
        }
        std::cout << std::endl;
    });

}