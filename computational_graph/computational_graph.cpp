
#include "computational_graph.hpp"



int main()
{
    double inp1, inp2, wt1, wt2, wtedSum1, wtedSum2, grad1, grad2, grad3, grad4, grad5, grad6, out;

    Double_Product prod1({&inp1, &wt1}, {&wtedSum1}, {&grad1, &grad2});
    Double_Product prod2({&inp2, &wt2}, {&wtedSum2}, {&grad3, &grad4});
    Double_Summation sum({&wtedSum1, &wtedSum2}, {&out}, {&grad5, &grad6});

    inp1 = 1.0;
    inp2 = 2.0;
    wt1 = 3.0;
    wt2 = 4.0;

    prod1.compute();
    prod2.compute();
    sum.compute();
    
    std::cout << "Output: " << out << std::endl;
    
}