
#include "mnist.hpp"

#include "../../neural_net.hpp"
#include "../../../computational_graph/mlcg.hpp"

/*

    MNIST Test for neural network implementations
        - direction implementation using std::vector 
        - implementation using computational graph modules in mlcg.hpp

    Key result:
        - Both modules produce the SAME results given the SAME random seed initialisation 
        - This can be proven by switching the ordering of the testing to see that the same results are produced.

*/

// test for first implementation of neural network in neural_network.hpp
void NN_1_TEST(const MNIST& mnist)
{
    std::cout << "DIRECT IMPLEMENTATION NN MODEL" << std::endl;
    gv::neural_network nn({28*28, 30, 10}, {"relu", "softmax"}, "crossentropy");

    // train model
    nn.train(mnist.m_train_x, mnist.m_train_y, 0.01, 1, 1);

    // evaluate model for samples from each number 0,...,9
    int ind = 0;
    for (int i = 0; i < 10000; ++i)
    {
        if (ind == 10)
            break;
        
        if (mnist.m_test_y[i][ind] == 1)
        {
            auto h = nn.eval(mnist.m_test_x[i]);
            std::cout << "Probability of success in test for digit (" << ind << "): " << (h[ind])*100 << "%" << std::endl;
            ++ind;
        }
    }
}

// test for computational graph implementation of neural network using elements from mlcg.hpp
void NN_COMP_GRAPH_TEST(const MNIST& mnist)
{
    std::cout << "COMPUTATIONAL GRAPH NN MODEL" << std::endl;
    gv::mlcg::dense_lyr     lyr1;
    gv::mlcg::dense_lyr     lyr2;
    gv::mlcg::crossentropy  loss;

    lyr1.init({28*28, 30}, "relu");
    lyr2.init({30, 10}, "softmax");
    loss.init(10);

    // connect output to hidden1
    gv::mlcg::dense_lyr::connect(&lyr1, &lyr2);

    // connect loss to output
    lyr2.add_loss(&loss);

    for (int i = 0; i < 60000; ++i)
    {
        lyr1.set_vec_input(&(mnist.m_train_x[i]));
        loss.set_truth(&(mnist.m_train_y[i]));

        // forward pass
        lyr1.comp();
        lyr2.comp();
        loss.comp();

        // backpropagation
        loss.grad();
        lyr2.grad();
        lyr1.grad();
        
        lyr1.grad_descent(0.01);
        lyr2.grad_descent(0.01);
    }

    // evaluate model for samples from each number 0,...,9
    int ind = 0;
    for (int i = 0; i < 10000; ++i)
    {
        if (ind == 10)
            break;
        
        if (mnist.m_test_y[i][ind] == 1)
        {
            lyr1.set_vec_input(&(mnist.m_test_x[i]));
            loss.set_truth(&(mnist.m_test_y[i]));
            lyr1.comp();
            lyr2.comp();
            loss.comp();
            std::cout << "Probability of success in test for digit (" << ind << "): " << (lyr2.p_act->m_y[ind])*100 << "%" << std::endl;
            ++ind;
        }
    }
}

int main()
{
    MNIST mnist("./data/train-images.idx3-ubyte", 
        "./data/train-labels.idx1-ubyte",
        "./data/t10k-images.idx3-ubyte",
        "./data/t10k-labels.idx1-ubyte");

    NN_1_TEST(mnist);

    NN_COMP_GRAPH_TEST(mnist);
    
    

}