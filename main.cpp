#include <iostream>
#include "neuralNet.h"

using namespace std;

int main()
{
    NeuralNet net = NeuralNet(1,{256,256},1);

    net.train(10000, .03);

    net.print1x1NetworkImage("Gen " + std::to_string(net.iteration));


    return 0;
}
