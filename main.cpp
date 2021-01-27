#include <iostream>
#include "neuralNet.h"

using namespace std;

int main()
{
    NeuralNet net = NeuralNet(1,{32,64,32},1);

    net.train(1000000, .025);

    net.printNetworkImage();


    return 0;
}
