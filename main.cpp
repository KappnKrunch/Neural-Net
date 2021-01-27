#include <iostream>
#include "neuralNet.h"

using namespace std;

int main()
{
    NeuralNet net = NeuralNet(1,{128,32,128},1);

    //net.print();

    //net.train(4000, .05);

    net.train(100000, .025);

    net.printNetworkImage();

    for(int i(0); i < 10; i++)
    {


        //net.printNetworkImage();
    }


    return 0;
}
