#include <iostream>
#include "neuralNet.h"
#include "matrix.h"

using namespace std;

int main()
{
    NeuralNet net = NeuralNet(1,{7,6},1);

    net.print();

    net.train(10000, .05);

    /*
    for(int i(0); i < 10; i++)
    {
        net.train(100, 0.05);

        //net.print();

        net.printExamples(7);

        cout << endl;
    }
    */

    net.printExamples(50);

    cout << endl;

    net.print();

    return 0;
}
