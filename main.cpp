#include <iostream>
#include <time.h>
#include "neuralNet.h"

using namespace std;

int main()
{
    NeuralNet net = NeuralNet(1, {10,20,10}, 1);

    time_t startTime = time(0);

    cout << "t" << endl;


    net.trainStochastically(100, 10000, .01);

    //net.train(10000, .1);

    net.print();

    net.print1x1NetworkImage("sin 1x1 " + std::to_string(net.iteration));


    cout << "took " << difftime(time(0), startTime) << " seconds" << endl;


    return 0;
}
