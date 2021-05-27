#include <iostream>
#include <time.h>
#include "neuralNet.h"

using namespace std;

int main()
{
    NeuralNet net = NeuralNet(1, {128,32,64}, 1);

    time_t startTime = time(nullptr);

    int batches = 100;


    for(int i(0); i < batches; i++)
    {
        cout << "batch " << i + 1 << "/" << batches <<endl;

        //net.trainStochastically(100, 1000, .001);

        //net.setDropoutPresevervationRates(1.0);

        net.train(5000, 0.01);

        net.print1x1NetworkImage("sin 1x1 " + std::to_string(net.iteration));
    }

    net.print();


    cout << "took " << difftime(time(nullptr), startTime) << " seconds" << endl;


    return 0;
}
