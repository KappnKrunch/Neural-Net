#ifndef NEURALNET_H
#define NEURALNET_H

#include "layers.h"
#include <fstream>
#include <string>

typedef std::pair<MatrixXd, MatrixXd> TrainingData;

class NeuralNet
{
public:
    InputLayer inputLayer;
    std::vector<HiddenLayer> hiddenLayers;
    OutputLayer outputLayer;

    double learningRate;
    int iteration;

public:

    NeuralNet(int inputSize, int hiddenSize, int outputSize);

    NeuralNet(int inputSize, std::vector<int> hiddenSize, int outputSize);

    void train(int iterations, double learningRate);

    void trainStochastically(int sampleSize, int sampleCount, double learningRate);

    void feedForward(MatrixXd inputs);

    void backpropagate(MatrixXd desiredOutput);

    void setDropoutPresevervationRates(double allLayersRate);

    void setDropoutPresevervationRates(std::vector<double> layerRates);

    std::vector<double> getDropoutPresevervationRates();

    void print();

    void printExamples(int examples);

    void print1x1NetworkImage(std::string name);

    void print2x1NetworkImage(std::string name);

    TrainingData xORExample();

    TrainingData sinExample();
};

struct Color{int r,g,b;};

#endif // NEURALNET_H
