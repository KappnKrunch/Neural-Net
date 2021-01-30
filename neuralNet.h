#ifndef NEURALNET_H
#define NEURALNET_H

#include "layers.h"
#include <fstream>
#include <string>

typedef std::pair<std::vector<double>, std::vector<double>> TrainingData;

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

    void trainStochastically(int sampleSize, int iterations, double learningRate);

    void feedForward(std::vector<double> inputs);

    void backpropagate(std::vector<double> desiredOutput);

    void print();

    void printExamples(int examples);

    void print1x1NetworkImage(std::string name);

    TrainingData generateTrainingData();

    TrainingData xORExample();

    TrainingData fakeExample();

    TrainingData powXExample();

    TrainingData sinXExample();

    TrainingData waveExample();

    double normalSinWave(double x);

    double randomWave(double x);
};

template <typename T>
std::ostream & operator << (std::ostream & output, const std::vector<T> & vector);

struct Color{int r,g,b;};

#endif // NEURALNET_H
