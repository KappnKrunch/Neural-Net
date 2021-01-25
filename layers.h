#ifndef LAYERS_H
#define LAYERS_H
#include <random>
#include "matrix.h"


class InputLayer
{
public:

    Matrix inputs;

    Matrix outputs;

    int size;

    InputLayer(int size);

    void feedFrom(std::vector<double>& arr);

    void printInputs();

    void printOutputs();

    void print();
};


static double sigmoid(double x);

static double dSigmoid(double x);

static double relu(double x);

static double dRelu(double x);

static double softPlus(double x);

static double dSoftPlus(double x);


class HiddenLayer : public InputLayer
{
public:

    Matrix inputWeights;

    HiddenLayer();

    HiddenLayer(int inputSize, int outputSize);

    void feedFrom(Matrix feed);

    Matrix calculateErrors(Matrix desiredOutputs);

    Matrix backpropogateWith(Matrix lastGradient);

    void setAllBiases(double bias);

    void setBiases(std::vector<double> biases);

    void setRandomBiases();

    void setRandomWeights();

    void printInputWeights();

    void print();
};


class OutputLayer : public HiddenLayer
{
public:

    OutputLayer(int inputSize, int outputSize);

    void feedFrom(Matrix feed);

    Matrix backpropogateWith(Matrix desiredOutputs, double learningRate);

    Matrix backpropogateWith(std::vector<double> desiredOutputs, double learningRate);
};

#endif // LAYERS_H
