#ifndef LAYERS_H
#define LAYERS_H

#include <random>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <Eigen/Dense>
#include <numeric>

using namespace Eigen;


static double sigmoid(double x);

static double dSigmoid(double x);

static double relu(double x);

static double dRelu(double x);

static double softPlus(double x);

static double dSoftPlus(double x);

static double nada(double x);

static double dNada(double x);

static MatrixXd map(MatrixXd, double (*func)(double) );

static MatrixXd maskMatrix(MatrixXd mat, MatrixXd mask, double strength);

static MatrixXd randomMatrix(int r, int c);


class InputLayer
{
public:

    MatrixXd inputs;

    MatrixXd outputs;

    int size;

    double dropoutPreservationRate;

    MatrixXd dropoutValuesForOutputs;

    InputLayer(int size);

    void feedFrom(MatrixXd newInputs);

    void printInputs();

    void printOutputs();

    void print();
};


class HiddenLayer : public InputLayer
{
public:

    MatrixXd inputWeights;

    HiddenLayer();

    HiddenLayer(int inputSize, int outputSize);

    void feedFrom(MatrixXd feed, bool withDropout);

    MatrixXd backpropogateWith(MatrixXd lastGradient);

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

    MatrixXd calculateError(MatrixXd desiredOutputs, double learningRate);
};

#endif // LAYERS_H
