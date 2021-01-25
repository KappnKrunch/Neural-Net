#include "layers.h"






InputLayer::InputLayer(int size)
    :inputs(Matrix(size,1)), outputs(Matrix(size,1)), size(size)
{}

void InputLayer::feedFrom(std::vector<double>& arr)
{
    //gives information to the layer

    if(arr.size() != size)
    {
        std::cout << "Input error; array size different than inputs size";

        std::cout << std::endl;

        exit(1);
    }

    //for an input layer, the outputs are almost exactly the inputs
    for(int i(0); i < size; i++)
    {
        inputs.at(i,0) = 1.0 * arr[i];
        outputs.at(i,0) = 1.0 * arr[i];
    }
}

void InputLayer::printOutputs()
{
    outputs.print();
}

void InputLayer::printInputs()
{
    inputs.print();
}

void InputLayer::print()
{
    std::cout << "inputs" << std::endl;

    printInputs();

    std::cout << std::endl;

    std::cout << "outputs" << std::endl;

    printOutputs();
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double dSigmoid(double x)
{
    return exp(-x) * sigmoid(x) * sigmoid(x);
}

double relu(double x)
{
    return (x >= 0 ? x : 0);
}

double dRelu(double x)
{
    return (x > 0 ? 1 : 0);
}

static double softPlus(double x)
{
    return log(1 + exp(x));
}

static double dSoftPlus(double x)
{
    return sigmoid(x);
}


HiddenLayer::HiddenLayer()
    :HiddenLayer(0,0)
{}

HiddenLayer::HiddenLayer(int inputSize, int outputSize)
    :InputLayer(0),inputWeights(Matrix(outputSize, inputSize + 1))
{
    size = outputSize;

    inputs = MatrixMath::addRow(Matrix(inputSize, 1), 1.0);

    outputs = MatrixMath::addRow(Matrix(outputSize, 1), 1.0);

    //setAllBiases(1.0);

    //setRandomBiases();

    setRandomWeights();

    //inputWeights.at(layerSize, inputs.size) = 1;
}

void HiddenLayer::feedFrom(Matrix feed)
{
    inputs = MatrixMath::addRow(feed, 1.0);

    outputs = MatrixMath::multiply(inputWeights, inputs);

    outputs = MatrixMath::map(outputs, relu);
}

Matrix HiddenLayer::backpropogateWith(Matrix lastGradient)
{
    Matrix gradient = MatrixMath::hadamard(lastGradient, MatrixMath::map(outputs, dRelu));

    Matrix deltaE = MatrixMath::multiply(gradient, MatrixMath::transpose(inputs));

    Matrix gradientOut = MatrixMath::multiply(MatrixMath::transpose(inputWeights), lastGradient);

    gradientOut = MatrixMath::rowPopBack(gradientOut);


    inputWeights = MatrixMath::add(inputWeights, deltaE);


    return gradientOut;
}

void HiddenLayer::setAllBiases(double bias)
{
    for(int i(0); i < size; i++)
        inputWeights.at(i,inputWeights.getWidth()-1) = bias;
}

void HiddenLayer::setRandomBiases()
{
    for(int i(0); i < size; i++)
        inputWeights.at(i, inputWeights.getWidth()-1) =
                                      1.0 - (double(rand() % 2000) / 1000);
}

void HiddenLayer::setBiases(std::vector<double> biases)
{
    for(int i(0); i < size; i++)
        inputWeights.at(i,inputWeights.getWidth()-1) = biases.at(i);
}

void HiddenLayer::printInputWeights()
{
    inputWeights.print();
}

void HiddenLayer::setRandomWeights()
{
    for(int i(0); i < size; i++)
    {
        for(int j(0); j < inputWeights.getWidth()-1; j++)
        {
            inputWeights.at(i,j) = 1.0 - (double(rand() % 2000) / 1000);
            //inputWeights.at(i,j) *= (1.0 - (double(rand() % 2000) / 1000));
        }
    }
}

void HiddenLayer::print()
{
    std::cout << "inputs" << std::endl;

    printInputs();

    std::cout << std::endl;

    std::cout << "input weights " << std::endl;

    printInputWeights();

    std::cout << std::endl;

    std::cout << "outputs" << std::endl;

    printOutputs();
}









OutputLayer::OutputLayer(int inputSize, int outputSize)
    :HiddenLayer(0, 0)
{
    size = outputSize;

    inputs = MatrixMath::addRow(Matrix(inputSize, 1), 1.0);

    outputs = Matrix(outputSize, 1);

    inputWeights = Matrix(outputSize, inputSize + 1);

    //setAllBiases(1.0);

    //setRandomBiases();

    setRandomWeights();
}

void OutputLayer::feedFrom(Matrix feed)
{
    inputs = MatrixMath::addRow(feed, 1.0);

    outputs = MatrixMath::multiply(inputWeights, inputs);

    outputs = MatrixMath::map(outputs, relu);
}

Matrix OutputLayer::backpropogateWith(Matrix desiredOutputs, double learningRate)
{
    Matrix errors = MatrixMath::subtract(desiredOutputs, outputs);

    errors = MatrixMath::multiply(errors, learningRate);

    Matrix gradient = MatrixMath::hadamard(errors, MatrixMath::map(outputs, dRelu));

    Matrix deltaE = MatrixMath::multiply(gradient, MatrixMath::transpose(inputs));



    Matrix gradientOut = MatrixMath::multiply(MatrixMath::transpose(inputWeights), errors);

    gradientOut = MatrixMath::rowPopBack(gradientOut);


    inputWeights = MatrixMath::add(inputWeights, deltaE);

    //std::cout << "outs" << std::endl;

    //gradientOut.print();


    return gradientOut;
}

Matrix OutputLayer::backpropogateWith(std::vector<double> desiredOutputs, double learningRate)
{
    return OutputLayer::backpropogateWith(MatrixMath::fromArray(desiredOutputs), learningRate);
}
