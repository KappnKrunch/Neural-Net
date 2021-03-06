#include "layers.h"


InputLayer::InputLayer(int size)
    :inputs(MatrixXd(size, 1)), outputs(MatrixXd(size, 1)), size(size),
     dropoutPreservationRate(1.0), dropoutValuesForOutputs(MatrixXd(size, 1))
{}

void InputLayer::feedFrom(MatrixXd newInputs)
{
    //gives information to the layer

    if(inputs.rows() != size)
    {
        std::cout << "Input error; array size different than inputs size";

        std::cout << std::endl;

        exit(1);
    }

    inputs = newInputs;

    outputs = inputs;

    //mask the outputs to impllement dropout

    //MatrixXd droppedOuts = maskMatrix(outputs, randomMatrix(size, 1), dropoutPreservationRate);

    //outputs = droppedOuts;
}

void InputLayer::printOutputs()
{
    std::cout << outputs << std::endl;
}

void InputLayer::printInputs()
{
    std::cout << inputs << std::endl;
}

void InputLayer::print()
{
    std::cout << "inputs" << std::endl;

    printInputs();

    std::cout << std::endl;

    std::cout << "outputs" << std::endl;

    printOutputs();
}









HiddenLayer::HiddenLayer()
    :HiddenLayer(0,0)
{}

HiddenLayer::HiddenLayer(int inputSize, int outputSize)
    :InputLayer(0), inputWeights(MatrixXd(outputSize, inputSize + 1))
{
    size = outputSize;

    dropoutPreservationRate = 1.0;

    dropoutValuesForOutputs = MatrixXd::Random(outputSize, 1);

    inputs = ArrayXd(inputSize + 1);

    outputs = ArrayXd(outputSize);

    setAllBiases(0);

    setRandomWeights();
}

void HiddenLayer::feedFrom(MatrixXd feed, bool withDropout)
{
    inputs = feed;
    inputs.conservativeResize(feed.rows() + 1, 1);
    inputs.coeffRef(feed.rows(), 0) = 1.0;

    outputs = inputWeights * inputs;

    outputs = map(outputs, relu);

    //mask outputs to implement dropout
    if(withDropout)
    {
        MatrixXd droppedOuts = maskMatrix(outputs, randomMatrix(size, 1), dropoutPreservationRate);

        outputs = droppedOuts;
    }
}

MatrixXd HiddenLayer::backpropogateWith(MatrixXd lastGradient)
{
    MatrixXd gradient = lastGradient.cwiseProduct(map(outputs, dRelu));

    MatrixXd deltaE = gradient * inputs.transpose();

    MatrixXd gradientOut = inputWeights.transpose() * lastGradient;

    gradientOut.conservativeResize(gradientOut.rows()-1, gradientOut.cols());


    inputWeights = inputWeights + deltaE;


    return gradientOut;
}

void HiddenLayer::setAllBiases(double bias)
{
    for(int i(0); i < size; i++)
        inputWeights.coeffRef(i, inputWeights.cols()-1) = bias;
}

void HiddenLayer::setRandomBiases()
{
    for(int i(0); i < size; i++)
        inputWeights.coeffRef(i, inputWeights.cols()-1) =
                                      1.0 - (double(rand() % 2000) / 1000);
}

void HiddenLayer::setBiases(std::vector<double> biases)
{
    for(int i(0); i < size; i++)
        inputWeights.coeffRef(i, inputWeights.cols()-1) = biases.at(i);
}

void HiddenLayer::printInputWeights()
{
    std::cout << inputWeights << std::endl;
}

void HiddenLayer::setRandomWeights()
{
    for(int i(0); i < size; i++)
    {
        for(int j(0); j < inputWeights.cols()-1; j++)
        {
            inputWeights.coeffRef(i,j) = 1.0 - (double(rand() % 2000) / 1000);
            inputWeights.coeffRef(i,j) *= ((double(rand() % 1000) / 1000));
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

    inputs.conservativeResize(size + 1, 1);

    outputs.conservativeResize(outputSize, 1);

    inputWeights.conservativeResize(outputSize, inputSize + 1);

    setAllBiases(0);

    setRandomWeights();
}

MatrixXd OutputLayer::calculateError(MatrixXd desiredOutputs, double learningRate)
{
    return ((desiredOutputs - outputs) * learningRate);
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

double softPlus(double x)
{
    return log(1 + exp(x));
}

double dSoftPlus(double x)
{
    return sigmoid(x);
}

double nada(double x)
{
    return x;
}

double dNada(double x)
{
    return 1.0;
}

MatrixXd map(MatrixXd matrix, double (*func)(double) )
{
    for(int i(0); i < matrix.rows(); i++)
        for(int j(0); j < matrix.cols(); j++)
        {
            matrix.coeffRef(i,j) = (*func)(matrix.coeff(i,j));
        }

    return matrix;
}

MatrixXd maskMatrix(MatrixXd mat, MatrixXd mask, double strength)
{
    //takes a similar matrix with values between 0-1 and a cutoff strength,
    //anything
    if(mat.rows() != mask.rows() || mat.cols() != mask.cols())
    {
        std::cout << "mask does not fit matrix" << std::endl;

        exit(1);
    }

    for(int i(0); i < mat.rows(); i++)
    {
        mat.coeffRef(i, 0) =
            ( (mask.coeff(i, 0) > (1.0-strength)) ? mat.coeff(1, 0) : 0.0 );
    }


    return mat;
}

MatrixXd randomMatrix(int r, int c)
{
    //random matrix between 0-1

    MatrixXd dropoutValuesForOutputs = MatrixXd::Random(r, c);

    dropoutValuesForOutputs = (dropoutValuesForOutputs + MatrixXd::Constant(r, c, 1.0))/2.0;


    return dropoutValuesForOutputs;
}
