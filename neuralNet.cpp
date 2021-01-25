#include "neuralNet.h"



NeuralNet::NeuralNet(int inputSize, int hiddenSize, int outputSize)
    :inputLayer(inputSize),
      hiddenLayers(0),
     outputLayer(hiddenSize, outputSize),
     iteration(0),learningRate(0.01)
{
    std::vector<double> initialFeed;

    hiddenLayers.push_back(HiddenLayer(inputSize, hiddenSize));

    initialFeed.resize(inputSize, 0.0);

    feedForward(initialFeed);
}

NeuralNet::NeuralNet(int inputSize, std::vector<int> hiddenSize, int outputSize)
    :inputLayer(inputSize),
      hiddenLayers(hiddenSize.size()),
     outputLayer(hiddenSize[hiddenSize.size()-1], outputSize),
     iteration(0),learningRate(0.01)
{
    std::vector<double> initialFeed;

    hiddenLayers[0] = HiddenLayer(inputSize, hiddenSize[0]);

    for(int i(1); i < hiddenSize.size(); i++)
        hiddenLayers[i] = HiddenLayer(hiddenSize[i-1], hiddenSize[i]);


    initialFeed.resize(inputSize, 0.0);

    feedForward(initialFeed);
}

void NeuralNet::feedForward(std::vector<double> inputs)
{
    inputLayer.feedFrom(inputs);

    hiddenLayers[0].feedFrom(inputLayer.outputs);

    for(int i(1); i < hiddenLayers.size(); i++)
    {
        hiddenLayers[i].feedFrom(hiddenLayers[i-1].outputs);
    }


    outputLayer.feedFrom(hiddenLayers[hiddenLayers.size() - 1].outputs);
}

void NeuralNet::backpropagate(std::vector<double> desiredOutput)
{
    Matrix delta = outputLayer.backpropogateWith(desiredOutput, learningRate);

    for(int i(0); i < hiddenLayers.size(); i++)
    {
        delta = hiddenLayers[hiddenLayers.size() - i -1].backpropogateWith(delta);
    }

    iteration++;
}

void NeuralNet::train(int iterations, double learningRate)
{
    std::cout << "Training from " << iteration << " to " << iteration + iterations << std::endl;
    TrainingData data;

    this->learningRate = learningRate;

    for(int i(0); i < iterations; i++)
    {
        data = generateTrainingData();

        feedForward(data.first);

        backpropagate(data.second);
    }

    feedForward(data.first);
}

void NeuralNet::print()
{
    std::cout << "Neural Net gen " << iteration << std::endl;
    std::cout << "Input Layer" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    inputLayer.print();

    std::cout << "-----------------------------------" << std::endl;
    std::cout << std::endl;


    for(int i(0); i < hiddenLayers.size(); i++)
    {
        std::cout << "Hidden Layer "<< i << std::endl;
        std::cout << "-----------------------------------" << std::endl;

        hiddenLayers[i].print();

        std::cout << "-----------------------------------" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "Outputs Layer" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    outputLayer.print();

    std::cout << "-----------------------------------" << std::endl;
    std::cout << std::endl;
}

void NeuralNet::printExamples(int examples)
{
    for(int i(0); i < examples; i++)
    {
        TrainingData trueData = generateTrainingData();


        feedForward(trueData.first);


        std::cout << "Inputs " << std::endl;

        MatrixMath::transpose(MatrixMath::fromArray(trueData.first)).print();


        std::cout << "True Outputs" << std::endl;

        MatrixMath::transpose(MatrixMath::fromArray(trueData.second)).print();


        std::cout << "Guessed Outputs" << std::endl;

        MatrixMath::transpose(outputLayer.outputs).print();

        std::cout << std::endl;
    }
}

TrainingData NeuralNet::xORExample()
{

    std::vector<double> inputs;

    std::vector<double> outputs;


    inputs.push_back( rand() % 2 );
    inputs.push_back( rand() % 2 );


    outputs.push_back((inputs[0] || inputs[1]) && (inputs[0] != inputs[1]));


    return {inputs, outputs};
}

TrainingData NeuralNet::fakeExample()
{
    std::vector<double> inputs;

    std::vector<double> outputs;

    inputs.resize(inputLayer.inputs.getHeight());

    outputs.resize(outputLayer.outputs.getHeight());

    for(int i(0); i < inputs.size(); i++)
        inputs.at(i) = rand() % 2;

    for(int i(0); i < outputs.size(); i++)
        outputs.at(i) = rand() % 2;

    return {inputs, outputs};
}


TrainingData NeuralNet::powXExample()
{
    std::vector<double> inputs;
    std::vector<double> outputs;

    inputs.push_back(rand() % 6);

    inputs[0] /= 5;

    outputs.push_back(pow(2.0,inputs[0]));

    outputs[0] /= 2*2*2*2*2;



    return {inputs, outputs};
}


TrainingData NeuralNet::sinXExample()
{
    std::vector<double> inputs;
    std::vector<double> outputs;

    double randPi = double(rand() % (2*314159));
    randPi /= 100000;

    inputs.push_back(randPi);

    outputs.push_back((sin(randPi)*0.5 + 0.5));

    //outputs[0] /= 2*2*2*2*2;



    return {inputs, outputs};
}

TrainingData NeuralNet::generateTrainingData()
{
    TrainingData data = sinXExample();

    if(inputLayer.inputs.getHeight() != data.first.size() ||
       outputLayer.outputs.getHeight() != data.second.size() )
    {
        std::cout << "The example does not work with this network configuration";

        std::cout << std::endl;

        exit(1);
    }

    return data;
}


template <typename T>
std::ostream & operator << (std::ostream & output, const std::vector<T> & vector)
{
    for(int i(0); i < vector.size(); i++)
    {
        output << vector[i] << ((i < vector.size() - 1)? ", " : "");
    }

    return output;
}
