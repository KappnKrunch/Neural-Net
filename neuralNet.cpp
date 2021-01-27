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
    MatrixXd delta = outputLayer.backpropogateWith(desiredOutput, learningRate);

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

        std::cout << trueData.first << std::endl;


        std::cout << "True Outputs" << std::endl;

        std::cout << trueData.second << std::endl;


        std::cout << "Guessed Outputs" << std::endl;

        std::cout << outputLayer.outputs.transpose() << std::endl;

        std::cout << std::endl;
    }
}

void NeuralNet::printNetworkImage()
{
    std::ofstream image("Neural_Net_Gen"+std::to_string(iteration)+".ppm");

    int res = 256;

    std::cout << "Printing image.." << std::endl;

    //
    std::vector<std::vector<Color>> baseImage;

    baseImage.resize(res);

    for(int i(0); i < res; i++)
        baseImage.at(i).resize(res, Color{254,254,254});

    for(int i(0); i < res; i++)
    {
        int y1 = ceil(randomWave(double(i)/res) * double(res));
        y1 = std::min(std::max(y1,0),res-1);

        int y2 = floor(randomWave(double(i)/res) * double(res));
        y2 = std::min(std::max(y2,0),res-1);

        baseImage[y1][i] = Color{50,100,200};
        baseImage[y2][i] = Color{50,100,200};
    }

    for(int i(0); i < res; i++)
    {
        feedForward({(double(i)/res)});
        int y = int(outputLayer.outputs.coeff(0,0) * double(res));

        y = std::max(std::min(y, res-1), 0);

        baseImage[y][i] = Color{0,0,0};
    }

    image << "P3" << std::endl;
    image << res << " " << res << std::endl;
    image << "255" << std::endl;


    for(int y(0); y < res; y++)
    {
        for(int x(0); x < res; x++)
        {
            image << baseImage[y][x].r << " " << baseImage[y][x].g << " " << baseImage[y][x].b << std::endl;
        }
    }

    image.close();

    std::cout << "Done!" << std::endl;
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

    inputs.resize(inputLayer.inputs.rows());

    outputs.resize(outputLayer.outputs.rows());

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

double NeuralNet::normalSinWave(double x)
{
    return sin(x * 2.0 * 3.15159)*0.5 + 0.5;
}

double NeuralNet::randomWave(double x)
{
    double out(0);

    std::vector<double> coefs{1,0,0,1,0,0,1,0,0,1};

    for(int i(0); i < coefs.size(); i++)
    {
        out += coefs[i] * normalSinWave(x * double(i));
    }

    return (out / coefs.size());
}


TrainingData NeuralNet::sinXExample()
{
    std::vector<double> inputs;
    std::vector<double> outputs;

    double randPi = double(rand() % 10000)/ 10000;

    inputs.push_back(randPi);

    outputs.push_back(normalSinWave(randPi));


    return {inputs, outputs};
}

TrainingData NeuralNet::waveExample()
{
    std::vector<double> inputs;
    std::vector<double> outputs;

    double randPi = double(rand() % 10000)/ 10000;

    inputs.push_back(randPi);

    outputs.push_back(randomWave(randPi));


    return {inputs, outputs};
}

TrainingData NeuralNet::generateTrainingData()
{
    TrainingData data = waveExample();

    if(inputLayer.inputs.rows() != data.first.size() ||
       outputLayer.outputs.rows() != data.second.size() )
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
