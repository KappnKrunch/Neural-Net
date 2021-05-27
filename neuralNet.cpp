#include "neuralNet.h"


NeuralNet::NeuralNet(int inputSize, int hiddenSize, int outputSize)
    :inputLayer(inputSize),
      hiddenLayers({HiddenLayer(inputSize,hiddenSize)}),
     outputLayer(hiddenSize, outputSize),
     iteration(0),learningRate(0.01)
{
    MatrixXd initialFeed = MatrixXd(inputSize,1);

    setDropoutPresevervationRates(1.0);

    initialFeed.setZero();

    feedForward(initialFeed);
}

NeuralNet::NeuralNet(int inputSize, std::vector<int> hiddenSize, int outputSize)
    :inputLayer(inputSize),
     hiddenLayers(hiddenSize.size()),
     outputLayer(hiddenSize[hiddenSize.size()-1], outputSize),
     iteration(0),learningRate(0.01)
{
    MatrixXd initialFeed = MatrixXd(inputSize,1);

    hiddenLayers[0] = HiddenLayer(inputSize, hiddenSize[0]);

    for(int i(1); i < hiddenSize.size(); i++)
        hiddenLayers[i] = HiddenLayer(hiddenSize[i-1], hiddenSize[i]);

    initialFeed.setZero();

    feedForward(initialFeed);
}




void NeuralNet::feedForward(MatrixXd inputs)
{
    inputLayer.feedFrom(inputs);

    //first hidden layer
    hiddenLayers[0].feedFrom(inputLayer.outputs, false);

    for(int i(1); i < hiddenLayers.size(); i++)
    {
        hiddenLayers[i].feedFrom(hiddenLayers[i-1].outputs, false);
    }

    outputLayer.feedFrom(hiddenLayers[hiddenLayers.size() - 1].outputs, false);
}

void NeuralNet::backpropagate(MatrixXd desiredOutput)
{
    MatrixXd delta = outputLayer.calculateError(desiredOutput, learningRate);

    delta = outputLayer.backpropogateWith(delta);

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
        data = sinExample();

        feedForward(data.first);

        backpropagate(data.second);
    }

    feedForward(data.first);
}

void NeuralNet::trainStochastically(int sampleSize, int sampleCount, double learningRate)
{
    std::cout << "Training stochastically... " << std::endl;

    TrainingData data;

    MatrixXd summedError = MatrixXd(outputLayer.outputs.rows(), 1);



    this->learningRate = learningRate;

    for(int i(0); i < sampleCount; i++)
    {
        summedError = summedError.setZero();

        for(int j(0); j < sampleSize; j++)
        {
            data = sinExample();

            feedForward(data.first);

            summedError += outputLayer.calculateError(data.second, learningRate);
        }

        summedError /= sampleSize;

        backpropagate(data.second);
    }


    feedForward(data.first);

    std::cout << "Finished training." << std::endl;
}

void NeuralNet::setDropoutPresevervationRates(double allLayersRate)
{
    inputLayer.dropoutPreservationRate = allLayersRate;

    for(int i(0); i < hiddenLayers.size(); i++)
        hiddenLayers[i].dropoutPreservationRate = allLayersRate;
}

void NeuralNet::setDropoutPresevervationRates(std::vector<double> layerRates)
{
    if(layerRates.size() < hiddenLayers.size() + 1)
    {
        std::cout << "error dropout rates vector missized" << std::endl;
        exit(1);
    }

    inputLayer.dropoutPreservationRate = layerRates[0];

    for(int i(0); i < hiddenLayers.size(); i++)
        hiddenLayers[i].dropoutPreservationRate = layerRates[i + 1];
}


std::vector<double> NeuralNet::getDropoutPresevervationRates()
{
    std::vector<double> outRates;

    outRates.resize(hiddenLayers.size()+1);

    outRates[0] = inputLayer.dropoutPreservationRate;

    for(int i(0); i < outRates.size()-1; i++)
        outRates[i+1] = hiddenLayers[i].dropoutPreservationRate;

    return outRates;
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
        TrainingData trueData = xORExample();


        feedForward(trueData.first);


        std::cout << "Inputs " << std::endl;

        std::cout << trueData.first.transpose() << std::endl;


        std::cout << "True Outputs" << std::endl;

        std::cout << trueData.second.transpose() << std::endl;


        std::cout << "Guessed Outputs" << std::endl;

        std::cout << outputLayer.outputs.transpose() << std::endl;

        std::cout << std::endl;
    }
}


TrainingData NeuralNet::xORExample()
{
    MatrixXd inputs = MatrixXd(2,1);
    MatrixXd outputs = MatrixXd(1,1);


    inputs.coeffRef(0,0) = rand() % 2;
    inputs.coeffRef(1,0) = rand() % 2;


    outputs.coeffRef(0,0) = double((inputs.coeff(0,0) || inputs.coeff(1,0)) &&
            (inputs.coeff(0,0) != inputs.coeff(1,0)));


    return {inputs, outputs};
}

TrainingData NeuralNet::sinExample()
{
    MatrixXd inputs = MatrixXd(1,1);
    MatrixXd outputs = MatrixXd(1,1);


    inputs.coeffRef(0,0) = (double(rand() % 10000) / 10000);

    outputs.coeffRef(0,0) = 0.5*sin(inputs.coeff(0,0) * 6.0 * 3.14159) + 0.5;


    return {inputs, outputs};
}



void NeuralNet::print1x1NetworkImage(std::string name)
{
    std::ofstream image(name + ".ppm");
    int res = 256;
    std::vector<std::vector<Color>> baseImage;

    std::vector<double> oldDropRates = getDropoutPresevervationRates();
    std::vector<double> newDropRates;

    newDropRates.resize(oldDropRates.size(), 1.0);
    setDropoutPresevervationRates(newDropRates);

    std::cout << "Printing image.." << std::endl;

    baseImage.resize(res+1);

    int y;
    int lastHeight = 0;

    for(int i(0); i < res+1; i++)
        baseImage.at(i).resize(res+1, Color{254,254,254});

    for(int i(0); i < res; i++)
    {
        y = ceil( (0.5*sin(6.0 * 3.14159 * double(i)/res) + 0.5) * double(res));
        y = std::min(std::max(y, 0), res-1);

        baseImage[y][i] = Color{50, 100, 200};

        lastHeight = y;
    }

    MatrixXd yMatrix = MatrixXd(1,1);    

    for(int i(0); i < res; i++)
    {
        yMatrix.coeffRef(0, 0) = (double(i)/res);

        feedForward(yMatrix);

        y = int(outputLayer.outputs.coeff(0, 0) * double(res));
        y = std::max(std::min(y, res-1), 0);

        lastHeight = i != 0? lastHeight : y;

        for(int j(std::min(lastHeight, y)); j <= std::max(lastHeight, y); j++)
        {
            baseImage[j][i] = Color{0, 0, 0};
        }

        lastHeight = y;
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

    setDropoutPresevervationRates(oldDropRates);

    std::cout << "Done!" << std::endl;
}


void NeuralNet::print2x1NetworkImage(std::string name)
{
    std::ofstream image(name + ".ppm");

    int res = 256;

    std::cout << "Printing image.." << std::endl;

    image << "P3" << std::endl;
    image << res << " " << res << std::endl;
    image << "255" << std::endl;

    Color pixel;

    MatrixXd pixelLocation = MatrixXd(2,1);

    double pixelStrength;

    for(int y(0); y < res; y++)
    {
        for(int x(0); x < res; x++)
        {
            pixelLocation.coeffRef(0,0) = x / res;

            pixelLocation.coeffRef(1,0) = y / res;

            feedForward(pixelLocation);


             pixelStrength = outputLayer.outputs.coeff(0,0);


            pixel = Color{
                    int(254 * pixelStrength),
                    int(254 * pixelStrength),
                    int(254 * pixelStrength) };

            image << pixel.r << " " << pixel.g << " " << pixel.b << std::endl;
        }
    }

    image.close();

    std::cout << "Done!" << std::endl;
}

/*
void NeuralNet::print1x1NetworkImage(std::string name)
{
    std::ofstream image(name + ".ppm");

    int res = 256;

    std::cout << "Printing image.." << std::endl;

    std::vector<std::vector<Color>> baseImage;

    baseImage.resize(res);

    for(int i(0); i < res; i++)
        baseImage.at(i).resize(res, Color{254,254,254});

    for(int i(0); i < res; i++)
    {
        int y1 = ceil(randomWave(double(i)/res) * double(res));
        y1 = std::min(std::max(y1, 0), res-1);

        int y2 = floor(randomWave(double(i)/res) * double(res));
        y2 = std::min(std::max(y2, 0), res-1);

        baseImage[y1][i] = Color{50, 100, 200};
        baseImage[y2][i] = Color{50, 100, 200};
    }

    for(int i(0); i < res; i++)
    {
        feedForward({(double(i)/res)});
        int y = int(outputLayer.outputs.coeff(0,0) * double(res));

        y = std::max(std::min(y, res-1), 0);

        baseImage[y][i] = Color{0, 0, 0};
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
*/
