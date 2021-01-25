#include "matrix.h"

Matrix::Matrix(int rows, int columns, double def)
    :width(columns), height(rows)
{
    matrix.resize(rows);

    for(int i(0); i < rows; i++)
    {
        matrix[i].resize(columns, def);
    }
}

Matrix::Matrix(int rows, int columns)
    :Matrix(rows, columns, 0.0)
{}

Matrix MatrixMath::multiply(Matrix lh, Matrix rh)
{
    if(lh.getWidth() != rh.getHeight())
    {
        std::cout << "Matrix mult error" << std::endl;

        lh.print();

        std::cout << std::endl;

        rh.print();

        exit(1);
    }

    Matrix newMatrix = Matrix(lh.getHeight(),rh.getWidth());

    for(int i(0); i < lh.getHeight(); i++)
    {
        for(int j(0); j < rh.getWidth(); j++)
        {
            for(int x(0); x < lh.getWidth(); x++)
            {
                newMatrix.at(i,j) += lh.at(i,x) * rh.at(x,j);
            }
        }
    }

    return newMatrix;
}

Matrix MatrixMath::multiply(Matrix lh, double rh)
{
    Matrix newMatrix = Matrix(lh.getHeight(), lh.getWidth());

    for(int i(0); i < lh.getHeight(); i++)
    {
        for(int j(0); j < lh.getWidth(); j++)
        {
            newMatrix.at(i,j) += lh.at(i, j) * rh;
        }
    }

    return newMatrix;
}

Matrix MatrixMath::add(Matrix lh, Matrix rh)
{
    if(lh.getWidth() != rh.getWidth() || lh.getHeight() != rh.getHeight())
    {
        std::cout << "Matrix addition error" << std::endl;

        lh.print();

        std::cout << std::endl;

        rh.print();

        exit(1);
    }

    Matrix newMatrix = Matrix(lh.getHeight(), lh.getWidth());

    for(int i(0); i < lh.getHeight(); i++)
    {
        for(int j(0); j < lh.getWidth(); j++)
        {
            newMatrix.at(i, j) = lh.at(i, j) + rh.at(i, j);
        }
    }

    return newMatrix;
}

Matrix MatrixMath::subtract(Matrix lh, Matrix rh)
{
    if(lh.getWidth() != rh.getWidth() || lh.getHeight() != rh.getHeight())
    {
        std::cout << "Matrix subtraction error" << std::endl;

        lh.print();

        std::cout << std::endl;

        rh.print();

        exit(1);
    }

    Matrix newMatrix = Matrix(lh.getHeight(), lh.getWidth());

    for(int i(0); i < lh.getHeight(); i++)
    {
        for(int j(0); j < lh.getWidth(); j++)
        {
            newMatrix.at(i, j) = lh.at(i, j) - rh.at(i, j);
        }
    }

    return newMatrix;
}

Matrix MatrixMath::pow2(Matrix matrix)
{
    for(int i(0); i < matrix.getHeight(); i++)
    {
        for(int j(0); j < matrix.getWidth(); j++)
        {
            matrix.at(i,j) = matrix.at(i,j) * matrix.at(i,j);
        }
    }

    return matrix;
}

Matrix MatrixMath::transpose(Matrix matrix)
{
    Matrix newMatrix = Matrix(matrix.getWidth(), matrix.getHeight());

    for(int i(0);i < matrix.getHeight(); i++)
    {
        for(int j(0); j < matrix.getWidth(); j++)
        {
            newMatrix.at(j, i) = matrix.at(i, j);
        }
    }

    return newMatrix;
}

Matrix MatrixMath::fromArray(std::vector<double> arr)
{
    Matrix newMatrix = Matrix(arr.size(), 1);

    for(int i(0); i < arr.size(); i++)
        newMatrix.at(i, 0) = arr.at(i);

    return newMatrix;
}

void Matrix::print()
{
    for(int i(0); i < height; i++)
    {
        for(int j(0); j < width; j++)
        {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << (at(i,j) >= 0? " " : "") << at(i,j)
                      << (j==width-1? "" : ",");
        }

        std::cout << std::endl;
    }
}

Matrix MatrixMath::addRow(Matrix matrix, double defaultVal)
{
    Matrix newMatrix = Matrix(matrix.getHeight() + 1, matrix.getWidth());

    for(int i(0); i < matrix.getHeight(); i++)
        for(int j(0); j < matrix.getWidth(); j++)
            newMatrix.at(i,j) = matrix.at(i,j);

    for(int j(0); j < matrix.getWidth(); j++)
        newMatrix.at(matrix.getHeight(), j) = defaultVal;

    return newMatrix;
}

Matrix MatrixMath::addRow(Matrix matrix)
{
    return addRow(matrix, 0.0);
}

Matrix MatrixMath::rowPopBack(Matrix matrix)
{
    Matrix newMatrix = Matrix(matrix.getHeight() - 1, matrix.getWidth());

    for(int i(0); i < newMatrix.getHeight(); i++)
        for(int j(0); j < matrix.getWidth(); j++)
            newMatrix.at(i,j) = matrix.get(i,j);

    return newMatrix;
}

Matrix MatrixMath::columnPopBack(Matrix matrix)
{
    Matrix newMatrix = Matrix(matrix.getHeight(), matrix.getWidth() - 1);

    for(int i(0); i < newMatrix.getHeight(); i++)
        for(int j(0); j < newMatrix.getWidth(); j++)
            newMatrix.at(i,j) = matrix.at(i,j);

    return newMatrix;
}

Matrix MatrixMath::map(Matrix matrix, double (*func)(double) )
{
    for(int i(0); i < matrix.getHeight(); i++)
        for(int j(0); j < matrix.getWidth(); j++)
        {
            matrix.at(i,j) = (*func)(matrix.at(i,j));
        }

    return matrix;
}

Matrix MatrixMath::hadamard(Matrix lh, Matrix rh)
{
    if(lh.getHeight() != rh.getHeight() || lh.getWidth() != rh.getWidth())
    {
        std::cout << "hadamard product error" << std::endl;

        lh.print();

        std::cout << std::endl;

        rh.print();

        std::cout << std::endl;

        exit(1);
    }

    for(int i(0); i < lh.getHeight(); i++)
        for(int j(0); j < lh.getWidth(); j++)
            lh.at(i,j) *= rh.at(i,j);

    return lh;
}

double& Matrix::at(int i, int j)
{
    return matrix[i][j];
}

double Matrix::get(int i, int j)
{
    return matrix[i][j];
}

int Matrix::getWidth()
{
    return width;
}

int Matrix::getHeight()
{
    return height;
}
