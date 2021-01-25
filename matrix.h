#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>

class Matrix
{
private:
    std::vector<std::vector<double>> matrix;
    int width;
    int height;

public:

    Matrix();

    Matrix(int rows, int columns, double def);

    Matrix(int rows, int columns);

    void print();

    double& at(int i, int j);

    double get(int i, int j);

    int getWidth();

    int getHeight();
};

class MatrixMath
{
public:

    static Matrix multiply(Matrix lh, Matrix rh);

    static Matrix multiply(Matrix lh, double rh);

    static Matrix add(Matrix lh, Matrix rh);

    static Matrix subtract(Matrix lh, Matrix rh);

    static Matrix fromArray(std::vector<double> arr);

    static Matrix transpose(Matrix matrix);

    static Matrix pow2(Matrix matrix);

    static Matrix addRow(Matrix matrix, double defaultVal);

    static Matrix addRow(Matrix matrix);

    static Matrix rowPopBack(Matrix matrix);

    static Matrix columnPopBack(Matrix matrix);

    static Matrix map(Matrix matrix, double (*func)(double) );

    static Matrix hadamard(Matrix lh, Matrix rh);
};



#endif // MATRIX_H
