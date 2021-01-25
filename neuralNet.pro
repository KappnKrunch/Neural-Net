TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        layers.cpp \
        main.cpp \
        matrix.cpp \
        neuralNet.cpp

HEADERS += \
    layers.h \
    matrix.h \
    neuralNet.h
