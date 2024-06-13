//
// Created by anthony on 6/13/24.
//
#pragma once

#include <array>
#include <iostream>

template <std::size_t N>
void prettyPrintMatrix(const std::array<float, N>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i < 10 || i >= rows - 10 || j < 10 || j >= cols - 10) {
                std::cout << matrix[i * cols + j] << " ";
            } else if (j == 10 && cols > 20) {
                std::cout << "... ";
            }
        }
        if (i == 10 && rows > 20) {
            std::cout << "\n...\n";
        } else {
            std::cout << "\n";
        }
    }
}