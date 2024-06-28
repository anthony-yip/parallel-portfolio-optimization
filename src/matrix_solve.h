//
// Created by anthony on 6/28/24.
//
#pragma once
#include <array>
#include <cmath>
#include <iostream>

template<int N>
class MatSol {
public:
    using Matrix = std::array<float, N * N>; ///< Matrix type
    using Vector = std::array<float, N>; ///< Vector type

    static Vector solve(Matrix matrix, Vector vector) {
        gaussianElimination(matrix, vector);
        return backSubstitution(matrix, vector);
    }

    static bool VectorEquals(const Vector& v1, const Vector& v2) {
        for (int i = 0; i < N; i++) {
            if (std::abs(v1[i] - v2[i]) > 1e-4) {
                return false;
            }
        }
        return true;
    }

    static void PrintVector(const Vector& v) {
        for (int i = 0; i < N; i++) {
            std::cout << v[i] << " ";
        }
        std::cout << std::endl;
    }

    static void PrintMatrix(const Matrix& m) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << m[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
    }

private:
// Function to swap two rows in the matrix
    static float& index(Matrix& matrix, int i, int j) {
        return matrix[i * N + j];
    }

    static void swapRows(Matrix& matrix, Vector& vector, int row1, int row2) {
        for (int col = 0; col < N; col++) {
            std::swap(index(matrix, row1, col), index(matrix, row2, col));
        }
        std::swap(vector[row1], vector[row2]);
    }

// Function to multiply a row by a scalar
    static void scaleRow(Matrix& matrix, Vector& vector, int row, float scale) {
        for (int col = 0; col < N; col++) {
            index(matrix, row, col) *= scale;
        }
        vector[row] *= scale;
    }

// Function to add a multiple of one row to another
    static void rowOperation(Matrix& matrix, Vector& vector, int target_row, int source_row, float scale) {
        for (int col = 0; col < N; col++) {
            index(matrix, target_row, col) += scale * index(matrix, source_row, col);
        }
        vector[target_row] += scale * vector[source_row];
    }

// Gaussian elimination method
    static void gaussianElimination(Matrix& matrix, Vector& vector) {
        int i = 0;
        int j = 0;
        while (i < N && j < N) {
            // Find maximum in the current column starting from the pivot
            int max_row = i;
            for (int k = i + 1; k < N; k++) {
                if (std::abs(index(matrix, k, j)) > std::abs(index(matrix, max_row, j))) {
                    max_row = k;
                }
            }

            // Swap maximum row with current row
            if (max_row != i) {
                swapRows(matrix, vector, i, max_row);
            }

            // Make all rows below this one 0 in current column
            for (int k = i + 1; k < N; k++) {
                float scale = -index(matrix, k, j) / index(matrix, i, j);
                rowOperation(matrix, vector, k, i, scale);
            }

            i++;
            j++;
        }
    }

// Back substitution method
    static Vector backSubstitution(Matrix& matrix, Vector& vector) {
        Vector x;
        for (int i = N - 1; i >= 0; i--) {
            float sum = 0;
            for (int j = i + 1; j < N; j++) {
                sum += index(matrix, i, j) * x[j];
            }
            x[i] = (vector[i] - sum) / index(matrix, i, i);
        }
        return x;
    }
};


