//
// Created by anthony on 6/7/24.
//

#include "mvo.h"
#include <fstream>
#include <sstream>
#include <string>
#include <assert.h>
#include <vector>
#include <cmath>

TimeSeriesStockData MVO::read_historical_data(const std::string &file_path) {
    std::ifstream file(file_path);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    TimeSeriesStockData data;
    std::string line, word;
    int line_number = 0;
    while (std::getline(file, line)) {
        /// ignore first row (header containing stock names)
        if (line_number++ < 1) {
            continue;
        }

        std::stringstream s(line);
        int i = 0;
        while (std::getline(s, word, ',')) {
            // Skip date column and columns beyond num_cols
            if (i != 0) {
                data[i-1].push_back(std::stod(word));
            }
            i++;
        }
    }

    file.close();
    return data;
}

ExpectedReturns MVO::estimate_expected_returns(const TimeSeriesStockData &data) {
    /// Arithmetic mean of returns
    ExpectedReturns expected_returns;
    for (int stock = 0; stock < data.size(); stock++) {
        float sum_of_returns = 0;
        for (int date = 0; date < data[stock].size(); date++) {
            sum_of_returns += data[stock][date];
        }
        expected_returns[stock] = sum_of_returns / (data[stock].size());
    }
    return expected_returns;
}


CovarianceMatrix MVO::calculate_sample_covariance_matrix(const TimeSeriesStockData &data, const ExpectedReturns &expected_returns) {
    CovarianceMatrix covariance_matrix;
    assert(num_stocks == data.size());
    for (int i = 0; i < num_stocks; i++) {
        for (int j = 0; j < num_stocks; j++) {
            float sum = 0;
            for (int date = 0; date < data[i].size(); date++) {
                sum += (data[i][date] - expected_returns[i]) * (data[j][date] - expected_returns[j]);
            }
            index_cov_matrix(covariance_matrix, i, j) = sum / (data[i].size() - 1);
        }
    }
    return covariance_matrix;
}

CovarianceMatrix MVO::shrink_covariance_matrix(CovarianceMatrix sample_covariance_matrix, float shrinkage_factor) {
    /// Since our shrinkage target is the constant variance matrix, we can just scale the off-diagonal entries.
    float mean_variance = 0;
    for (int i = 0; i < num_stocks; i++) {
        mean_variance += index_cov_matrix(sample_covariance_matrix, i, i);
    }
    mean_variance /= num_stocks;
    /// do shrinkage:
    for (int i = 0; i < num_stocks; i++) {
        for (int j = 0; j < num_stocks; j++) {
            float shrinkage_target;
            if (i != j) {
                shrinkage_target = 0;
            }
            else {
                shrinkage_target = mean_variance;
            }
            index_cov_matrix(sample_covariance_matrix, i, j) = shrinkage_factor * shrinkage_target + (1 - shrinkage_factor) * index_cov_matrix(sample_covariance_matrix, i, j);
        }
    }
}

/// Perform Gaussian decomposition on the covariance matrix to find the optimal portfolio weights.
Portfolio MVO::solve(CovarianceMatrix covariance_matrix, ExpectedReturns expected_returns) {
    gaussianElimination(covariance_matrix, expected_returns);
    return backSubstitution(covariance_matrix, expected_returns);
}


// Function to swap two rows in the matrix
void swapRows(CovarianceMatrix& matrix, ExpectedReturns& vector, int row1, int row2) {
    for (int col = 0; col < num_stocks; col++) {
        std::swap(index_cov_matrix(matrix, row1, col), index_cov_matrix(matrix, row2, col));
    }
    std::swap(vector[row1], vector[row2]);
}

// Function to multiply a row by a scalar
void scaleRow(CovarianceMatrix& matrix, ExpectedReturns& vector, int row, float scale) {
    for (int col = 0; col < num_stocks; col++) {
        index_cov_matrix(matrix, row, col) *= scale;
    }
    vector[row] *= scale;
}

// Function to add a multiple of one row to another
void rowOperation(CovarianceMatrix& matrix, ExpectedReturns& vector, int target_row, int source_row, float scale) {
    for (int col = 0; col < num_stocks; col++) {
        index_cov_matrix(matrix, target_row, col) += scale * index_cov_matrix(matrix, source_row, col);
    }
    vector[target_row] += scale * vector[source_row];
}

// Gaussian elimination method
void gaussianElimination(CovarianceMatrix& matrix, ExpectedReturns& vector) {
    int i = 0;
    int j = 0;
    while (i < num_stocks && j < num_stocks) {
        // Find maximum in the current column starting from the pivot
        int max_row = i;
        for (int k = i + 1; k < num_stocks; k++) {
            if (std::abs(matrix[k * num_stocks + j]) > std::abs(matrix[max_row * num_stocks + j])) {
                max_row = k;
            }
        }

        // Swap maximum row with current row
        if (max_row != i) {
            swapRows(matrix, vector, i, max_row);
        }

        // Make all rows below this one 0 in current column
        for (int k = i + 1; k < num_stocks; k++) {
            float scale = -index_cov_matrix(matrix, k, j) / index_cov_matrix(matrix, i, j);
            rowOperation(matrix, vector, k, i, scale);
        }

        i++;
        j++;
    }
}

// Back substitution method
Portfolio backSubstitution(CovarianceMatrix& matrix, ExpectedReturns& vector) {
    Portfolio x;
    for (int i = num_stocks - 1; i >= 0; i--) {
        float sum = 0;
        for (int j = i + 1; j < num_stocks; j++) {
            sum += index_cov_matrix(matrix, i, j) * x[j];
        }
        x[i] = (vector[i] - sum) / index_cov_matrix(matrix, i, i);
    }
    return x;
}
