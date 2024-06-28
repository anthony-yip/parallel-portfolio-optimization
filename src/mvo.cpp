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
#include "matrix_solve.h"

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
    return sample_covariance_matrix;
}

/// Perform Gaussian decomposition on the covariance matrix to find the optimal portfolio weights.
Portfolio MVO::solve(CovarianceMatrix covariance_matrix, ExpectedReturns expected_returns) {
    // Transform expected returns into expected excess returns
    for (int i = 0; i < num_stocks; i++) {
        expected_returns[i] -= risk_free_rate;
    }
    return MatSol<num_stocks>::solve(covariance_matrix, expected_returns);
}


