#pragma once

#include <vector>
#include <array>


constexpr __constant__ int num_stocks = 96; ///< Number of stocks in the model
constexpr __constant__ float shrinkage_factor = 0.2; ///< Shrinkage factor for Ledoit-Wolf.
constexpr __constant__ float risk_free_rate = 0.02; ///< Risk-free rate for the model

/// @note I suspect this has bad locality.
using TimeSeriesStockData = std::array<std::vector<float>, num_stocks>; ///< Historical stock data type
using ExpectedReturns = std::array<float, num_stocks>; ///< Expected returns type
/// Covariance matrix type. We use a 1D array because Thrust will likely require this representation.
using CovarianceMatrix = std::array<float, num_stocks * num_stocks>;
using Portfolio = std::array<float, num_stocks>; ///< Portfolio type

inline float& index_cov_matrix(CovarianceMatrix& matrix, int i, int j) {
    return matrix[i * num_stocks + j];
}