//
// Created by anthony on 6/7/24.
//
#include <vector>
#include <array>
#include <string>
#include "constants.cuh"

/// Stocks in the model
enum class Stock {
    GOOG,
    AAPL,
    FB,
    BABA,
    AMZN,
    GE,
    AMD
};


class MVO {
private:
    /// Historical stock data for the stocks in the model

public:
    /**
     * @brief Read historical stock data from a file, returning it.
     *
     * @param file_path Path to the file containing the historical stock data
     * @return Data structure containing the historical stock data
     */
    static TimeSeriesStockData read_historical_data(const std::string &file_path);

    /// Return Value Optimization should allow you to return the data structure directly.
    /// data is used twice, so no rvalue reference. Since no copy is made, const reference is sufficient.
    static ExpectedReturns estimate_expected_returns(const TimeSeriesStockData &data);

    static CovarianceMatrix
    calculate_sample_covariance_matrix(const TimeSeriesStockData &data, const ExpectedReturns &expected_returns);

    static CovarianceMatrix shrink_covariance_matrix(CovarianceMatrix sample_covariance_matrix, float shrinkage_factor);

    /**
     * @brief Solve a matrix equation Ax = b
     * @param covariance_matrix Covariance matrix A
     * @param expected_returns Expected returns b
     */
    static Portfolio solve(CovarianceMatrix covariance_matrix, ExpectedReturns expected_returns);
};



