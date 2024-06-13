//
// Created by anthony on 6/7/24.
//
#include <vector>
#include <array>
#include <string>
#include "constants.h"

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
    static TimeSeriesStockData read_historical_data(const std::string& file_path);
    /// Return Value Optimization should allow you to return the data structure directly.
    /// data is used twice, so no rvalue reference. Since no copy is made, const reference is sufficient.
    static ExpectedReturns estimate_expected_returns(const TimeSeriesStockData& data);
    static CovarianceMatrix calculate_sample_covariance_matrix(const TimeSeriesStockData& data, const ExpectedReturns& expected_returns);
    static CovarianceMatrix shrink_covariance_matrix(CovarianceMatrix sample_covariance_matrix, float shrinkage_factor);
    static Portfolio solve(CovarianceMatrix covariance_matrix, ExpectedReturns expected_returns);

private:
    static void swapRows(CovarianceMatrix& matrix, ExpectedReturns& vector, int row1, int row2);
    static void scaleRow(CovarianceMatrix& matrix, ExpectedReturns& vector, int row, float scale);
    static void rowOperation(CovarianceMatrix& matrix, ExpectedReturns& vector, int target_row, int source_row, float scale);
    static void gaussianElimination(CovarianceMatrix& matrix, ExpectedReturns& vector);
    static Portfolio backSubstitution(CovarianceMatrix& matrix, ExpectedReturns& vector);
};



