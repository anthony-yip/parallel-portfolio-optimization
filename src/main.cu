//
// Created by anthony on 6/7/24.
//
#include "mvo.cuh"
#include <iostream>

extern void test_3x3();
extern void test_3x3_cuSolver();

int main() {
    // test_3x3();
    test_3x3_cuSolver();

//     std::string file_path = "../data/monthly_returns.csv";
//     TimeSeriesStockData data = MVO::read_historical_data(file_path);
//     ExpectedReturns expected_returns = MVO::estimate_expected_returns(data);
//     CovarianceMatrix sample_covariance_matrix = MVO::shrink_covariance_matrix(MVO::calculate_sample_covariance_matrix(data, expected_returns), shrinkage_factor);
//     Portfolio portfolio = MVO::solve(sample_covariance_matrix, expected_returns); ///< @note use std::move after testing for better performance.
//
//
//     /// Testing:
//     std::cout << "Number of stocks: " << data.size() << std::endl;
//     for (int i = 0; i < data.size(); i++) {
//         int num_days = data[i].size();
//         std::cout << "Stock " << i << " has " << num_days << " days of data" << std::endl;
//         std::cout << "Price on last day: " << data[i][num_days-1] << std::endl;
//         std::cout << "Price on third last day: " << data[i][num_days-3] << std::endl;
//         std::cout << "Price on fifth last day: " << data[i][num_days-5] << std::endl;
//     }
//     std::cout << "Expected returns: ";
//     for (int i = 0; i < expected_returns.size(); i++) {
//         std::cout << expected_returns[i] << " ";
//     }
//     std::cout << std::endl;
//
// //    std::cout << "Sample covariance matrix: " << std::endl;
// //    for (int i = 0; i < num_stocks; i++) {
// //        for (int j = 0; j < num_stocks; j++) {
// //            std::cout << sample_covariance_matrix[i * num_stocks + j] << " ";
// //        }
// //        std::cout << std::endl;
// //    }
//
//     std::cout << "Portfolio: ";
//     for (int i = 0; i < num_stocks; i++) {
//         std::cout << portfolio[i] << " ";
//     }
//     std::cout << std::endl;
    return 0;
}