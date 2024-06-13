#include <vector>
#include <array>

inline constexpr int num_stocks = 96; ///< Number of stocks in the model
/// @note I suspect this has bad locality.
typedef std::array<std::vector<float>, num_stocks> TimeSeriesStockData; ///< Historical stock data type
typedef std::array<float, num_stocks> ExpectedReturns; ///< Expected returns type
/// Covariance matrix type. We use a 1D array because Thrust will likely require this representation.
typedef std::array<float, num_stocks * num_stocks> CovarianceMatrix;
inline float& index_cov_matrix(CovarianceMatrix& matrix, int i, int j) {
    return matrix[i * num_stocks + j];
}
inline constexpr float shrinkage_factor = 0.2; ///< Shrinkage factor for Ledoit-Wolf.
typedef std::array<float, num_stocks> Portfolio; ///< Portfolio type