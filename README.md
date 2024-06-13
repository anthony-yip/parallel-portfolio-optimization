# parallel-portfolio-optimization

Welcome! This here is a project aimed to integrate my recent courses in linear algebra and computational finance
with my knowledge of parallel computing. I also seek to learn more about modern portfolio optimization, C++
optimization techniques and "functional" programming with move semantics.

My goal is to create a mean-variance optimization algorithm that is parallelized using CUDA, and to compare its 
performance to a serial implementation given the overhead of copying data to and from the GPU.

This algorithm uses historical stock data of companies in the Nasdaq 100 from Yahoo Finance to calculate:
- Expected returns
- Covariance matrix using Ledoit-Wolf shrinkage

Assuming the presence of a risk-free asset, long and short positions and a 1-period (1 month) model,
the algorithm relies on the One-Fund Theorem to reduce the optimization problem to a system of linear equations,
which can be solved using Gaussian elimination. A ratio of amount of money to invest into each stock is returned.