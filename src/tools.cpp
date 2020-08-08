#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::endl;
using std::cout;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculate the root mean squared error (RMSE) given KF estimations and ground truth vectors.
   */
  
  // Initialize rmse vector
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  
  // Check that input vector sizes are valid
  if (estimations.size() != ground_truth.size() 
      || estimations.size() < 0.00001) {
    cout << "Invalid vector size of estimations and/or ground truth." << endl;
    return rmse;
  }

  // Square each residual and take the summation of squares
  for (unsigned int i=0; i < estimations.size(); ++i) {
    
    // Initialize residual vector, square each difference, and accumulate
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }
  
  // Calculate the mean of accumulated residuals
  rmse = rmse / estimations.size();
  
  // Calculate the square root to return the rmse
  rmse = rmse.array().sqrt();
  
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * Calculate a Jacobian matrix given a state variable vector.
   */
  
  // Recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  // Calculate intermediate coefficients
  float px_2 = px * px;
  float py_2 = py * py;
  float denom_1_2 = sqrt(px_2 + py_2);
  float denom_3_2 = sqrt((px_2 + py_2) * (px_2 + py_2) * (px_2 + py_2));
  
  // Check for division by zero
  MatrixXd Hj_(3,4);
  if (px_2 + py_2 < 0.00001) {
    cout << "Error: Could not divide by zero." << endl;
    return Hj_;
  }
  
  // Compute Jacobian matrix components
  Hj_(0,0) = px / denom_1_2;
  Hj_(0,1) = py / denom_1_2;
  Hj_(0,2) = 0.0;
  Hj_(0,3) = 0.0;
  Hj_(1,0) = -py / (px_2 + py_2);
  Hj_(1,1) = px / (px_2 + py_2);
  Hj_(1,2) = 0.0;
  Hj_(1,3) = 0.0;
  Hj_(2,0) = py * (vx * py - vy * px) / denom_3_2;
  Hj_(2,1) = px * (vy * px - vx * py) / denom_3_2;
  Hj_(2,2) = px / denom_1_2;
  Hj_(2,3) = py / denom_1_2;

  return Hj_;
}
