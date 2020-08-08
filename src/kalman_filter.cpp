#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * Predict the state
   */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * Update the state by using Kalman Filter equations
   */

  H_ << 1, 0, 0, 0,
        0, 1, 0, 0;                     // laser measurement matrix
  VectorXd z_pred = H_ * x_;		    // map predicted state onto measurement space
  VectorXd y = z - z_pred;				// calculate measurement residual
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;			// calculate Kalman gain
  
  // Updated state and covariance estimates
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size,x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * Update the state by using Extended Kalman Filter equations
   */
  
  Tools tools;
  
  // Map predicted locations from Cartesian to polar coordinates
  VectorXd hx_(3);
  float px_p = x_(0);
  float py_p = x_(1);
  float vx_p = x_(2);
  float vy_p = x_(3);
  hx_(0) = sqrt(px_p * px_p + py_p * py_p);
  hx_(1) = atan2(py_p, px_p);
  hx_(2) = (px_p * vx_p + py_p * vy_p) / hx_(0);
  
  VectorXd y = z - hx_;
  
  // Adjust radar angle phi to be in range [-PI, PI]
  const float PI = 3.14159265358979f;
  while (y(1) < -PI) {
    y(1) += 2 * PI;
  }
  while (y(1) > PI) {
    y(1) -= 2 * PI;
  }
  
  H_ = tools.CalculateJacobian(x_);	// calculate Jacobian measurement matrix
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;			// Kalman gain
  
  // Updated state and covariance estimates
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size,x_size);
  P_ = (I - K * H_) * P_;
}
