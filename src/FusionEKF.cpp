#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;
using std::sin;
using std::cos;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // Kalman filter variables
  ekf_.x_ = VectorXd(4);		// state vector
  ekf_.P_ = MatrixXd(4,4);
  ekf_.F_ = MatrixXd(4,4);  	// state transition matrix
  ekf_.Q_ = MatrixXd(4,4);  	// process covariance matrix
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    // First measurement
    previous_timestamp_ = measurement_pack.timestamp_;
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    ekf_.P_ << 1, 0, 0, 0,
    		   0, 1, 0, 0,
    		   0, 0, 1000, 0,
    		   0, 0, 0, 1000;		// initial state covariance matrix based on initial velocity uncertainty
    
    // Initial measurement is from radar
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar measurements from polar to Cartesian coordinates
      ekf_.x_ << measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]),
                 measurement_pack.raw_measurements_[1] * sin(measurement_pack.raw_measurements_[1]), 
                 0, 
                 0;
    }
    // Initial measurement is from lidar
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_[0],
                 measurement_pack.raw_measurements_[1],
                 0,
                 0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */
  
  // Update state transition matrix F with elapsed time
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; // Convert from microseconds to seconds
  previous_timestamp_ = measurement_pack.timestamp_;    
    
  ekf_.F_ << 1, 0, dt, 0,
  			 0, 1, 0, dt,
  			 0, 0, 1, 0,
  			 0, 0, 0, 1;
  
  // Update process covariance matrix Q with acceleration noise components
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  
  // Set the process acceleration noise
  int noise_ax = 9;
  int noise_ay = 9;
  
  ekf_.Q_ << dt_4 * noise_ax / 4.0, 0, dt_3 * noise_ax / 2.0, 0,
  			 0, dt_4 * noise_ay / 4.0, 0, dt_3 * noise_ay / 2.0,
  			 dt_3 * noise_ax / 2.0, 0, dt_2 * noise_ax, 0,
  			 0, dt_3 * noise_ay / 2.0, 0, dt_2 * noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Measurement update for radar data
    ekf_.R_ = R_radar_;
    ekf_.H_ = Hj_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } 
  else {
    // Measurement update for lidar data
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
