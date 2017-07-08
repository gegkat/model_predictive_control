#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

class MPC {
 public:

  double ref_v;
  double Kd_delta;
  double Kd_a;
  double Kp_cte;
  double Kp_psi;
  double Kp_v;
  double Kp_delta;
  double Kp_a;
  MPC();

  virtual ~MPC();

  /*
  * Initialize MPC.
  */
void Init(double ref_v_, double Kd_delta_, double Kd_a_, double Kp_cte_, 
  double Kp_psi_, double Kp_v_, double Kp_a_, double Kp_delta_);


  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
};

#endif /* MPC_H */
