#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

size_t steer_idx = 6;
size_t throttle_idx = steer_idx +1;
size_t trajectory_start_idx = throttle_idx + 1;


// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main(int argc, char* argv[])
{
  // Command line arguments
  double ref_v, Kd_delta, Kd_a, Kp_cte, Kp_psi, Kp_v, Kp_delta, Kp_a;
  if (argc > 1) {
    Kp_cte = atof(argv[1]);
    Kp_psi = atof(argv[2]);
    Kp_v = atof(argv[3]);
    Kp_delta = atof(argv[4]);
    Kp_a = atof(argv[5]);
    Kd_delta = atof(argv[6]);
    Kd_a = atof(argv[7]);
    ref_v = atof(argv[8]);

  } else {
    std::cout << "Using default parameter tunings" << endl;
    Kp_cte = 8;
    Kp_psi = 8;
    Kp_v = 10;
    Kp_delta = 10;
    Kp_a = 10;
    Kd_delta = 5;
    Kd_a = 10;
    ref_v = 200;
  }

  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;  // Initialize the pid variable.
  mpc.Init(ref_v, Kd_delta, Kd_a, Kp_cte, Kp_psi, Kp_v, Kp_delta, Kp_a);
  std::cout << " Kp_cte: "    << Kp_cte;
  std::cout << " Kp_psi: "    << Kp_psi;
  std::cout << " Kp_v: "      << Kp_v;
  std::cout << " Kp_delta: "  << Kp_delta;
  std::cout << " Kp_a: "      << Kp_a;
  std::cout << " Kd_delta: "  << Kd_delta;
  std::cout << " Kd_a: "      << Kd_a;
  std::cout << " ref_v: "      << ref_v;
  std::cout << std::endl;
  std::cout << "cte, psi_error, steer, throttle, speed" << std::endl;


  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double prev_steer = j[1]["steering_angle"];
          double prev_throttle = j[1]["throttle"];

          prev_steer *= deg2rad(25);
          prev_throttle = prev_throttle*2-1;

          /*
          * Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */

          json msgJson;

          // Convert waypoints to vehicle frame
          vector<double> waypoints_in_vehicle_frame_x;
          vector<double> waypoints_in_vehicle_frame_y;

          double dx, dy;
          // Cache calculation for efficiency
          double ct = cos(psi);
          double st = sin(psi);

          // Convert each way point to vehicle frame
          for (int i = 0; i < ptsx.size(); i++) {
            dx = ptsx[i] - px;
            dy = ptsy[i] - py;
            waypoints_in_vehicle_frame_x.push_back( dx*ct + dy*st);
            waypoints_in_vehicle_frame_y.push_back(-dx*st + dy*ct);
          }

          // Send waypoints for display
          msgJson["next_x"] = waypoints_in_vehicle_frame_x;
          msgJson["next_y"] = waypoints_in_vehicle_frame_y;

          // Convert waypoints to Eigen Vector and do polyfit
          Eigen::Map<Eigen::VectorXd> waypoints_vector_x (&waypoints_in_vehicle_frame_x[0], 6);
          Eigen::Map<Eigen::VectorXd> waypoints_vector_y (&waypoints_in_vehicle_frame_y[0], 6);
          Eigen::VectorXd coeffs = polyfit(waypoints_vector_x, waypoints_vector_y, 3); 

          // Cross track error is simply the x intercept of the polynomial
          double cte = coeffs[0];

          // Psi error is the derivative of the polynomial evaluate at x = 0
          double psi_error = -atan(coeffs[1]);

          // Propagate forward by the delay
          /*
      fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t] = psi1 - (psi0 - v0 * delta0 / Lf * dt); // sign flip
      fg[1 + v_start + t] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + t] =
          cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + t] =
          epsi1 - ((psi0 - psides0) - v0 * delta0 / Lf * dt); // sign flip
*/

          double dt = 0.1;
          const double Lf = 2.67;
          double x = v*cos(psi_error)*dt;
          double y = v*sin(psi_error)*dt;
          double next_psi = - v * prev_steer / Lf * dt;
          double next_v = v + prev_throttle * dt;
          double next_cte = cte + v*sin(psi_error)*dt;
          double next_psi_error = psi_error - v * prev_steer / Lf * dt;

          // Define the state in the body coordinate frame, x, y, psi are 0
          Eigen::VectorXd state(6);
          //state << 0, 0, 0, v, cte, psi_error;
          state << x, y, next_psi, next_v, next_cte, next_psi_error;

          // Use mpc object to solve for actuator commands
          auto vars = mpc.Solve(state, coeffs);

          // Pull out the steer and throttle values from vars
          double steer_value = vars[steer_idx];
          double throttle_value = vars[throttle_idx];

          // Send commands
          msgJson["steering_angle"] = steer_value/(deg2rad(25));
          msgJson["throttle"] = throttle_value;

          // Output for plotting in python
          //std::cout << cte << ", " << rad2deg(psi_error) << ", " << steer_value << ", " << throttle_value << ", " << v << endl;
          std::cout << prev_steer << ", " << steer_value << ", " << throttle_value << ", " << prev_throttle << endl;

          // Get the MPC predicted trajectory for display
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
  
          int i = trajectory_start_idx;
          while (i < vars.size()) {
              mpc_x_vals.push_back(vars[i]);
              i++;
              mpc_y_vals.push_back(vars[i]);
              i++;
          }
          
          // Send mpc trajectory for display
          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          //std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
