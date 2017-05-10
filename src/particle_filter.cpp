/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  //set number of particles
  num_particles = 300;

  //resize weight vector and set all weights to 1
  weights.resize(num_particles,1.0);

  // initialize random engine
  std::default_random_engine gen;
  
  // create normal distributions for x, y and theta
  std::normal_distribution<double> dist_x(x,std[0]);
  std::normal_distribution<double> dist_y(y,std[1]);
  std::normal_distribution<double> dist_theta(theta,std[2]);

  for (int i = 0; i < num_particles; i++) {

    struct Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
  
  std::default_random_engine gen;

  std::normal_distribution<double> dist_x_noise(0,std_pos[0]);
  std::normal_distribution<double> dist_y_noise(0,std_pos[1]);
  std::normal_distribution<double> dist_theta_noise(0,std_pos[2]);


  for (int i = 0; i < num_particles; i++){
    Particle particle = particles[i];

    double theta_tmp = particle.theta;
    double v_yaw = velocity/yaw_rate;
    double yaw_dt = theta_tmp + delta_t * yaw_rate;
    
    particle.x += v_yaw * (sin(yaw_dt)-sin(theta_tmp));
    particle.y += v_yaw * (cos(theta_tmp) - cos(yaw_dt));
    particle.theta += yaw_dt;
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
  
  //Precalculate elements of Multivariate-Gaussian-Probability
  double denominator = 2 * M_PI * std_landmark[0] * std_landmark[1];
  double exp_den_1 = 2 * pow(std_landmark[0],2);
  double exp_den_2 = 2 * pow(std_landmark[1],2);

  for (int i = 0; i < num_particles; i++) {
    Particle particle = particles[i];
    double weight = 1.0;
    
    //Transform car measurements from local coordinate system to map coordinate system
    for (int j = 0; j < observations.size(); j++) {

      // Untransformed observation
      LandmarkObs observation = observations[j];
      
      // Define variable for transformed observation
      LandmarkObs obs;
      obs.id = observation.id;
      obs.x = particle.x + (observation.x * cos(particle.theta)) - (observation.y * sin(particle.theta));
      obs.y = particle.y + (observation.x * sin(particle.theta)) + (observation.y * cos(particle.theta));
    
      //Associate each measurement with the closest landmark identifier
      Map::single_landmark_s closest_lm;

      double closest_dist = std::numeric_limits<double>::max();

      for (int lm; lm < map_landmarks.landmark_list.size(); lm++){
        Map::single_landmark_s map_lm = map_landmarks.landmark_list[lm];
        double dist_curr = dist(obs.x, obs.y, map_lm.x_f, map_lm.y_f);
        if (dist_curr < closest_dist) {
          closest_lm = map_lm;
          closest_dist = dist_curr;
        }
      }

      //Calculate particle weight values
      double nominator = exp(-pow(obs.x - closest_lm.x_f,2)/exp_den_1 + pow(obs.y - closest_lm.y_f,2)/exp_den_2);
      
      weight *= nominator/denominator;
    }
    particles[i].weight = weight; 
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::vector<Particle> particles_resampled;

  int N = weights.size();

  //initialize random engine
  std::default_random_engine gen;
  std::uniform_real_distribution<> dis_real(0,1);
  std::uniform_int_distribution<> dis_int(0,N-1);

  int index = dis_int(gen);
  double beta = 0.0;
  double w_max = *std::max_element(std::begin(weights),std::end(weights));
  
  for (int i = 0; i < N; i++) {
    beta += dis_real(gen) * 2.0 * w_max;
    while (beta > weights[index]){
      beta -= weights[index];
      index = (index + 1) % N;
    }

    Particle tmp = particles[index];
    Particle particle;
    particle.x = tmp.x;
    particle.y = tmp.y;
    particle.theta = tmp.theta;
    particle.weight = tmp.weight;
    particles_resampled.push_back(particle);
  }
  particles = particles_resampled;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
