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
  num_particles = 50;

  //resize weight vector and set all weights to 1
  weights.resize(num_particles,1.0);

  // initialize random engine
  std::default_random_engine gen;
  
  // create normal distributions for x, y and theta
  std::normal_distribution<double> dist_x(x,std[0]);
  std::normal_distribution<double> dist_y(y,std[1]);
  std::normal_distribution<double> dist_theta(theta,std[2]);

  for (int i = 0; i < num_particles; i++) {

    Particle particle;
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

  
  for (int i = 0; i < num_particles; i++){
    Particle particle = particles[i];

    double theta_new = particle.theta + yaw_rate * delta_t;
    particle.x += velocity * (sin(theta_new) - sin(particle.theta)) / yaw_rate;
    particle.y += velocity * (cos(particle.theta) - cos(theta_new)) / yaw_rate;
    particle.theta = theta_new;
  
    std::normal_distribution<double> dist_x(particle.x,std_pos[0]);
    std::normal_distribution<double> dist_y(particle.y,std_pos[1]);
    std::normal_distribution<double> dist_theta(particle.theta,std_pos[2]);

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);

    particles[i] = particle;
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for(int i = 0; i < observations.size(); i++) {
    double closest_dist = std::numeric_limits<double>::max();
    int closest_id;
    LandmarkObs observation = observations[i];

    for (int j = 0; j < predicted.size(); j++) {
      double dist_curr = dist(observation.x, observation.y, predicted[j].x, predicted[j].y);
      if(dist_curr < closest_dist) {
        closest_dist = dist_curr;
        closest_id = predicted[j].id;
      }
    }
    
    observations[i].id = closest_id;
  }
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
  
  //Predefine required variables
  int obs_size = observations.size();
  
  //Precalculate elements of Multivariate-Gaussian-Probability
  double nominator;
  double denominator = sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);

  for (int i = 0; i < num_particles; i++) {
    Particle particle = particles[i];
    double weight = 1.0;

    //Transform car measurements from local coordinate system to map coordinate system
    std::vector<LandmarkObs> transformed_observations;
    for (int j = 0; j < obs_size; j++) {

      // Untransformed observation
      LandmarkObs observation = observations[j];
      
      // Create transformed observation
      LandmarkObs obs;
      obs.x = particle.x + (observation.x * cos(particle.theta)) - (observation.y * sin(particle.theta));
      obs.y = particle.y + (observation.x * sin(particle.theta)) + (observation.y * cos(particle.theta));
      transformed_observations.push_back(obs);
    } 

    
    //Get map landmarks in sensor range
    std::vector<LandmarkObs> potential_landmarks;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      if (dist(particle.x, particle.y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) < sensor_range) {
        LandmarkObs potential_landmark;
        potential_landmark.x = map_landmarks.landmark_list[j].x_f;
        potential_landmark.y = map_landmarks.landmark_list[j].y_f;
        potential_landmark.id = map_landmarks.landmark_list[j].id_i;
        potential_landmarks.push_back(potential_landmark);
      }
    }



    //Associate each measurement with the closest landmark identifier
    dataAssociation(potential_landmarks, transformed_observations);

    //Calculate particle weight values
    for (int j = 0; j < obs_size; j++) {
      LandmarkObs obs = transformed_observations[j];
      LandmarkObs landmark;
      bool landmark_found = false;
      for (int k = 0; k < potential_landmarks.size(); k++){
        if (potential_landmarks[k].id == obs.id){
          landmark = potential_landmarks[k];
          landmark_found = true;
        }
      }

      if(!landmark_found){continue;}


      double delta_x = obs.x - landmark.x;
      double delta_y = obs.y - landmark.y;


       nominator = exp(-0.5 * (pow(delta_x, 2.0) / std_landmark[0] + pow(delta_y, 2.0) / std_landmark[1]));
        weight *= nominator/denominator;
    }

    particles[i].weight = weight;
    weights[i] = weight;
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
    particles_resampled.push_back(tmp);
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
