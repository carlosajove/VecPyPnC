#pragma once

#include <towr_plus/models/endeffector_mappings.h>
#include <towr_plus/models/kinematic_model.h>
#include <towr_plus/models/single_rigid_body_dynamics.h>

namespace towr_plus {

class AtlasKinematicModel : public KinematicModel {
public:
  AtlasKinematicModel() : KinematicModel(2) {
    const double x_nominal_b = -0.008;
    const double y_nominal_b = 0.111;
    const double z_nominal_b = -0.765;

    foot_half_length_ = 0.11;
    foot_half_width_ = 0.065;

    nominal_stance_.at(L) << x_nominal_b, y_nominal_b, z_nominal_b;
    nominal_stance_.at(R) << x_nominal_b, -y_nominal_b, z_nominal_b;

    max_dev_from_nominal_ << 0.18, 0.1, 0.05;
    min_dev_from_nominal_ << -0.18, -0.1, -0.05;
  }
};

class AtlasDynamicModel : public SingleRigidBodyDynamics {
public:
  /* Atlas Reduced model
   Mass:
   98.4068
   Inertia:
      4.48975 -0.0282483   0.386339
   -0.0282483    4.62886  0.0325983
     0.386339  0.0325983   0.830916
  */
  AtlasDynamicModel()
      //: SingleRigidBodyDynamics(98.4068, 18.58, 15.41, 4.08, -0.01, -0.03,
      //0.06, 2) {}
      : SingleRigidBodyDynamics(98.4068, 34., 27.5, 14.4, 0.15, 4.1, -0.06, 2) {
  }
};

} /* namespace towr_plus */
