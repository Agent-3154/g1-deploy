#include "mujoco/mjspec.h"
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

#include <mujoco/mujoco.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <cmath>


using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;


inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;
    for (uint32_t i = 0; i < len; i++) {
        xbit = 1 << 31;
        data = ptr[i];
        for (uint32_t bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            } else {
                CRC32 <<= 1;
            }
            if (data & xbit) CRC32 ^= dwPolynomial;
    
            xbit >>= 1;
        }
    }
    return CRC32;
};

// Helper function to convert quaternion [w,x,y,z] to RPY [roll, pitch, yaw]
template<typename T>
inline void quatToRPY(const T quat[4], T rpy[3]) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    
    // Roll (x-axis rotation)
    T sinr_cosp = T(2.0) * (w * x + y * z);
    T cosr_cosp = T(1.0) - T(2.0) * (x * x + y * y);
    rpy[0] = std::atan2(sinr_cosp, cosr_cosp);
    
    // Pitch (y-axis rotation)
    T sinp = T(2.0) * (w * y - z * x);
    if (std::abs(sinp) >= T(1.0)) {
        rpy[1] = std::copysign(M_PI / T(2.0), sinp); // Use 90 degrees if out of range
    } else {
        rpy[1] = std::asin(sinp);
    }
    
    // Yaw (z-axis rotation)
    T siny_cosp = T(2.0) * (w * z + x * y);
    T cosy_cosp = T(1.0) - T(2.0) * (y * y + z * z);
    rpy[2] = std::atan2(siny_cosp, cosy_cosp);
}

// Generic RobotData template - can hold float (hardware) or double (mujoco) data
template<typename T>
struct RobotData {
    std::array<T, 29> q;
    std::array<T, 29> dq;
    std::array<T, 29> tau;

    std::array<T, 4> quaternion;
    std::array<T, 3> rpy;
    std::array<T, 3> omega;
};


// Generic base interface - T can be float or double
template<typename T>
class G1Interface {
protected:
    RobotData<T> robot_data_;
    mjModel *model_;
    mjData *data_;
public:
    G1Interface() {
        model_ = nullptr;
        data_ = nullptr;
    }
    
    void loadMJCF(std::string mjcf_path) {
        char error[1000] = {0};
        model_ = mj_loadXML(mjcf_path.c_str(), NULL, error, 1000);
        
        if (!this->model_) {
            std::cerr << "ERROR: Failed to load MuJoCo model from " << mjcf_path << std::endl;
            std::cerr << "Error: " << error << std::endl;
            throw std::runtime_error("Failed to load MuJoCo model");
        }
        
        data_ = mj_makeData(this->model_);
        if (!data_) {
            std::cerr << "ERROR: Failed to create MuJoCo data" << std::endl;
            mj_deleteModel(this->model_);
            throw std::runtime_error("Failed to create MuJoCo data");
        }
        
        std::cout << "Model loaded successfully" << std::endl;
        std::cout << "Model has " << this->model_->nq << " DOFs" << std::endl;
    }
    
    RobotData<T> getData() const {
        return robot_data_;
    }
};


// Hardware interface uses float (matching hardware data types)
class G1HarwareInterface: public G1Interface<float> {
private:
    ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
    ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;

    void LowStateCallback(const void *message) {
        LowState_ low_state = *(const LowState_ *)message;
        if (low_state.crc() != Crc32Core((uint32_t *)&low_state, (sizeof(LowState_) >> 2) - 1)) {
            std::cout << "[ERROR] CRC Error" << std::endl;
            return;
        }
        for (int i = 0; i < 29; i++) {
            this->robot_data_.q[i] = low_state.motor_state()[i].q();
            this->robot_data_.dq[i] = low_state.motor_state()[i].dq();
            this->robot_data_.tau[i] = low_state.motor_state()[i].tau_est();
        }
        this->robot_data_.quaternion = low_state.imu_state().quaternion();
        this->robot_data_.rpy = low_state.imu_state().rpy();
        this->robot_data_.omega = low_state.imu_state().gyroscope();
    }
public:
    G1HarwareInterface(std::string networkInterface) {
        ChannelFactory::Instance()->Init(0, networkInterface);

        lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>("rt/lowstate"));
        lowstate_subscriber_->InitChannel(std::bind(&G1HarwareInterface::LowStateCallback, this, std::placeholders::_1), 1);

        // lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>("rt/lowcmd"));
        // lowcmd_publisher_->InitChannel();
    }
};


// MuJoCo interface uses double (matching mjtNum, avoiding conversions)
class G1MujocoInterface: public G1Interface<double> {
private:
    std::thread physics_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> should_stop_;
    std::mutex step_mutex_;
    double timestep_;  // Physics timestep in seconds
    
    void step() {
        std::lock_guard<std::mutex> lock(step_mutex_);
        if (this->model_ && this->data_) {
            mj_step(this->model_, this->data_);
            
            // Joint positions: qpos[7:] (skip root position + quaternion, copy up to 29 joints)
            int num_joints = std::min(29, std::max(0, this->model_->nq - 7));
            if (num_joints > 0 && this->model_->nq >= 7) {
                std::copy(this->data_->qpos + 7, 
                         this->data_->qpos + 7 + num_joints,
                         this->robot_data_.q.begin());
            }
            
            // Joint velocities: qvel[6:] (skip root velocity, copy up to 29 joints)
            int num_velocities = std::min(29, std::max(0, this->model_->nv - 6));
            if (num_velocities > 0 && this->model_->nv >= 6) {
                std::copy(this->data_->qvel + 6,
                         this->data_->qvel + 6 + num_velocities,
                         this->robot_data_.dq.begin());
                
                // Joint torques: use qfrc_actuator if available, otherwise qfrc_applied
                // for (int i = 0; i < num_velocities; i++) {
                //     int vel_idx = 6 + i;
                //     if (this->model_->nu > 0 && vel_idx < this->model_->nu) {
                //         this->robot_data_.tau[i] = this->data_->qfrc_actuator[vel_idx];
                //     } else if (vel_idx < this->model_->nv) {
                //         this->robot_data_.tau[i] = this->data_->qfrc_applied[vel_idx];
                //     }
                // }
            }
            
            // Quaternion: qpos[3:7] (4 elements: w, x, y, z)
            if (this->model_->nq >= 7) {
                std::copy(this->data_->qpos + 3,
                         this->data_->qpos + 7,
                         this->robot_data_.quaternion.begin());
                
                // Convert quaternion to RPY
                quatToRPY(this->data_->qpos + 3, this->robot_data_.rpy.data());
            }
        }
    }
    
    void physics_loop() {
        auto last_time = std::chrono::steady_clock::now();
        
        while (!should_stop_.load()) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<double>(current_time - last_time).count();
            
            // Step physics at the specified timestep rate
            if (elapsed >= timestep_) {
                step();
                last_time = current_time;
            } else {
                // Sleep to avoid busy-waiting
                std::this_thread::sleep_for(std::chrono::microseconds(
                    static_cast<int>((timestep_ - elapsed) * 1000000)
                ));
            }
        }
        
        running_.store(false);
    }

public:
    G1MujocoInterface(std::string mjcf_path, double timestep = -1.0) 
        : running_(false), should_stop_(false), timestep_(timestep) {
        this->loadMJCF(mjcf_path);
        // Use model's timestep if not specified
        if (timestep_ < 0 && this->model_) {
            timestep_ = this->model_->opt.timestep;
        }
    }
    
    ~G1MujocoInterface() {
        stop();
        if (data_) {
            mj_deleteData(data_);
        }
        if (model_) {
            mj_deleteModel(model_);
        }
    }
    
    void run_async() {
        if (!running_.load()) {
            should_stop_.store(false);
            running_.store(true);
            physics_thread_ = std::thread(&G1MujocoInterface::physics_loop, this);
        } else {
            // throw error
            throw std::runtime_error("Physics thread already running");
        }
    }
    
    void stop() {
        if (running_.load()) {
            should_stop_.store(true);
            if (physics_thread_.joinable()) {
                physics_thread_.join();
            }
        } else {
            // throw error
            throw std::runtime_error("Physics thread not running");
        }
    }
    
    bool is_running() const {
        return running_.load();
    }
    
    void set_timestep(double timestep) {
        timestep_ = timestep;
    }
    
    double get_timestep() const {
        return timestep_;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(step_mutex_);
        if (this->model_ && this->data_) {
            mj_resetData(this->model_, this->data_);
        }
    }

};

