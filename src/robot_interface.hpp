#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>

#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/g1/loco/g1_loco_api.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>

#include "gamepad.hpp"

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
using namespace unitree_go::msg::dds_;

const std::array<int, 6> wrist_joint_indices = {23, 24, 25, 26, 27, 28};
const float default_wrist_stiffness = 16.0f;
const float default_wrist_damping = 2.0f;

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

// Gamepad state structure for Python exposure
struct GamepadState {
    // Analog stick values
    float lx = 0.0f;
    float rx = 0.0f;
    float ry = 0.0f;
    float ly = 0.0f;
    float l2 = 0.0f;
    
    // Button states: pressed, on_press, on_release
    struct ButtonState {
        bool pressed = false;
        bool on_press = false;
        bool on_release = false;
    };
    
    ButtonState R1, L1, start, select, R2, L2, F1, F2;
    ButtonState A, B, X, Y;
    ButtonState up, right, down, left;
};

// Generic RobotData template - can hold float (hardware) or double (mujoco) data
template<typename T>
struct RobotData {
    std::array<T, 3> root_pos_w;
    std::array<T, 3> root_lin_vel_w;
    std::array<T, 3> root_ang_vel_w;

    std::array<T, 29> q;
    std::array<T, 29> q_target;
    std::array<T, 29> dq;
    std::array<T, 29> dq_target;
    std::array<T, 29> tau;
    std::array<T, 29> joint_stiffness;
    std::array<T, 29> joint_damping;
    std::array<T, 3> projected_gravity;

    std::array<T, 4> quaternion;
    std::array<T, 3> rpy;
    std::array<T, 3> omega;

    // Body positions: [n_bodies, 3] - Eigen matrix for efficient contiguous storage and operations
    Eigen::Matrix<T, Eigen::Dynamic, 3> body_positions;
    Eigen::Matrix<T, Eigen::Dynamic, 4> body_quaternions;
    // std::array<T, 6> body_velocities; // linear velocity and angular velocity
    bool is_user_control_;
};


enum class ControlState {
    BUILTIN_CONTROL,
    USER_CONTROL,
    DAMPING_MODE
};


// Generic base interface - T can be float or double
template<typename T>
class G1Interface {
protected:
    RobotData<T> robot_data_;
    mjModel *model_;
    mjData *data_;
    ControlState control_state_ = ControlState::BUILTIN_CONTROL;

    void computeFK() {
        if (this->model_ && this->data_) {
            // Copy robot_data_ into data_ to compute FK
            // Root position: qpos[0:3] = [x, y, z]
            std::copy(this->robot_data_.root_pos_w.begin(),
                        this->robot_data_.root_pos_w.end(),
                        this->data_->qpos);

            // Root quaternion: qpos[3:7] = [w, x, y, z]
            std::copy(this->robot_data_.quaternion.begin(),
                        this->robot_data_.quaternion.end(),
                        this->data_->qpos + 3);
            
            // Joint positions: qpos[7:7+29] = robot_data_.q[0:29]
            for (int i = 0; i < this->njoints_; i++) {
                this->data_->qpos[7 + i] = static_cast<mjtNum>(this->robot_data_.q[i]);
            }
            
            mj_forward(this->model_, this->data_);

            // Update body positions from data_->xpos [nbody, 3] using Eigen::Map for efficient copy
            // excluding the world body
            if (this->nbodies_ > 0) {
                Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> xpos_map(
                    this->data_->xpos, this->model_->nbody, 3);
                this->robot_data_.body_positions = xpos_map.bottomRows(this->nbodies_).cast<T>();
                
                Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>> xquat_map(
                    this->data_->xquat, this->model_->nbody, 4);
                this->robot_data_.body_quaternions = xquat_map.bottomRows(this->nbodies_).cast<T>();
            }
        }
    }

public:
    int njoints_;
    int nbodies_;

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
        std::cout << "Model has " << this->model_->nbody << " bodies" << std::endl;
        
        this->njoints_ = this->model_->nq - 7;
        this->nbodies_ = this->model_->nbody - 1; // exclude the world body

        // Initialize body_positions Eigen matrix to match number of bodies [nbody, 3]
        this->robot_data_.body_positions.resize(this->nbodies_, 3);
        this->robot_data_.body_quaternions.resize(this->nbodies_, 4);
        
        mj_forward(this->model_, this->data_);
    }
    
    RobotData<T> getData() const {
        RobotData<T> data = robot_data_;
        data.is_user_control_ = (this->control_state_ == ControlState::USER_CONTROL);
        return data;
    }

    void setJointStiffness(const std::array<T, 29> &joint_stiffness) {
        this->robot_data_.joint_stiffness = joint_stiffness;
    }

    void setJointDamping(const std::array<T, 29> &joint_damping) {
        this->robot_data_.joint_damping = joint_damping;
    }

    void writeJointPositionTarget(const std::array<T, 29> &joint_position_target) {
        this->robot_data_.q_target = joint_position_target;
    }

    void writeJointVelocityTarget(const std::array<T, 29> &joint_velocity_target) {
        this->robot_data_.dq_target = joint_velocity_target;
    }
};


// Hardware interface uses float (matching hardware data types)
class G1HarwareInterface: public G1Interface<float> {
private:
    
    ThreadPtr lowcmd_thread_; // thread for writing lowcmd
    ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
    ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;
    ChannelSubscriberPtr<SportModeState_> estimate_state_subscriber_;

    std::shared_ptr<b2::MotionSwitcherClient> msc_;
    std::shared_ptr<g1::LocoClient> loco_client_;

    Gamepad gamepad_;
    REMOTE_DATA_RX rx_;

    int lowstate_counter_ = 0;
    int fk_counter_ = 0;
    uint8_t mode_machine_ = 0;
    std::mutex mode_switch_mutex_;  // Mutex for thread-safe mode switching
    std::mutex gamepad_mutex_;  // Mutex for thread-safe gamepad access 

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
        auto quaternion = low_state.imu_state().quaternion();
        this->robot_data_.quaternion = quaternion;
        Eigen::Quaternionf quat(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
        // Compute projected gravity: rotate gravity vector [0, 0, -1] from world to body frame
        Eigen::Vector3f gravity_world(0.0f, 0.0f, -1.0f);
        Eigen::Vector3f gravity_body = quat.inverse() * gravity_world;
        this->robot_data_.projected_gravity[0] = gravity_body.x();
        this->robot_data_.projected_gravity[1] = gravity_body.y();
        this->robot_data_.projected_gravity[2] = gravity_body.z();
        
        this->robot_data_.rpy = low_state.imu_state().rpy();
        this->robot_data_.omega = low_state.imu_state().gyroscope();
        this->mode_machine_ = low_state.mode_machine();
        lowstate_counter_ = (lowstate_counter_ + 1) % 100;

        // update gamepad
        {
            std::lock_guard<std::mutex> lock(gamepad_mutex_);
            memcpy(rx_.buff, &low_state.wireless_remote()[0], 40);
            gamepad_.update(rx_.RF_RX);
        }
        
        // Update control FSM based on gamepad input
        updateControlFSM();
        
        if (lowstate_counter_ % 10 == 0) {
            this->computeFK();
        }
    }

    void OdomMessageHandler(const void* message){
        SportModeState_ state = *(SportModeState_*)message;
        this->robot_data_.root_pos_w[0] = state.position()[0];
        this->robot_data_.root_pos_w[1] = state.position()[1];
        this->robot_data_.root_pos_w[2] = state.position()[2];
    }

    void LowCommandWriter() {
        if (control_state_ == ControlState::USER_CONTROL) {
            LowCmd_ dds_low_command;
            dds_low_command.mode_pr() = static_cast<uint8_t>(0);
            dds_low_command.mode_machine() = mode_machine_;
            
            for (int i = 0; i < this->njoints_; i++) {
                dds_low_command.motor_cmd()[i].mode() = 1; // 1:Enable, 0:Disable
                dds_low_command.motor_cmd()[i].tau() = 0.0f;
                dds_low_command.motor_cmd()[i].q() = this->robot_data_.q_target[i];
                dds_low_command.motor_cmd()[i].dq() = this->robot_data_.dq_target[i];
                dds_low_command.motor_cmd()[i].kp() = this->robot_data_.joint_stiffness[i];
                dds_low_command.motor_cmd()[i].kd() = this->robot_data_.joint_damping[i];
            }
            dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
            this->lowcmd_publisher_->Write(dds_low_command);
        } else if (control_state_ == ControlState::DAMPING_MODE) {
            LowCmd_ dds_low_command;
            dds_low_command.mode_pr() = static_cast<uint8_t>(0);
            dds_low_command.mode_machine() = mode_machine_;

            for (int i = 0; i < this->njoints_; i++) {
                dds_low_command.motor_cmd()[i].mode() = 1; // 1:Enable, 0:Disable
                dds_low_command.motor_cmd()[i].q() = this->robot_data_.q[i];
                dds_low_command.motor_cmd()[i].dq() = 0.0f;
                dds_low_command.motor_cmd()[i].kp() = 0.0f;
                dds_low_command.motor_cmd()[i].kd() = 4.0f;
            }
            dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
            this->lowcmd_publisher_->Write(dds_low_command);
        }
    }

public:
    G1HarwareInterface(std::string networkInterface) {
        ChannelFactory::Instance()->Init(0, networkInterface);

        lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>("rt/lowstate"));
        lowstate_subscriber_->InitChannel(std::bind(&G1HarwareInterface::LowStateCallback, this, std::placeholders::_1), 1);

        lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>("rt/lowcmd"));
        lowcmd_publisher_->InitChannel();

        estimate_state_subscriber_.reset(new ChannelSubscriber<SportModeState_>("rt/odommodestate"));
        estimate_state_subscriber_->InitChannel(std::bind(&G1HarwareInterface::OdomMessageHandler, this, std::placeholders::_1), 1);

        // ensure the robot is in default control
        msc_ = std::make_shared<unitree::robot::b2::MotionSwitcherClient>();
        msc_->SetTimeout(5.0f);
        msc_->Init();

        loco_client_ = std::make_shared<g1::LocoClient>();
        loco_client_->Init();
        loco_client_->SetTimeout(5.0f);

        toDefaultControl();

        lowcmd_thread_ = CreateRecurrentThreadEx("lowcmd", UT_CPU_ID_NONE, 2000, &G1HarwareInterface::LowCommandWriter, this);
    }
    
    void toUserControl() {
        std::lock_guard<std::mutex> lock(mode_switch_mutex_);
        // switch off the built-in control
        std::string form, name;
        while (msc_->CheckMode(form, name), !name.empty()) {
            if (msc_->ReleaseMode())
                std::cout << "Failed to switch to Release Mode\n";
            sleep(1);
        }
        // set a zero stiffness and small damping as fallback
        // they should be overwritten by the user control shortly
        for (int i = 0; i < this->njoints_; i++) {
            this->robot_data_.joint_stiffness[i] = 0.0f;
            this->robot_data_.joint_damping[i] = 4.0f;
        }

        // set the wrist joints to the default stiffness and damping
        for (int i : wrist_joint_indices) {
            this->robot_data_.q_target[i] = 0.0f;
            this->robot_data_.dq_target[i] = 0.0f;
            this->robot_data_.joint_stiffness[i] = default_wrist_stiffness;
            this->robot_data_.joint_damping[i] = default_wrist_damping;
        }

        std::cout << "Release Mode succeeded\n";
        this->control_state_ = ControlState::USER_CONTROL;
    }

    void toDefaultControl() {
        std::lock_guard<std::mutex> lock(mode_switch_mutex_);
        // switch on the built-in control
        int ret = msc_->SelectMode("ai");
        if (ret != 0) {
            std::cout << "Failed to switch to AI Mode\n";
        }
        std::cout << "Switching to AI Mode succeeded\n";
        this->control_state_ = ControlState::BUILTIN_CONTROL;
    }
    
    void toDampingMode() {
        std::lock_guard<std::mutex> lock(mode_switch_mutex_);
        this->control_state_ = ControlState::DAMPING_MODE;
        std::cout << "Switching to Damping Mode\n";
    }
    
    void updateControlFSM() {
        // Read gamepad state atomically
        bool l1_r1, l2_b;
        ControlState current_state;
        {
            std::lock_guard<std::mutex> lock(gamepad_mutex_);
            l1_r1 = gamepad_.L1.pressed && gamepad_.R1.pressed;
            l2_b = gamepad_.L2.pressed && gamepad_.B.pressed;
        }
        {
            std::lock_guard<std::mutex> lock(mode_switch_mutex_);
            current_state = control_state_;
        }
        
        // Check for state transitions (avoid nested locks)
        if (l1_r1 && current_state == ControlState::BUILTIN_CONTROL) {
            toUserControl();
        } else if (l2_b && current_state == ControlState::USER_CONTROL) {
            toDampingMode();
        } else if (l1_r1 && current_state == ControlState::DAMPING_MODE) {
            toDefaultControl();
        }
    }
    
    GamepadState getGamepadState() {
        std::lock_guard<std::mutex> lock(gamepad_mutex_);
        GamepadState state;
        
        // Copy analog values
        state.lx = gamepad_.lx;
        state.rx = gamepad_.rx;
        state.ry = gamepad_.ry;
        state.ly = gamepad_.ly;
        state.l2 = gamepad_.l2;
        
        // Copy button states
        auto copyButton = [](const unitree::common::Button& src, GamepadState::ButtonState& dst) {
            dst.pressed = src.pressed;
            dst.on_press = src.on_press;
            dst.on_release = src.on_release;
        };
        
        copyButton(gamepad_.R1, state.R1);
        copyButton(gamepad_.L1, state.L1);
        copyButton(gamepad_.start, state.start);
        copyButton(gamepad_.select, state.select);
        copyButton(gamepad_.R2, state.R2);
        copyButton(gamepad_.L2, state.L2);
        copyButton(gamepad_.F1, state.F1);
        copyButton(gamepad_.F2, state.F2);
        copyButton(gamepad_.A, state.A);
        copyButton(gamepad_.B, state.B);
        copyButton(gamepad_.X, state.X);
        copyButton(gamepad_.Y, state.Y);
        copyButton(gamepad_.up, state.up);
        copyButton(gamepad_.right, state.right);
        copyButton(gamepad_.down, state.down);
        copyButton(gamepad_.left, state.left);
        
        return state;
    }
};


// MuJoCo interface uses double (matching mjtNum, avoiding conversions)
class G1MujocoInterface: public G1Interface<double> {
private:
    std::thread physics_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> should_stop_;
    std::mutex step_mutex_;
    double timestep_;  // Physics timestep in ssereconds
    bool async_; // whether to run physics simulation in a separate thread

    void physicsStep() {
        std::lock_guard<std::mutex> lock(step_mutex_);
        if (this->model_ && this->data_) {
            // compute torques using PD control
            // tau = joint_stiffness * (q_target - q) + joint_damping * (0 - dq)
            for (int i = 0; i < this->njoints_; i++) {
                // Get current joint position and velocity from MuJoCo state
                double q_curr = this->data_->qpos[7 + i];
                double dq_curr = this->data_->qvel[6 + i];
                
                // Compute PD control torque
                double tau = this->robot_data_.joint_stiffness[i] * (this->robot_data_.q_target[i] - q_curr)
                            + this->robot_data_.joint_damping[i] * (this->robot_data_.dq_target[i] - dq_curr);
                
                // Set control input for actuator
                this->data_->ctrl[i] = tau;
            }

            mj_step(this->model_, this->data_);
        }
    }

    void updateState() {
        if (this->model_ && this->data_) {
            // Joint positions: qpos[7:] (skip root position + quaternion, copy up to 29 joints)
            if (this->njoints_ > 0) {
                std::copy(this->data_->qpos + 7, 
                         this->data_->qpos + 7 + this->njoints_,
                         this->robot_data_.q.begin());
            }
            
            // Joint velocities: qvel[6:] (skip root velocity, copy up to 29 joints)
            if (this->njoints_ > 0) {
                std::copy(this->data_->qvel + 6,
                         this->data_->qvel + 6 + this->njoints_,
                         this->robot_data_.dq.begin());
            }
            // Position: qpos[0:3] (3 elements: x, y, z)
            std::copy(this->data_->qpos,
                this->data_->qpos + 3,
                this->robot_data_.root_pos_w.begin());
            // Quaternion: qpos[3:7] (4 elements: w, x, y, z)
            std::copy(this->data_->qpos + 3,
                this->data_->qpos + 7,
                this->robot_data_.quaternion.begin());
            // Convert quaternion to RPY
            quatToRPY(this->data_->qpos + 3, this->robot_data_.rpy.data());
            
            // velocity: qvel[0:3] (3 elements: x, y, z)
            std::copy(this->data_->qvel,
                this->data_->qvel + 3,
                this->robot_data_.root_lin_vel_w.begin());
            // angular velocity: qvel[3:6] (3 elements: x, y, z)
            std::copy(this->data_->qvel + 3,
                this->data_->qvel + 6,
                this->robot_data_.root_ang_vel_w.begin());
            
            // Compute projected gravity: rotate gravity vector [0, 0, -1] from world to body frame
            Eigen::Quaterniond quat(this->robot_data_.quaternion[0], 
                                    this->robot_data_.quaternion[1], 
                                    this->robot_data_.quaternion[2], 
                                    this->robot_data_.quaternion[3]);
            Eigen::Vector3d gravity_world(0.0, 0.0, -1.0);
            Eigen::Vector3d gravity_body = quat.inverse() * gravity_world;
            this->robot_data_.projected_gravity[0] = gravity_body.x();
            this->robot_data_.projected_gravity[1] = gravity_body.y();
            this->robot_data_.projected_gravity[2] = gravity_body.z();
            
            mj_forward(this->model_, this->data_);

            // Update body positions from data_->xpos [nbody, 3] using Eigen::Map for efficient copy
            // excluding the world body
            
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> xpos_map(
                this->data_->xpos, this->model_->nbody, 3);
            this->robot_data_.body_positions = xpos_map.bottomRows(this->nbodies_).cast<double>();
            
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>> xquat_map(
                this->data_->xquat, this->model_->nbody, 4);
            this->robot_data_.body_quaternions = xquat_map.bottomRows(this->nbodies_).cast<double>();
        }
    }
    
    void physicsLoop() {        
        while (!should_stop_.load()) {
            auto iter_start_time = std::chrono::steady_clock::now();
            
            physicsStep();
            updateState();

            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<double>(current_time - iter_start_time).count();
            auto sleep_time = std::chrono::duration<double>(timestep_ - elapsed);
            if (sleep_time.count() > 0.0) {
                // Sleep to avoid busy-waiting
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(sleep_time.count() * 1000000.0)));
            }
        }
        
        running_.store(false);
    }

public:
    G1MujocoInterface(std::string mjcf_path, double timestep = -1.0) 
        : running_(false),
        should_stop_(false),
        timestep_(timestep),
        async_(false)
    {
        this->loadMJCF(mjcf_path);
        // Use model's timestep if not specified
        if (timestep_ < 0 && this->model_) {
            timestep_ = this->model_->opt.timestep;
        }
        updateState();
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
    
    void run(bool sync = false) {
        // zero initialize the targets in robot_data_
        for (int i = 0; i < this->njoints_; i++) {
            this->robot_data_.q_target[i] = 0.0;
            this->robot_data_.dq_target[i] = 0.0;
        }
        if (sync) {
            this->async_ = false;
        } else {
            this->async_ = true;
            if (!running_.load()) {
                should_stop_.store(false);
                running_.store(true);
                physics_thread_ = std::thread(&G1MujocoInterface::physicsLoop, this);
            } else {
                // throw error
                throw std::runtime_error("Physics thread already running");
            }
        }
    }
    
    void stop() {
        if (this->async_) {
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

    void reset(const std::array<double, 29> &joint_pos) {
        std::lock_guard<std::mutex> lock(step_mutex_);
        if (this->model_ && this->data_) {
            mj_resetData(this->model_, this->data_);
            std::copy(joint_pos.begin(), joint_pos.end(), this->data_->qpos + 7);
            std::copy(joint_pos.begin(), joint_pos.end(), this->robot_data_.q_target.begin());
        }
    }

    void step() {
        // throw an error if not in sync mode
        if (this->async_) {
            throw std::runtime_error("Not in sync mode");
        }
        physicsStep();
        updateState();
    }

};

