#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

#include <mujoco/mujoco.h>


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


struct RobotData {
    std::array<float, 29> q;
    std::array<float, 29> dq;
    std::array<float, 29> tau;

    std::array<float, 4> quaternion;
    std::array<float, 3> rpy;
    std::array<float, 3> omega;
};


class G1Interface {
    private:
        RobotData robot_data_;
        mjModel *model_;
        mjData *data_;
    public:
        ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
        ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;

        void LowStateCallback(const void *message) {
            LowState_ low_state = *(const LowState_ *)message;
            if (low_state.crc() != Crc32Core((uint32_t *)&low_state, (sizeof(LowState_) >> 2) - 1)) {
                std::cout << "[ERROR] CRC Error" << std::endl;
                return;
            }
            for (int i = 0; i < 29; i++) {
                robot_data_.q[i] = low_state.motor_state()[i].q();
                robot_data_.dq[i] = low_state.motor_state()[i].dq();
                robot_data_.tau[i] = low_state.motor_state()[i].tau_est();
            }
            robot_data_.quaternion = low_state.imu_state().quaternion();
            robot_data_.rpy = low_state.imu_state().rpy();
            robot_data_.omega = low_state.imu_state().gyroscope();
        }
    
    G1Interface(std::string networkInterface) {
        ChannelFactory::Instance()->Init(0, networkInterface);

        lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>("rt/lowstate"));
        lowstate_subscriber_->InitChannel(std::bind(&G1Interface::LowStateCallback, this, std::placeholders::_1), 1);

        // lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>("rt/lowcmd"));
        // lowcmd_publisher_->InitChannel();
        std::string mjcf_path = "../mjcf/g1.xml";
    }
    
    void loadMJCF(std::string mjcf_path) {
        char error[1000] = {0};
        model_ = mj_loadXML(mjcf_path.c_str(), NULL, error, 1000);
        
        if (!this->model_) {
            std::cerr << "ERROR: Failed to load MuJoCo model from " << mjcf_path << std::endl;
            std::cerr << "Error: " << error << std::endl;
            throw std::runtime_error("Failed to load MuJoCo model");
        }
        
        mjData *data = mj_makeData(this->model_);
        if (!data) {
            std::cerr << "ERROR: Failed to create MuJoCo data" << std::endl;
            mj_deleteModel(this->model_);
            throw std::runtime_error("Failed to create MuJoCo data");
        }
        
        std::cout << "Model loaded successfully" << std::endl;
        std::cout << "Model has " << this->model_->nq << " DOFs" << std::endl;
    }
    
    RobotData getData() const {
        return robot_data_;
    }
};