import struct
import time
from enum import Enum
from typing import Any, Optional, List

import serial
import os

from dataclasses import dataclass, field

class IMUType(Enum):
    MAIN_IMU = 0
    FRAME_IMU = 1

class TriggerPin(Enum):
    RC_Input_Roll = 1
    RC_Input_Pitch = 2
    EXT_FC_Input_Roll = 3
    EXT_FC_Input_Pitch = 4
    RC_Input_Yaw = 5
    Pin_AUX1 = 16
    Pin_AUX2 = 17
    Pin_AUX3 = 18
    Pin_Buzzer = 32
    Pin_SSAT_Power = 33

class TriggerPinState(Enum):
    LOW = 0
    HIGH = 1 # 3.3V
    FLOATING = 2

class ControlMode(Enum):
    No_Control = 0
    Ignore = 7
    Speed = 1
    Angle = 2
    Angle_Shortest = 8
    Speed_Angle = 3
    RC = 4
    RC_High_Res = 6
    Angle_Rel_Frame = 5

class ControlModeFlags(Enum):
    Auto_Task = 1 << 6 # for MODE_ANGLE, MODE_ANGLE_SHORTEST, MODE_ANGLE_REL_FRAME
    Force_RC_Speed = 1 << 6 # for MODE_RC
    High_Res_Speed = 1 << 7 # for all modes
    Target_Precise = 1 << 5 # for MODE_ANGLE, MODE_ANGLE_SHORTEST, MODE_ANGLE_REL_FRAM
    Mix_Follow = 1 << 4 # for MODE_SPEED, MODE_ANGLE, MODE_ANGLE_SHORTEST

class ModeFlags(Enum):
    FLAG_DISABLE_ANGLE_ERR_CORR = 1 << 0 # Disable angle error correction
    Default = 0
    
class MotorAction(Enum):
    MOTOR_OFF_FLOATING = 1
    MOTOR_OFF_BRAKE = 2
    MOTOR_OFF_SAFE = 3  
    MOTOR_ON = 4
    HOME_POSITION = 5
    SEARCH_HOME = 6

def generate_control_mode(mode: ControlMode, flags: ControlModeFlags = ControlModeFlags.High_Res_Speed) -> int:
    if mode not in ControlMode:
        raise ValueError("Invalid control mode")
    if flags not in ControlModeFlags:
        raise ValueError("Invalid control mode flags")

    return (mode.value & 0x0F) | (flags.value & 0xF0)

@dataclass
class BoardInfo:
    board_version : Optional[int] = 0 # 1 byte unsigned integer
    firmware_version : Optional[int] = 0 # 2 bytes unsigned integer
    state_flags : Optional[int] = 0 # 1 byte unsigned integer
    board_features : Optional[int] = 0 # 2 byte unsigned integer
    connection_flag : Optional[int] = 0 # 1 byte unsigned integer
    frw_extra_id : Optional[int] = 0 # 4 bytes unsigned integer
    board_features_ext : Optional[int] = 0 # 2 bytes unsigned integer
    main_imu_sens_model : Optional[int] = 0 # 1 byte unsigned integer
    frame_imu_sens_model : Optional[int] = 0 # 1 byte unsigned integer
    build_number : Optional[int] = 0 # 1 byte unsigned integer
    base_frw_ver : Optional[int] = 0 # 2 bytes unsigned integer

@dataclass
class RealtimeData:
    axis_1_acc_data: Optional[int] = None
    axis_1_gyro_data: Optional[int] = None
    axis_2_acc_data: Optional[int] = None
    axis_2_gyro_data: Optional[int] = None
    axis_3_acc_data: Optional[int] = None
    axis_3_gyro_data: Optional[int] = None
    serial_err_cnt: Optional[int] = None
    system_error: Optional[int] = None
    system_sub_error: Optional[int] = None
    rc_roll: Optional[int] = None
    rc_pitch: Optional[int] = None
    rc_yaw: Optional[int] = None
    rc_cmd: Optional[int] = None
    ext_fc_roll: Optional[int] = None
    ext_fc_pitch: Optional[int] = None
    imu_angles: List[int] = field(default_factory=list)
    frame_imu_angles: List[int] = field(default_factory=list)
    target_angles: List[int] = field(default_factory=list)
    cycle_time: Optional[int] = None
    i2c_error_cnt: Optional[int] = None
    bat_level: Optional[int] = None
    rt_data_flags: Optional[int] = None
    cur_imu: Optional[IMUType] = None
    cur_profile: Optional[int] = None
    motor_power: List[int] = field(default_factory=list)

@dataclass
class MotorStateDataSet:
    control_mode : Optional[bool] = False # 0. bit
    torque : Optional[bool] = False # 1. bit
    torque_setpoint : Optional[bool] = False # 2. bit
    speed32 : Optional[bool] = False # 5. bit
    speed32_setpoint : Optional[bool] = False # 6. bit
    angle32 : Optional[bool] = False # 9. bit
    angle32_setpoint : Optional[bool] = False # 10. bit

@dataclass
class MotorStateData:
    control_mode : Optional[int] = 0 # 0: position, 1: speed, 2: torque
    torque : Optional[int] = 0 # 2 bytes signed integer
    torque_setpoint : Optional[int] = 0 # 2 bytes signed integer
    speed32 : Optional[int] = 0 # 4 bytes signed integer (unit: micro-radians/sec)
    speed32_setpoint : Optional[int] = 0 # 4 bytes signed integer (unit: micro-radians/sec)
    angle32 : Optional[int] = 0 # 4 bytes signed integer (unit: 0,00034332275390625 deg)
    angle32_setpoint : Optional[int] = 0 # 4 bytes signed integer (unit: 0,00034332275390625 deg)

@dataclass
class Angles:
    axis_1_imu_angle: Optional[int] = None  # 2 bytes signed integer (unit: 0.02197265625 degrees)
    axis_1_target_angle: Optional[int] = None  # 2 bytes signed integer (unit: 0.02197265625 degrees)
    axis_2_imu_angle: Optional[int] = None  # 2 bytes signed integer (unit: 0.02197265625 degrees)
    axis_2_target_angle: Optional[int] = None  # 2 bytes signed integer (unit: 0.02197265625 degrees)
    axis_3_imu_angle: Optional[int] = None  # 2 bytes signed integer (unit: 0.02197265625 degrees)
    axis_3_target_angle: Optional[int] = None  # 2 bytes signed integer (unit: 0.02197265625 degrees)
    target_speed: Optional[int] = None  # 2 bytes signed integer (unit: 0.1220740379 degree/sec)

# Some magical constants for Serial protocol
START_BIT = 0x24
SBGC_CRC16_POLYNOM = 0x8005

# Some message IDs to use
CMD_BOARD_INFO = 86 # çalışıyor
CMD_SELECT_IMU_3 = 24 # çalışıyor
CMD_REALTIME_DATA_3 = 23 # çalışıyor
CMD_GET_ANGLES = 73 # çalışıyor
CMD_EXT_MOTORS_STATE = 131 # çalışmıyor
CMD_CONTROL_EXT = 121 # çalışıyor
CMD_CONTROL = 67 # test et
CMD_TRIGGER_PIN = 84 # çalışıyor
CMD_CONFIRM = 67
CMD_ERROR = 255
CMD_EXT_MOTORS_ACTION = 128

# todo: imu vs. mesajları için confirm ve error mesajlarını kontrol et

def crc16(data) -> int:
    """Calculate CRC16 using the SBGC polynomial"""
    crc_register = 0
    
    for byte in data:
        for shift_register in [1, 2, 4, 8, 16, 32, 64, 128]:
            data_bit = 1 if (byte & shift_register) else 0
            crc_bit = crc_register >> 15
            crc_register = (crc_register << 1) & 0xFFFF
            
            if data_bit != crc_bit:
                crc_register ^= SBGC_CRC16_POLYNOM
    
    return crc_register

connection = serial.Serial('/dev/ttyUSB1', 115200, timeout=1)

def read_packet(ser) -> tuple[bytes | None, int]:
    while True:
        byte = ser.read(1)
        if not byte:
            print("Timeout!")
            return None, 0
        if byte[0] == 0x24:  # start byte '$'
            break

    header = ser.read(3)
    if len(header) < 3:
        print("Header is missing!")
        return None, 0

    cmd_id = header[0]
    length = header[1]
    header_checksum = header[2]

    if (cmd_id + length) % 256 != header_checksum:
        print("Header checksum is wrong!")
        return None, 0

    # payload + CRC16
    payload = ser.read(length)
    crc_bytes = ser.read(2)
    if len(payload) < length or len(crc_bytes) < 2:
        print("Missing payload/CRC16!")
        return None, 0

    crc_expected = crc_bytes[0] | (crc_bytes[1] << 8)
    crc_data = bytes([cmd_id, length, header_checksum]) + payload
    crc_actual = crc16(crc_data)
    if crc_actual != crc_expected:
        print(f"CRC mismatch!")
        return None, 0

    print(f"Command response retrieved : ID={cmd_id}, Payload={payload.hex()}")
    return payload, cmd_id

def get_board_info() -> BoardInfo | None:
    message = generate_message(CMD_BOARD_INFO, bytearray())
    connection.write(message)
    
    time.sleep(0.01)

    payload, _ = read_packet(connection)
    if payload is None or len(payload) < 18: # payload is 18 bytes
        return None

    decoded_payload = struct.unpack('<BHBHBIHBBBH', payload)
    board_info = BoardInfo()
    board_info.board_version = decoded_payload[0]
    board_info.firmware_version = decoded_payload[1]
    board_info.state_flags = decoded_payload[2]
    board_info.board_features = decoded_payload[3]
    board_info.connection_flag = decoded_payload[4]
    board_info.frw_extra_id = decoded_payload[5]
    board_info.board_features_ext = decoded_payload[6]
    board_info.main_imu_sens_model = decoded_payload[7]
    board_info.frame_imu_sens_model = decoded_payload[8]
    board_info.build_number = decoded_payload[9]
    board_info.base_frw_ver = decoded_payload[10]
    return board_info

def select_imu(imu_type : IMUType):
    message = generate_message(CMD_SELECT_IMU_3, bytearray([imu_type.value]))
    connection.write(message)

def get_realtime_data() -> RealtimeData | None:
    message = generate_message(CMD_REALTIME_DATA_3, bytearray())
    connection.write(message)

    time.sleep(0.01)

    payload, _ = read_packet(connection)
    if payload is None or len(payload) < 63: # payload is 63 bytes
        return None

    data = RealtimeData()
    data.axis_1_acc_data = struct.unpack('<h', payload[0:2])[0]
    data.axis_1_gyro_data = struct.unpack('<h', payload[2:4])[0]
    data.axis_2_acc_data = struct.unpack('<h', payload[4:6])[0]
    data.axis_2_gyro_data = struct.unpack('<h', payload[6:8])[0]
    data.axis_3_acc_data = struct.unpack('<h', payload[8:10])[0]
    data.axis_3_gyro_data = struct.unpack('<h', payload[10:12])[0]
    data.serial_err_cnt = struct.unpack('<H', payload[12:14])[0]
    data.system_error = struct.unpack('<H', payload[14:16])[0]
    data.system_sub_error = payload[16]
    # 3 bytes reserved
    data.rc_roll = struct.unpack('<h', payload[20:22])[0]
    data.rc_pitch = struct.unpack('<h', payload[22:24])[0]
    data.rc_yaw = struct.unpack('<h', payload[24:26])[0]
    data.rc_cmd = struct.unpack('<h', payload[26:28])[0]
    data.ext_fc_roll = struct.unpack('<h', payload[28:30])[0]
    data.ext_fc_pitch = struct.unpack('<h', payload[30:32])[0]
    data.imu_angles = [struct.unpack('<h', payload[32+i:34+i])[0] for i in range(0, 6, 2)]
    data.frame_imu_angles = [struct.unpack('<h', payload[38+i:40+i])[0] for i in range(0, 6, 2)]
    data.target_angles = [struct.unpack('<h', payload[44+i:46+i])[0] for i in range(0, 6, 2)]
    data.cycle_time = struct.unpack('<H', payload[50:52])[0]
    data.i2c_error_cnt = struct.unpack('<H', payload[52:54])[0]
    # error code is deprecated, use system_error instead, skip it
    data.bat_level = struct.unpack('<H', payload[55:57])[0]
    data.rt_data_flags = payload[57]
    data.cur_imu = IMUType(payload[58])
    data.cur_profile = payload[59]
    data.motor_power = [payload[60+i] for i in range(3)]

    return data

def get_angles() -> Angles | None:
    message = generate_message(CMD_GET_ANGLES, bytearray())
    connection.write(message)

    time.sleep(0.01)

    payload, _ = read_packet(connection)
    if payload is None or len(payload) < 14: # payload is 14 bytes
        return None

    angles = Angles()
    angles.axis_1_imu_angle = struct.unpack('<h', payload[0:2])[0]
    angles.axis_1_target_angle = struct.unpack('<h', payload[2:4])[0]
    angles.axis_2_imu_angle = struct.unpack('<h', payload[4:6])[0]
    angles.axis_2_target_angle = struct.unpack('<h', payload[6:8])[0]
    angles.axis_3_imu_angle = struct.unpack('<h', payload[8:10])[0]
    angles.axis_3_target_angle = struct.unpack('<h', payload[10:12])[0]
    angles.target_speed = struct.unpack('<h', payload[12:14])[0]

    return angles


def set_trigger_pin(pin : TriggerPin, state : TriggerPinState):
    if pin not in TriggerPin or state not in TriggerPinState:
        raise ValueError("Invalid pin or state")

    payload = bytearray()
    payload.append(pin.value)  # 1 byte for pin
    payload.append(state.value)  # 1 byte for state

    message = generate_message(CMD_TRIGGER_PIN, payload)
    connection.write(message)

def control_motors(mode : ControlMode, axis1_speed : int, axis2_speed : int, axis3_speed : int, axis1_angle : int, axis2_angle : int, axis3_angle : int) -> bool:
    if mode not in ControlMode:
        raise ValueError("Invalid control mode")

    payload = bytearray()
    for i in range(3):
        payload.append(generate_control_mode(mode))  # 1 byte for control mode

    # 3 bytes for axis speeds
    payload.extend(struct.pack('<h', axis1_speed))
    payload.extend(struct.pack('<h', axis1_angle))
    payload.extend(struct.pack('<h', axis2_speed))
    payload.extend(struct.pack('<h', axis2_angle))
    payload.extend(struct.pack('<h', axis3_speed))
    payload.extend(struct.pack('<h', axis3_angle))

    message = generate_message(CMD_CONTROL, payload)
    connection.write(message)

    time.sleep(0.03)
    payload, res_id = read_packet(connection)
    if payload is None:  # payload is 3 bytes for minimum
        return False

    if res_id == CMD_CONFIRM:
        return True

    elif res_id != CMD_CONFIRM:
        error_response = parse_error(payload)
        print(f"Error: Command ID {error_response.cmd_id} returned error code {error_response.error_code}")

    return False

def control_motors_ext(enable_speed : bool, enable_angle : bool, axis_data : dict[str, dict[str, int]]) -> bool:
    if enable_speed == False and enable_angle == False:
        raise ValueError("At least one of enable_speed or enable_angle must be True")

    data_set_part = 0
    if enable_speed:
        data_set_part |= (1 << 3)
    if enable_angle:
        data_set_part |= (1 << 2)

    data_set = 0
    if "axis_1" in axis_data:
        data_set |= data_set_part
    if "axis_2" in axis_data:
        data_set |= (data_set_part << 5)
    if "axis_3" in axis_data:
        data_set |= (data_set_part << 10)

    payload = bytearray()
    payload.extend(struct.pack('<H', data_set))

    def append_axis_data(data: dict[str, int]) -> None:
        payload.append(struct.pack('<B', data['control_mode'])[0])
        payload.append(struct.pack('<B', data['mode_flags'])[0])

        if enable_speed:
            payload.extend(struct.pack('<i', data['speed']))
        if enable_angle:
            payload.extend(struct.pack('<i', data['angle']))

    if "axis_1" in axis_data:
        append_axis_data(axis_data["axis_1"])
    if "axis_2" in axis_data:
        append_axis_data(axis_data["axis_2"])
    if "axis_3" in axis_data:
        append_axis_data(axis_data["axis_3"])

    message = generate_message(CMD_CONTROL_EXT, payload)
    connection.write(message)

    time.sleep(0.01)

    payload, res_id = read_packet(connection)
    if payload is None or len(payload) < 3:  # payload is 3 bytes for minimum
        return False

    if res_id == CMD_CONFIRM:
        return True

    if res_id != CMD_CONFIRM:
        error_response = parse_error(payload)
        print(f"Error: Command ID {error_response.cmd_id} returned error code {error_response.error_code}")

    return False

def get_motors_state(motor_id : int, data_set : MotorStateDataSet) -> MotorStateData | None:
    if motor_id < 0 or motor_id > 6:
        raise ValueError("Invalid motor id. Must be between 0 and 6")

    payload = bytearray()
    payload.append(motor_id)  # 1 byte for motor id

    dataset = 0
    if data_set.control_mode:
        dataset |= (1 << 0)
    if data_set.torque:
        dataset |= (1 << 1)
    if data_set.torque_setpoint:
        dataset |= (1 << 2)
    if data_set.speed32:
        dataset |= (1 << 5)
    if data_set.speed32_setpoint:
        dataset |= (1 << 6)
    if data_set.angle32:
        dataset |= (1 << 9)
    if data_set.angle32_setpoint:
        dataset |= (1 << 10)
    payload.extend(struct.pack('<I', dataset))

    message = generate_message(CMD_EXT_MOTORS_STATE, payload)
    connection.write(message)

    time.sleep(0.01)

    response, _ = read_packet(connection)
    if response is None or len(response) < 12: # minimum length of response is 12 bytes
        return None

    i = 0
    data = MotorStateData()

    if data_set.control_mode:
        data.control_mode = response[i]
        i += 1
    if data_set.torque:
        data.torque = struct.unpack('<h', response[i:i+2])[0]
        i += 2
    if data_set.torque_setpoint:
        data.torque_setpoint = struct.unpack('<h', response[i:i+2])[0]
        i += 2
    if data_set.speed32:
        data.speed32 = struct.unpack('<i', response[i:i+4])[0]
        i += 4
    if data_set.speed32_setpoint:
        data.speed32_setpoint = struct.unpack('<i', response[i:i+4])[0]
        i += 4
    if data_set.angle32:
        data.angle32 = struct.unpack('<i', response[i:i+4])[0]
        i += 4
    if data_set.angle32_setpoint:
        data.angle32_setpoint = struct.unpack('<i', response[i:i+4])[0]
        i += 4
    if i != len(response):
        print(f"Warning: Not all data was read! Expected {len(response)} bytes, read {i} bytes.")

    return data

def set_motors_action(motor_id: int, action: MotorAction) -> bool:
    if motor_id < 0 or motor_id > 6:
        raise ValueError("Invalid motor id. Must be between 0 and 6")
    
    if action not in MotorAction:
        raise ValueError("Invalid motor action")

    payload = bytearray()
    payload.append(motor_id)  # 1 byte for motor id
    payload.append(action.value)  # 1 byte for action

    message = generate_message(CMD_EXT_MOTORS_ACTION, payload)
    connection.write(message)

    time.sleep(0.01)

    response, res_id = read_packet(connection)
    if response is None or len(response) < 1:  # minimum length of response is 1 byte
        return False

    print("ğ")
    for i, b in enumerate(response):
        print("{}: {}".format(i, b))

    if res_id != CMD_CONFIRM:
        error_response = parse_error(response)
        print(f"Error: Command ID {error_response.cmd_id} returned error code {error_response.error_code}")

    return response[0] == CMD_CONFIRM

def execute_menu(cmd_id : int) -> bool:
    if cmd_id < 0 or cmd_id > 255:
        raise ValueError("Invalid command id. Must be between 0 and 255")

    payload = bytearray()
    payload.append(cmd_id)  # 1 byte for command id

    message = generate_message(69, payload)
    connection.write(message)

    time.sleep(0.01)

    response, res_id = read_packet(connection)
    if response is None or len(response) < 1:  # minimum length of response is 1 byte
        return False

    if res_id != CMD_CONFIRM:
        error_response = parse_error(response)
        print(f"Error: Command ID {error_response.cmd_id} returned error code {error_response.error_code}")

    return res_id == CMD_CONFIRM

def generate_message(command_id : int, payload : bytearray) -> bytes:
    if command_id < 0 or command_id > 255:
        raise ValueError("Invalid command id. Must be between 0 and 255")
    
    header = bytearray()
    header.append(command_id) # Add command id (1 byte unsigned)
    header.append(len(payload)) # Add payload length (1 byte unsigned)
    header.append(((command_id + len(payload)) % 256)) # Add header checksum (1 byte unsigned)

    message = bytearray()
    message.append(START_BIT) # Add start bit "$"
    message.extend(header)

    message.extend(payload) # Add payload bytes
    crc_input = header + payload
    message.extend(struct.pack("<H", crc16(crc_input)))

    return message

@dataclass
class ErrorResponse:
    cmd_id: int
    error_code: int

def parse_error(payload: bytes) -> ErrorResponse:
    if len(payload) < 2:
        return "Invalid error response"

    cmd_id = payload[0]
    error_code = payload[1]
    
    return ErrorResponse(cmd_id=cmd_id, error_code=error_code)

if __name__ == "__main__":
    #print(get_board_info())
    #print(get_realtime_data())
    #print(get_angles())
    #set_trigger_pin(TriggerPin.Pin_AUX3, TriggerPinState.HIGH)
    #time.sleep(5)
    #set_trigger_pin(TriggerPin.Pin_AUX3, TriggerPinState.LOW)
    #select_imu(IMUType.FRAME_IMU)
    #print(get_realtime_data())
    #axis = {"axis_1": {"speed": 10, "control_mode": generate_control_mode(mode = ControlMode.Speed), "mode_flags": ModeFlags.Default.value}}
    #print(control_motors_ext(enable_speed = True, enable_angle = False, axis_data = axis))

    #print(generate_control_mode(ControlMode.RC, ControlModeFlags.High_Res_Speed))
    #print(execute_menu(11))

    # bu çalışıyor
    """print(control_motors(mode = ControlMode.No_Control, axis1_speed = 0, axis2_speed = 0, axis3_speed = 0, axis1_angle = 0, axis2_angle = 0, axis3_angle = 0))
    time.sleep(0.5)
    print(control_motors(mode = ControlMode.Speed_Angle, axis1_speed = 50, axis2_speed = 50, axis3_speed = 50, axis1_angle = 0, axis2_angle = -20000, axis3_angle = 4000)) # 2.si pitch 3.sü yaw
    time.sleep(2)
    print(control_motors(mode = ControlMode.No_Control, axis1_speed = 0, axis2_speed = 0, axis3_speed = 0, axis1_angle = 0, axis2_angle = 0, axis3_angle = 0))"""

    # bu çalışıyor speed iel komtrol
    print(control_motors(mode=ControlMode.No_Control, axis1_speed=0, axis2_speed=0, axis3_speed=0, axis1_angle=0,
                         axis2_angle=0, axis3_angle=0))
    time.sleep(0.5)
    print(control_motors(mode=ControlMode.Speed, axis1_speed=0, axis2_speed=-5000, axis3_speed=32000, axis1_angle=0,
                         axis2_angle=-20000, axis3_angle=16000))  # 2.si pitch 3.sü yaw
    time.sleep(5)
    print(control_motors(mode=ControlMode.No_Control, axis1_speed=0, axis2_speed=0, axis3_speed=0, axis1_angle=0,
                         axis2_angle=0, axis3_angle=0))

    """print(control_motors(mode=ControlMode.No_Control, axis1_speed=0, axis2_speed=0, axis3_speed=0, axis1_angle=0,
                         axis2_angle=0, axis3_angle=0))
    time.sleep(0.5)
    print(control_motors(mode=ControlMode.RC, axis1_speed=0, axis2_speed=-5000, axis3_speed=5000, axis1_angle=0,
                         axis2_angle=0, axis3_angle=-500))  # 2.si pitch 3.sü yaw
    time.sleep(12)
    print(control_motors(mode=ControlMode.No_Control, axis1_speed=0, axis2_speed=0, axis3_speed=0, axis1_angle=0,
                         axis2_angle=0, axis3_angle=0))"""