"""No-hardware regression checks: python -m unittest discover -s scripts -p test_collect_zp10d.py"""

import argparse
import csv
import io
from pathlib import Path
import re
import tempfile
import time
import unittest
from unittest.mock import patch

import collect_zp10d as collector


class SerialEmulator:
    def __init__(self, mode=1, cbm_firmware=False):
        self.mode = mode
        self.cbm_firmware = cbm_firmware
        self.current = 1500
        self.buffer = bytearray()
        self.commands = []
        self.fail_position = False

    @property
    def in_waiting(self):
        return len(self.buffer)

    def reset_input_buffer(self):
        self.buffer.clear()

    def flush(self):
        pass

    def write(self, packet):
        command = packet.decode("ascii")
        self.commands.append(command)
        if command == "#007PVER!":
            reply = "#007@ ZL-ZServo_AD_CBM V2.1.16STG!\r\n" if self.cbm_firmware else "#007PV0.97!"
        elif command == "#007PMOD!":
            reply = f"#007MOD{self.mode}!" if self.cbm_firmware else f"#007PMOD{self.mode}!"
        elif command == "#007PRAD!":
            reply = "" if self.fail_position else f"#007P{self.current:04d}!"
        elif command in ("#007PDST!", "#007PULR!"):
            reply = "#OK!"
        else:
            match = re.fullmatch(r"#007P(\d{4})T\d{4}!", command)
            if not match:
                raise ValueError(f"Unexpected command: {command}")
            self.current = int(match.group(1))
            reply = ""
        # Echo and wrong-ID replies must never count as actual feedback.
        self.buffer.extend((command + "#008P1999!" + reply).encode("ascii"))
        return len(packet)

    def read(self, count):
        # Exercise fragmented frames, even if in_waiting advertises more.
        count = min(count, 3)
        data = self.buffer[:count]
        del self.buffer[:count]
        return bytes(data)


class CaptureTests(unittest.TestCase):
    def test_probe_is_read_only_and_matches_id_not_echo(self):
        serial = SerialEmulator(mode=3)
        result = collector.ZP10D(serial, 7).probe()
        self.assertEqual(result["version"], "0.97")
        self.assertEqual(result["position_pulse_us"], 1500)
        self.assertEqual(result["nominal_travel_deg"], 180)
        self.assertEqual(serial.commands, ["#007PVER!", "#007PMOD!", "#007PRAD!"])

    def test_missing_feedback_never_uses_wrong_id(self):
        serial = SerialEmulator()
        serial.fail_position = True
        device = collector.ZP10D(serial, 7, timeout=0.01)
        with self.assertRaises(TimeoutError):
            device.position()
        self.assertIn("error", device.transactions[-1])
        self.assertTrue(device.transactions[-1]["rx_hex"])

    def test_actual_cbm_firmware_version_and_mode_replies(self):
        serial = SerialEmulator(cbm_firmware=True)
        result = collector.ZP10D(serial, 7).probe()
        self.assertEqual(result["version"], "ZL-ZServo_AD_CBM V2.1.16STG")
        self.assertEqual(result["mode"], 1)
        self.assertEqual(result["position_pulse_us"], 1500)

    def test_record_data_and_stop_without_return_on_error(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                serial = SerialEmulator()
                device = collector.ZP10D(serial, 7, timeout=0.01)
                report = {"probe": device.probe()}
                args = argparse.Namespace(output=Path(directory), profile="steps", amplitude=50,
                                          voltage=6.6, hz=200)
                plan = [(0.0, 25, 0), (0.025, -25, 0), (0.05, 0, 0)]
                if fail:
                    serial.fail_position = True
                with patch.object(collector, "make_plan", return_value=(plan, 0.075)), \
                     patch("sys.stdout", new_callable=io.StringIO):
                    if fail:
                        with self.assertRaises(TimeoutError):
                            collector.capture(device, args, report)
                    else:
                        collector.capture(device, args, report)
                self.assertEqual(serial.commands[-1], "#007PDST!")
                self.assertTrue(report["stop_acknowledged"])
                with (Path(directory) / "samples.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if fail:
                    self.assertEqual(rows, [])
                    self.assertEqual(serial.commands[-2], "#007PRAD!")
                else:
                    self.assertEqual(report["status"], "complete")
                    self.assertGreaterEqual(len(rows), 3)
                    times = [float(row["timestamp_s"]) for row in rows]
                    self.assertEqual(times, sorted(set(times)))
                    self.assertEqual(serial.current, 1500)
                    for row in rows:
                        self.assertGreaterEqual(float(row["reply_latency_s"]), 0)
                        self.assertEqual(row["goal_pulse_us"], row["position_pulse_us"])

    def test_motor_mode_refused_before_motion(self):
        serial = SerialEmulator(mode=5)
        device = collector.ZP10D(serial, 7)
        report = {"probe": device.probe()}
        with self.assertRaises(ValueError):
            collector.capture(device, argparse.Namespace(), report)
        self.assertFalse(any("T" in cmd for cmd in serial.commands))

    def test_query_after_position_write_waits_for_bus(self):
        class SpacedSerial(SerialEmulator):
            def __init__(self):
                super().__init__()
                self.position_sent = None

            def write(self, packet):
                if re.fullmatch(rb"#007P\d{4}T\d{4}!", packet):
                    self.position_sent = time.perf_counter()
                if packet == b"#007PRAD!" and self.position_sent is not None:
                    if time.perf_counter() - self.position_sent < 0.015:
                        self.commands.append(packet.decode())
                        return len(packet)  # emulate the observed lost query
                return super().write(packet)

        for gap in (0, 0.02):
            with self.subTest(command_gap=gap):
                serial = SpacedSerial()
                device = collector.ZP10D(serial, 7, timeout=0.01, command_gap=gap)
                device.exchange("P1525T1000")
                if gap == 0:
                    with self.assertRaises(TimeoutError):
                        device.position()
                else:
                    self.assertEqual(device.position(), 1525)

    def test_physical_feedback_window_stops_motion(self):
        class OvershootingSerial(SerialEmulator):
            def position_after_command(self):
                return any(re.fullmatch(r"#007P\d{4}T\d{4}!", cmd) for cmd in self.commands)

            def write(self, packet):
                if packet == b"#007PRAD!" and self.position_after_command():
                    self.current = 1575  # 10.125 degrees from the initial position.
                return super().write(packet)

        serial = OvershootingSerial()
        device = collector.ZP10D(serial, 7, timeout=.01)
        report = {"probe": device.probe()}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory), profile="steps", amplitude=60,
                                      voltage=7.8, hz=100, max_deviation_deg=10)
            with patch.object(collector, "make_plan", return_value=([(0., 60, 0)], .075)), \
                 patch("sys.stdout", new_callable=io.StringIO):
                with self.assertRaisesRegex(RuntimeError, "outside the test window"):
                    collector.capture(device, args, report)
        self.assertEqual(serial.commands[-1], "#007PDST!")
        self.assertEqual(report["samples"], 1)
        self.assertLessEqual(report["feedback_deviation_limit_deg"], 10)

    def test_torque_restore_precedes_recording_without_startup_writes(self):
        serial = SerialEmulator()
        device = collector.ZP10D(serial, 7, timeout=.01)
        report = {"probe": device.probe()}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory), profile="steps", amplitude=20,
                                      voltage=7.8, hz=100, restore_torque=True)
            with patch.object(collector, "make_plan", return_value=([(0., 20, 0)], .075)), \
                 patch("sys.stdout", new_callable=io.StringIO):
                collector.capture(device, args, report)
        self.assertTrue(report["torque_restore_acknowledged"])
        self.assertLess(serial.commands.index("#007PULR!"), serial.commands.index("#007P1520T0000!"))
        self.assertFalse(any("PCS" in cmd for cmd in serial.commands))

    def test_fixed_reference_prevents_window_drift_between_tests(self):
        serial = SerialEmulator()
        device = collector.ZP10D(serial, 7, timeout=.01)
        report = {"probe": device.probe()}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory), profile="steps", amplitude=60,
                                      voltage=7.8, hz=100, max_deviation_deg=10, center_pulse=1520)
            with patch.object(collector, "make_plan", return_value=([(0., 0, 0)], .075)), \
                 patch("sys.stdout", new_callable=io.StringIO):
                collector.capture(device, args, report)
        self.assertIn("#007P1520T0000!", serial.commands)
        self.assertEqual(report["center_pulse_us"], 1520)
        report["probe"]["position_pulse_us"] = 1400
        before = list(serial.commands)
        with self.assertRaisesRegex(ValueError, "outside the fixed reference window"):
            collector.capture(device, args, report)
        self.assertEqual(serial.commands, before)

    def test_larger_motion_requires_explicit_window_and_retains_pulse_margin(self):
        serial = SerialEmulator()
        device = collector.ZP10D(serial, 7, timeout=.01)
        report = {"probe": device.probe()}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory), profile="steps", amplitude=333,
                                      voltage=7.8, hz=100, center_pulse=1500)
            before = list(serial.commands)
            with self.assertRaisesRegex(ValueError, "explicit confirmed feedback window"):
                collector.capture(device, args, report)
            self.assertEqual(serial.commands, before)
            args.max_deviation_deg = 55
            with patch.object(collector, "make_plan", return_value=([(0., 333, 0)], .025)), \
                 patch("sys.stdout", new_callable=io.StringIO):
                collector.capture(device, args, report)
            self.assertIn("#007P1833T0000!", serial.commands)
            self.assertLessEqual(report["feedback_deviation_limit_deg"], 55)
            before = list(serial.commands)
            args.center_pulse = 700
            with self.assertRaisesRegex(ValueError, "pulse-range margin"):
                collector.capture(device, args, report)
            self.assertEqual(serial.commands, before)


if __name__ == "__main__":
    unittest.main()
