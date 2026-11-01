"""Collect ZP10D serial responses on Windows/Linux; probe never commands motion.

Requires only pyserial: python -m pip install pyserial
See playground/open_duck_mini_zp10d/SERVO_IDENTIFICATION.md before recording.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time


FIELDS = (
    "timestamp_s", "query_started_s", "reply_latency_s", "goal_pulse_us",
    "position_pulse_us", "goal_position_rad", "position_rad", "command_time_ms",
)


class ZP10D:
    def __init__(self, connection, servo_id, timeout=0.2, command_gap=0.02):
        self.connection = connection
        self.servo_id = servo_id
        self.timeout = timeout
        self.command_gap = command_gap
        self.started = time.perf_counter()
        self.transactions = []

    def exchange(self, suffix, reply_pattern=None):
        command = f"#{self.servo_id:03d}{suffix}!"
        event = {"command": command, "tx_started_s": time.perf_counter() - self.started}
        raw = bytearray()
        self.transactions.append(event)
        try:
            self.connection.reset_input_buffer()
            packet = command.encode("ascii")
            if self.connection.write(packet) != len(packet):
                raise OSError("Incomplete serial write")
            self.connection.flush()
            event["tx_finished_s"] = time.perf_counter() - self.started
            if reply_pattern is None:
                # CBM V2.1.16STG on CH340 drops an immediate query after a
                # position command; a 20 ms gap passed the actual hold test.
                event["post_write_gap_s"] = self.command_gap
                time.sleep(self.command_gap)
                return None
            deadline = time.perf_counter() + self.timeout
            while time.perf_counter() < deadline:
                raw.extend(self.connection.read(max(1, self.connection.in_waiting)))
                if len(raw) > 4096:
                    raise ValueError("Too much unexpected serial data")
                # Discard echoes, unrelated IDs and incomplete frames.
                for frame in re.findall(rb"#[^#!\r\n]{1,100}!", raw):
                    if frame == packet:
                        continue
                    match = re.fullmatch(reply_pattern.encode("ascii"), frame)
                    if match:
                        event["reply"] = frame.decode("ascii")
                        return match.group(1).decode("ascii")
            raise TimeoutError(f"No matching reply to {command}")
        except BaseException as error:
            event["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            event["finished_s"] = time.perf_counter() - self.started
            event["rx_hex"] = raw.hex()

    def position(self):
        pulse = int(self.exchange("PRAD", rf"#{self.servo_id:03d}P(\d{{4}})!"))
        if not 500 <= pulse <= 2500:
            raise ValueError(f"Out-of-range PRAD reply: {pulse}")
        return pulse

    def probe(self):
        try:
            version = self.exchange("PVER", rf"#{self.servo_id:03d}(?:PV|@\s*)([^!]+)!")
        except TimeoutError:
            version = None
        mode = int(self.exchange("PMOD", rf"#{self.servo_id:03d}P?MOD([1-8])!"))
        position = self.position()
        return {"version": version, "mode": mode, "position_pulse_us": position,
                "nominal_travel_deg": {1: 270, 2: 270, 3: 180, 4: 180}.get(mode)}


def make_plan(profile, amplitude):
    if profile == "slow":
        return [(0.0, 0, 1000), (1.5, amplitude, 1000),
                (3.0, -amplitude, 1000), (4.5, 0, 1000)], 6.0
    # Increasing step sizes, both directions; never larger than +/- amplitude.
    offsets = [0]
    for fraction in (0.25, 0.5, 1.0):
        offset = max(1, round(amplitude * fraction))
        offsets.extend([offset, 0, -offset, 0])
    return [(float(i), offset, 0) for i, offset in enumerate(offsets)], float(len(offsets))


def capture(device, args, report):
    probe = report["probe"]
    if probe["mode"] not in (1, 2, 3, 4):
        raise ValueError("Not in position mode (1-4); no movement command sent")
    center = getattr(args, "center_pulse", None)
    if center is None:
        center = probe["position_pulse_us"]
    report["center_pulse_us"] = center
    if not 600 <= center - args.amplitude <= center + args.amplitude <= 2400:
        raise ValueError("Insufficient pulse-range margin around current position")
    if args.amplitude > 100 and getattr(args, "max_deviation_deg", None) is None:
        raise ValueError("Larger motion requires an explicit confirmed feedback window (--max-deviation-deg)")
    plan, duration = make_plan(args.profile, args.amplitude)
    report["planned_commands"] = [
        {"time_s": t, "pulse_us": center + offset, "time_ms": ms}
        for t, offset, ms in plan
    ]
    report["planned_duration_s"] = duration
    # Relative angles only: the center is NOT assumed to be pendulum-down zero.
    scale = math.radians(probe["nominal_travel_deg"]) / 2000
    report["angle_reference"] = (
        "relative to fixed center_pulse_us; nominal mode scaling"
        if getattr(args, "center_pulse", None) is not None else
        "relative to initial PRAD; nominal mode scaling"
    )
    amplitude_deg = math.degrees(args.amplitude * scale)
    max_deviation_deg = getattr(args, "max_deviation_deg", None)
    if max_deviation_deg is not None and amplitude_deg >= max_deviation_deg:
        raise ValueError("Command amplitude must be smaller than the feedback deviation limit")
    deviation_limit = args.amplitude + 50
    if max_deviation_deg is not None:
        deviation_limit = min(deviation_limit, math.floor(math.radians(max_deviation_deg) / scale))
    report["feedback_deviation_limit_p"] = deviation_limit
    report["feedback_deviation_limit_deg"] = math.degrees(deviation_limit * scale)
    if abs(probe["position_pulse_us"] - center) > deviation_limit:
        raise ValueError("Initial position outside the fixed reference window; no motion command sent")
    print(f"Record {args.profile}: center P{center}, +/-{amplitude_deg:.3f} deg, "
          f"{duration:g} s, voltage {args.voltage:g} V", flush=True)
    goal, command_ms, index = center, 0, 0
    received = []
    record_start = time.perf_counter()
    report["record_started_s"] = record_start - device.started
    motion_attempted = False
    try:
        with (args.output / "samples.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            if getattr(args, "restore_torque", False):
                # The vendor defines PULR as restoring torque at the current
                # position. It changes runtime holding, not startup settings.
                motion_attempted = True
                try:
                    device.exchange("PULR", r"#(OK)!")
                    report["torque_restore_acknowledged"] = True
                except TimeoutError:
                    report["torque_restore_acknowledged"] = False
                    print("Torque restore has no ACK; actual holding is unverified", file=sys.stderr)
                position = device.position()
                if abs(position - center) > deviation_limit:
                    raise RuntimeError("Position moved outside the test window while restoring torque")
                record_start = time.perf_counter()
                report["record_started_s"] = record_start - device.started
            while time.perf_counter() - record_start < duration:
                elapsed = time.perf_counter() - record_start
                if index < len(plan) and elapsed >= plan[index][0]:
                    scheduled, offset, command_ms = plan[index]
                    if elapsed - scheduled > 0.1:
                        raise RuntimeError("Serial loop delayed >100 ms; aborting instead of skipping commands")
                    goal = center + offset
                    motion_attempted = True
                    device.exchange(f"P{goal:04d}T{command_ms:04d}")
                    index += 1
                position = device.position()
                event = device.transactions[-1]
                timestamp = event["finished_s"] - report["record_started_s"]
                received.append(timestamp)
                writer.writerow(dict(zip(FIELDS, (
                    timestamp, event["tx_started_s"] - report["record_started_s"],
                    event["finished_s"] - event["tx_started_s"], goal, position,
                    (goal - center) * scale, (position - center) * scale, command_ms,
                ))))
                stream.flush()
                if abs(position - center) > deviation_limit:
                    raise RuntimeError("Reported position outside the test window; stopping")
                time.sleep(max(0, 1 / args.hz - (time.perf_counter() - device.started
                                               - event["tx_started_s"])))
            report["status"] = "complete"
    finally:
        # Stop at the current angle on success, interruption or failed feedback.
        # Do not release torque or blindly command a return after an error.
        if motion_attempted:
            try:
                device.exchange("PDST", r"#(OK)!")
                report["stop_acknowledged"] = True
            except (OSError, TimeoutError, ValueError) as error:
                report["stop_acknowledged"] = False
                report["stop_error"] = str(error)
                print(f"STOP was not acknowledged: {error}", file=sys.stderr)
        intervals = [b - a for a, b in zip(received, received[1:])]
        report["samples"] = len(received)
        if intervals:
            report["actual_hz"] = 1 / statistics.mean(intervals)
            report["maximum_sample_gap_s"] = max(intervals)
            print(f"{len(received)} samples, {report['actual_hz']:.1f} Hz actual, "
                  f"max gap {max(intervals) * 1000:.1f} ms", flush=True)
        if report.get("status") == "complete" and not report.get("stop_acknowledged"):
            report["status"] = "complete_but_stop_unacknowledged"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("ports", "probe", "record"))
    parser.add_argument("--port", help="COM3 on Windows, /dev/ttyUSB0 on Linux")
    parser.add_argument("--id", type=int, default=0, dest="servo_id")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--output", type=Path, default=Path("zp10d_capture"))
    parser.add_argument("--voltage", type=float, help="Actual measured supply voltage, not a target")
    parser.add_argument("--profile", choices=("slow", "steps"), default="slow")
    parser.add_argument("--amplitude", type=int, default=50,
                        help="Max pulse offset (1-900 us); >100 requires an explicit feedback window; default +/-6.75 deg")
    parser.add_argument("--center-pulse", type=int,
                        help="Fixed reference P for repeated tests; default is initial PRAD")
    parser.add_argument("--hz", type=float, default=100, help="Requested feedback polling Hz (10-200)")
    parser.add_argument("--command-gap-ms", type=float, default=20,
                        help="Wait after a position write before querying (0-100 ms; CH340 default 20)")
    parser.add_argument("--max-deviation-deg", type=float,
                        help="Abort if feedback exceeds this nominal angle from the fixed/initial center")
    parser.add_argument("--restore-torque", action="store_true",
                        help="Request PULR at the current position before recording; does not change startup settings")
    parser.add_argument("--setup", default="unloaded; describe any horn/load separately",
                        help="Describe the fixture/load; stored verbatim")
    args = parser.parse_args()
    if args.action != "ports" and not args.port:
        parser.error("--port is required")
    if not 0 <= args.servo_id <= 254 or args.baud <= 0:
        parser.error("ID must be 0-254 (no broadcast); baud must be positive")
    if not 1 <= args.amplitude <= 900 or not 10 <= args.hz <= 200:
        parser.error("amplitude must be 1-900 us and hz must be 10-200")
    if args.center_pulse is not None and not 500 <= args.center_pulse <= 2500:
        parser.error("center-pulse must be in [500, 2500]")
    if not 0 <= args.command_gap_ms <= 100:
        parser.error("command-gap-ms must be 0-100")
    if args.max_deviation_deg is not None and not (
        math.isfinite(args.max_deviation_deg) and 0 < args.max_deviation_deg <= 180
    ):
        parser.error("max-deviation-deg must be finite and in (0, 180]")
    if args.action == "record" and (args.voltage is None or not 5 <= args.voltage <= 8.4):
        parser.error("record requires --voltage with actual ZP10D supply (5-8.4 V)")
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        parser.exit(1, "Install pyserial first: python -m pip install pyserial\n")
    if args.action == "ports":
        for port in list_ports.comports():
            print(f"{port.device}: {port.description}")
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / ("probe.json" if args.action == "probe" else "capture.json")
    if path.exists() or (args.action == "record" and (args.output / "samples.csv").exists()):
        parser.error(f"Existing data in {args.output}; use a different test-condition directory")
    report = dict(schema="zp10d_serial_capture_v1", action=args.action, status="incomplete",
                  created_utc=datetime.now(timezone.utc).isoformat(), servo_model="ZP10Dg (user reported)",
                  port=args.port, servo_id=args.servo_id, baud=args.baud,
                  voltage_v=args.voltage, voltage_source="operator supplied, not servo telemetry",
                  serial_format="8N1; no hardware/software flow control",
                  requested_hz=args.hz, profile=args.profile if args.action == "record" else None,
                  command_gap_ms=args.command_gap_ms,
                  fixed_center_pulse_us=args.center_pulse,
                  max_deviation_deg=args.max_deviation_deg,
                  restore_torque_requested=args.restore_torque if args.action == "record" else False,
                  setup=args.setup, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  timestamp_convention="PRAD response received; query bounds retained; not hardware timestamps",
                  bam_ready=False)
    device = None
    exit_code = 0
    try:
        connection = serial.Serial(
            port=None, baudrate=args.baud, bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
            xonxoff=False, rtscts=False, dsrdtr=False, timeout=0.005, write_timeout=0.2,
        )
        connection.port = args.port
        connection.dtr = False
        connection.rts = False
        with connection:
            device = ZP10D(connection, args.servo_id, command_gap=args.command_gap_ms / 1000)
            try:
                report["probe"] = device.probe()
                print(json.dumps(report["probe"], indent=2))
                if args.action == "record":
                    capture(device, args, report)
                else:
                    report["status"] = "complete_read_only"
            finally:
                report["transactions"] = device.transactions
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        report.update(status="aborted", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr)
        exit_code = 1
    finally:
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved {path}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
