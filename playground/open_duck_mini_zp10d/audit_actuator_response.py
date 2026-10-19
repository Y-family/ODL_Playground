"""Audit inherited motor response without robot load or changes to formal files.

Use scripts/cad/run_memory_limited.py. This is a simulation consistency check,
not a physical ZP10Dg calibration. Voltage alone does not identify motor response.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import xml.etree.ElementTree as ET

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / 'xmls/open_duck_mini_zp10d.xml'
LEGACY = ROOT.parent / 'open_duck_mini_v2/xmls/open_duck_mini_v2.xml'
MANUAL_SHA256 = '9905405f676031379ceaa51b25bd9743c269d278e0f10fd08f538f67a532e265'


def manual_evidence(path):
    """Facts transcribed from text and embedded tables in this exact manual."""
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != MANUAL_SHA256:
        raise ValueError('Manual differs from the reviewed PDF; recheck page/table applicability')
    return dict(path=str(path.resolve()), sha256=sha, pages=28,
        model_table=dict(page=12, model='ZP10D', supply_voltage_v=[5., 8.4],
                         torque_kgf_cm=10., torque_test_voltage_v=None, angle_deg=270),
        electrical_table=dict(page=14, model='unspecified; not established as ZP10D or ZP10Dg',
            rows=[dict(voltage_v=6., seconds_per_60_deg=.16, stall_torque_kgf_cm=13.5),
                  dict(voltage_v=6.6, seconds_per_60_deg=.14, stall_torque_kgf_cm=15.)],
            applicability='table torque differs from the 10 kgf cm ZP10D model row'),
        running_torque_guidance=dict(page=18, fraction_of_stall=1 / 3,
            conditional_nm_at_existing_cap=.980665 / 3,
            scope='manufacturer general guidance, not a measured continuous-duty rating'),
        bus_control=dict(pages=[17, 25, 26, 27], default_baud=115200,
            position_command='#000P1500T1000!', angle_read_command='#000PRAD!',
            note='T is requested movement duration, not a measured physical speed limit'),
        missing=['ZP10Dg suffix specification', 'model-specific no-load speed at approximately 8 V',
                 'torque test voltage', 'transient response, armature, gain and delay'],
        calibration_status='manual reviewed; exact-model response not identified; no model update')


def parameters(path, name):
    default = ET.parse(path).find(f'.//default[@class="{name}"]')
    joint, position = default.find('joint'), default.find('position')
    return dict(kp=float(position.get('kp')), damping=float(joint.get('damping')),
                frictionloss=float(joint.get('frictionloss')), armature=float(joint.get('armature')),
                torque_cap=max(abs(float(v)) for v in position.get('forcerange').split()))


def manual_6p6v_parameters(current, manual):
    """User-selected generic 6.6 V assumption; unknown response terms inherited."""
    row = next(row for row in manual['electrical_table']['rows'] if row['voltage_v'] == 6.6)
    cap = row['stall_torque_kgf_cm'] * .0980665
    speed = np.pi / (3 * row['seconds_per_60_deg'])
    return dict(current, torque_cap=cap, damping=(cap - current['frictionloss']) / speed)


def run(config, *, position_control=False):
    cap = config['torque_cap']
    actuator = (f'<position kp="{config["kp"]}" kv="0"' if position_control else '<motor')
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option gravity="0 0 0" timestep="0.002"><flag eulerdamp="disable"/></option>
      <worldbody><body><joint name="shaft" axis="0 0 1" limited="false"
        damping="{config['damping']}" frictionloss="{config['frictionloss']}" armature="{config['armature']}"/>
        <inertial pos="0 0 0" mass="0.000001" diaginertia="1e-9 1e-9 1e-9"/>
      </body></worldbody>
      <actuator>{actuator} joint="shaft" forcelimited="true" forcerange="{-cap} {cap}"/></actuator>
    </mujoco>''')
    data = mujoco.MjData(model)
    data.ctrl[0] = np.pi / 3 if position_control else cap
    rows = []
    # Allow lower-damping what-if cases to settle before checking terminal speed.
    duration = 1.5 if position_control else max(1.5, 20 * config['armature'] / config['damping'])
    for _ in range(round(duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
        rows.append([data.time, data.qpos[0], data.qvel[0], data.actuator_force[0]])
    samples = np.array(rows)
    assert np.isfinite(samples).all() and not any(data.warning.number)
    assert abs(samples[:, 3]).max() <= cap + 1e-12
    result = dict(final_angle_deg=float(np.rad2deg(data.qpos[0])),
                  final_velocity_rad_s=float(data.qvel[0]), peak_torque_nm=float(abs(samples[:, 3]).max()))
    if position_control:
        result['angle_after_0p2_s_deg'] = float(np.rad2deg(samples[99, 1]))
        result['first_90pct_target_time_s'] = (float(samples[samples[:, 1] >= .9 * np.pi / 3, 0][0])
                                             if (samples[:, 1] >= .9 * np.pi / 3).any() else None)
    else:
        expected = (cap - config['frictionloss']) / config['damping']
        assert abs(data.qvel[0] - expected) < 1e-7
        result['analytic_velocity_rad_s'] = expected
        result['equivalent_60deg_constant_speed_time_s'] = float((np.pi / 3) / expected)
    return result, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed-60-s', type=float, help='datasheet no-load seconds per 60 degrees')
    parser.add_argument('--speed-voltage-v', type=float, help='voltage corresponding to that datasheet speed')
    parser.add_argument('--manual', type=Path, help='verify the reviewed user PDF and include its scoped evidence')
    args = parser.parse_args()
    if ((args.speed_60_s is None) != (args.speed_voltage_v is None)
            or (args.speed_60_s is not None and (not np.isfinite([args.speed_60_s, args.speed_voltage_v]).all()
                                                or args.speed_60_s <= 0 or args.speed_voltage_v <= 0))):
        parser.error('provide a positive speed time and its voltage together')
    manual = manual_evidence(args.manual) if args.manual else None
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (CURRENT, LEGACY)}
    current, legacy = parameters(CURRENT, 'zp10d'), parameters(LEGACY, 'sts3215')
    report = dict(user_supply_voltage_v_approx=8., user_model_name='ZP10Dg',
                  voltage_source='user statement; approximate; not measured',
                  current_parameters=current, legacy_parameters=legacy,
                  scope='isolated shaft, zero gravity and external load; 1e-9 kg m2 dummy link inertia',
                  torque_voltage_scope='existing user torque cap retained; not rescaled from voltage',
                  calibration_status='speed, armature, gain and delay not identified; no model update',
                  manual=manual, cases={})
    arrays = {}
    for name, config in [('current', current), ('legacy', legacy)]:
        for label, position in [('constant_torque', False), ('60deg_step', True)]:
            key = name + '_' + label
            report['cases'][key], arrays[key] = run(config, position_control=position)
    if manual:
        report['generic_manual_sensitivity'] = {}
        for row in manual['electrical_table']['rows']:
            name = f'generic_{row["voltage_v"]:g}v'
            speed = np.pi / (3 * row['seconds_per_60_deg'])
            candidate = dict(current, damping=(current['torque_cap'] - current['frictionloss']) / speed)
            result = dict(source_row=row, parameters=candidate,
                scope='what-if only: generic speed with existing torque cap/friction at an unverified voltage; '
                      'inherited armature and gain; not a ZP10Dg calibration or 8 V extrapolation')
            for label, position in [('constant_torque', False), ('60deg_step', True)]:
                result[label], arrays[name + '_' + label] = run(candidate, position_control=position)
            report['generic_manual_sensitivity'][name] = result
        candidate = manual_6p6v_parameters(current, manual)
        result = dict(parameters=candidate, voltage_v=6.6, seconds_per_60_deg=.14,
            stall_torque_kgf_cm=15., scope='user-selected generic 6.6 V assumption; '
                'linear torque-speed approximation; inherited gain, armature and friction; no formal model update')
        for label, position in [('constant_torque', False), ('60deg_step', True)]:
            result[label], arrays['manual_6p6v_' + label] = run(candidate, position_control=position)
        report['manual_6p6v_assumption'] = result
    if args.speed_60_s is not None:
        speed = np.pi / (3 * args.speed_60_s)
        report['datasheet_speed'] = dict(seconds_per_60_deg=args.speed_60_s, voltage_v=args.speed_voltage_v)
        report['conditional_linear_model'] = dict(
            effective_drag_nm_s_rad=(current['torque_cap'] - current['frictionloss']) / speed,
            assumptions='linear torque-speed relation; existing dry friction and torque cap assumed at this voltage',
            status='candidate only; cannot identify armature, kp, deadband or delay from no-load speed')
    else:
        report['datasheet_speed'] = None
    assert before == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    report['protected_source_hashes'] = before
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = ROOT / 'training'
    output.mkdir(exist_ok=True)
    (output / 'actuator_response.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(output / 'actuator_response.npz', **arrays)
    print(json.dumps(report, indent=2), flush=True)
    print('PASS: analytic/native speed agreement and unchanged source models', flush=True)


if __name__ == '__main__':
    main()
