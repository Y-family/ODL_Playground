"""Fit an equivalent ZP10D response using BAM M1/M2 and the CAD-estimated load.

Run with scripts/cad/run_memory_limited.py. This does not write robot parameters.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bam-source', type=Path, required=True)
    parser.add_argument('--model', choices=('m1', 'm2'), default='m1')
    parser.add_argument('--iterations', type=int, default=120)
    parser.add_argument('--train-capture', default='upright_steps_8p1deg')
    parser.add_argument('--validation-capture', default='upright_validation_5p4deg')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
        parser.error('Run through scripts/cad/run_memory_limited.py')
    if args.iterations < 1:
        parser.error('iterations must be positive')
    sys.path.insert(0, str(args.bam_source.resolve()))
    import numpy as np
    from bam.actuator import Actuator
    from bam.model import models
    from bam.parameter import Parameter
    from bam.simulate import Simulator
    from bam.testbench import Testbench
    from scipy.optimize import differential_evolution

    root = Path(__file__).resolve().parents[2]
    data = root / 'playground/open_duck_mini_zp10d/training/actuator_capture/s09_s10_71g'
    rig_path = data / 'rig_geometry.json'
    rig = json.loads(rig_path.read_text())
    # This is a rigid assembly, not a point load at the S10 output shaft.
    class CadLoad(Testbench):
        def __init__(self, log):
            self.inertia = log['load_inertia_kg_m2']
            self.first_moment = log['gravity_first_moment_kg_m']
            self.phase = log['gravity_phase_rad']

        def compute_mass(self, q, dq):
            return self.inertia

        def compute_bias(self, q, dq):
            return -9.80665 * self.first_moment * np.sin(q + self.phase)

    class EquivalentServo(Actuator):
        """Effective position gain; no assumed firmware KP or motor kt/R."""
        def __init__(self):
            super().__init__(CadLoad)

        def initialize(self):
            self.model.position_gain = Parameter(13.37, .1, 60.)
            self.model.armature = Parameter(.001, 1e-7, .04)

        def control_unit(self):
            return 'Nm (equivalent model, not measured torque)'

        def compute_control(self, target, q, dq, dt):
            return self.model.position_gain.value * (target - q)

        def compute_torque(self, control, torque_enable, q, dq):
            # The project's existing cap is a fixed prior, not identified by
            # these small-load records. Do not extrapolate it to 7.8 V.
            return np.clip(control, -.980665, .980665) * torque_enable

        def get_extra_inertia(self):
            return self.model.armature.value

    load = dict(load_inertia_kg_m2=rig['inertia_about_S09_kg_m2'],
                gravity_first_moment_kg_m=rig['gravity_first_moment_kg_m'],
                gravity_phase_rad=math.pi)

    def make_model():
        model = models[args.model]()
        model.set_actuator(EquivalentServo())
        model.actuator.load_log(load)
        return model

    def self_check():
        model = make_model()
        model.friction_base.value = model.friction_viscous.value = 0.
        if model.stribeck:
            model.friction_stribeck.value = 0.
        sim = Simulator(model)
        sim.reset(.01, 0.)
        sim.step(0., True, .001)
        assert sim.dq > 0, 'Gravity must destabilize the upright pose'
        model.actuator.testbench.phase = 0.
        sim.reset(.01, 0.)
        sim.step(0., True, .001)
        assert sim.dq < 0, 'Gravity must restore the downward pose'
        assert math.isclose(model.actuator.testbench.compute_mass(0., 0.), load['load_inertia_kg_m2'])
        assert model.actuator.compute_torque(100., True, 0., 0.) == .980665
        assert model.actuator.compute_torque(100., False, 0., 0.) == 0.
        # Candidate batching must reproduce the scalar BAM dynamics.
        scalar, batch = Simulator(make_model()), Simulator(make_model())
        batch.reset(np.zeros(2), np.zeros(2))
        for _ in range(20):
            scalar.step(.2, True, .002)
            batch.step(np.array([.2, .2]), True, .002)
        assert np.allclose(batch.q, scalar.q, rtol=0., atol=1e-14)
        assert np.allclose(batch.dq, scalar.dq, rtol=0., atol=1e-14)
        print('PASS: BAM rigid-load inertia, upright/downward gravity, torque cap, enable and batching', flush=True)

    self_check()

    dt = .002

    def read_capture(name):
        path = data / name
        meta = json.loads((path / 'capture.json').read_text())
        rows = list(csv.DictReader((path / 'samples.csv').open()))
        if meta['status'] not in ('complete', 'complete_but_stop_unacknowledged'):
            raise ValueError(f'Incomplete capture: {name}')
        if meta['center_pulse_us'] != 1229 or meta['voltage_v'] != 7.8:
            raise ValueError('Unexpected reference or voltage')
        if len(rows) != meta['samples']:
            raise ValueError('Sample count mismatch')
        commands = []
        for event in meta['transactions']:
            match = re.fullmatch(r'#000P(\d{4})T(\d{4})!', event['command'])
            if match:
                if int(match[2]) != 0:
                    raise ValueError('Only T0000 fits are supported; exclude firmware interpolation')
                commands.append((event['tx_finished_s'] - meta['record_started_s'],
                                 math.radians((int(match[1])-1229) * .135)))
            elif event.get('error') and event['command'] != '#000PDST!':
                raise ValueError('Capture contains a communication error during movement')
        t = np.array([float(r['timestamp_s']) for r in rows])
        q = np.array([float(r['position_rad']) for r in rows])
        if not (len(commands) >= 3 and np.all(np.isfinite(q)) and np.all(np.diff(t) > 0)):
            raise ValueError('Invalid samples or commands')
        if np.ptp(q) < math.radians(1.5) or np.max(np.abs(q)) > math.radians(meta['feedback_deviation_limit_deg']):
            raise ValueError('Insufficient excitation or position outside the confirmed window')
        grid = np.arange(t[0], t[-1], dt)
        return dict(name=name, times=grid, observed=np.interp(grid, t, q),
                    command_times=np.array([-np.inf] + [v[0] for v in commands]),
                    goals=np.array([q[0]] + [v[1] for v in commands]),
                    hashes={p.name: digest(p) for p in (path/'capture.json',path/'samples.csv')})

    if args.train_capture == args.validation_capture:
        parser.error('Training and validation captures must be different')
    train = read_capture(args.train_capture)
    validation = read_capture(args.validation_capture)

    def rollout(x, log, keep=False):
        # Vectorizing candidate models uses BAM's original dynamics and M1
        # friction equations, with one native thread and no process pool.
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        model = make_model()
        model.position_gain.value = np.exp(x[0])
        model.armature.value = np.exp(x[1])
        model.friction_base.value = x[2]
        model.friction_viscous.value = x[3]
        model.q_offset.value = x[5]
        if model.stribeck:
            model.friction_stribeck.value = x[6]
            model.dtheta_stribeck.value = x[7]
            model.alpha.value = x[8]
        sim = Simulator(model)
        sim.reset(np.full(x.shape[1], log['observed'][0]), np.zeros(x.shape[1]))
        errors = np.zeros(x.shape[1])
        predicted = []
        torques = []
        for i, t in enumerate(log['times']):
            errors += (sim.q - log['observed'][i]) ** 2
            if keep:
                predicted.append(float(sim.q[0]))
            # Actual send times and zero-order-held targets preserve steps.
            # A transport delay does not turn them into linear ramps.
            index = np.searchsorted(log['command_times'], t-x[4], side='right')-1
            target = log['goals'][index]
            control = model.actuator.compute_control(target, sim.q, sim.dq, dt)
            if keep:
                torques.append(float(model.actuator.compute_torque(control, True, sim.q, sim.dq)[0]))
            # BAM's Stribeck branch uses += on friction_base. With batched
            # Parameter values this aliases and mutates the optimizer input.
            model.friction_base.value = x[2].copy()
            sim.step(control, True, dt)
        return (np.array(predicted), np.array(torques)) if keep else errors / len(log['times'])

    bounds = [(math.log(.1),math.log(60)),(math.log(1e-7),math.log(.04)),
              (0.,.15),(0.,1.),(0.,.10),(-.05,.05)]
    if args.model == 'm2':
        bounds += [(0.,.2),(.1,1.),(1.,10.)]
    check = np.array([math.log(2.6),math.log(.001),.05,.1,.03,0.]
                     + ([.05,.2,1.35] if args.model == 'm2' else []))
    batch = np.repeat(check[:,None],2,axis=1)
    original = batch.copy()
    batch_loss = rollout(batch, train)
    assert np.array_equal(batch, original), 'BAM must not mutate candidate parameters'
    assert np.allclose(batch_loss, rollout(check, train), rtol=0., atol=1e-14)
    print('PASS: complete recorded rollout matches scalar/batch and leaves candidates unchanged', flush=True)
    if args.self_check:
        return
    completed = 0

    def progress(x, convergence):
        nonlocal completed
        completed += 1
        if completed % 5 == 0:
            print(f'BAM {args.model.upper()} optimization generation {completed}/{args.iterations}', flush=True)

    result = differential_evolution(lambda x: rollout(x, train), bounds, seed=0,
                                    popsize=8, maxiter=args.iterations, tol=.003,
                                    polish=False, workers=1, updating='deferred',
                                    vectorized=True, callback=progress)
    names = ['position_gain_Nm_per_rad','armature_kg_m2','frictionloss_Nm',
             'damping_Nm_s_per_rad','transport_delay_s','mounting_offset_rad']
    if args.model == 'm2':
        names += ['stribeck_friction_Nm','stribeck_velocity_rad_s','stribeck_alpha']
    values = [math.exp(result.x[0]),math.exp(result.x[1]),*result.x[2:]]
    metrics, series = {}, {}
    for log in (train, validation):
        predicted, torques = rollout(result.x, log, keep=True)
        error = np.degrees(predicted-log['observed'])
        metrics[log['name']] = dict(mae_deg=float(np.mean(np.abs(error))),
                                    rmse_deg=float(np.sqrt(np.mean(error**2))),
                                    peak_error_deg=float(np.max(np.abs(error))),
                                    predicted_peak_motor_torque_Nm=float(np.max(np.abs(torques))),
                                    predicted_torque_cap_fraction=float(np.mean(np.abs(torques) >= .980665-1e-10)))
        series[log['name']] = predicted
    output = args.output_dir or data / 'bam'
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='preliminary_equivalent_response_fit', actuator='ZP10Dg equivalent position actuator',
                  bam_commit=subprocess.check_output(['git','-C',str(args.bam_source),'rev-parse','HEAD'],text=True).strip(),
                  model=f'BAM {args.model.upper()}', source_script_sha256=digest(Path(__file__)),
                  rig_sha256=digest(rig_path), source_capture_hashes={v['name']:v['hashes'] for v in (train,validation)},
                  parameters=dict(zip(names,values)), metrics=metrics, simulation_dt_s=dt,
                  optimizer=dict(generations=result.nit, vectorized_evaluations=result.nfev, converged=bool(result.success),
                                 termination_message=str(result.message), seed=0, workers=1),
                  near_search_bound=[names[i] for i,(lo,hi) in enumerate(bounds)
                                     if min(result.x[i]-lo,hi-result.x[i]) < .02*(hi-lo)],
                  torque_cap_prior_Nm=.980665, torque_cap_measured=False, formal_robot_model_updated=False,
                  validation_used_for_optimization=False,
                  validation_used_for_model_diagnosis=True,
                  limitations=['Single 71 g CAD-estimated load; mass distribution and physical angle scale not measured',
                               'Equivalent P gain, not internal firmware gain; no motor kt/R or current identified',
                               'Delay includes host/serial/feedback timing effects; first 20 ms after commands are not sampled',
                               'Small-load position fits do not uniquely establish all physical parameters or stall torque',
                               'Validation only covers the named held-out trajectory at 7.8 V'])
    (output/f'{args.model}_preliminary.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    for ax, log in zip(axes,(train,validation)):
        goals = log['goals'][np.searchsorted(log['command_times'], log['times'], side='right')-1]
        ax.step(log['times'],np.degrees(goals),where='post',label='Sent target',color='gray',alpha=.5)
        ax.plot(log['times'],np.degrees(log['observed']),label='Measured PRAD')
        ax.plot(log['times'],np.degrees(series[log['name']]),label=f'BAM {args.model.upper()} prediction')
        ax.set(title=log['name'],xlabel='Time [s]',ylabel='Angle from upright [deg]')
        ax.grid(alpha=.3)
        ax.legend()
    fig.savefig(output/f'{args.model}_preliminary.png',dpi=150)
    print(json.dumps(dict(parameters=report['parameters'],metrics=metrics,
                          near_search_bound=report['near_search_bound']),indent=2),flush=True)


if __name__ == '__main__':
    main()
