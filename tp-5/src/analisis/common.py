from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "output"


def format_float(value: float) -> str:
    """Formats floats for both CLI args and folder names."""
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    if text == "":
        text = "0"
    return text.replace(".", "p")


@dataclass(frozen=True)
class SimulationParams:
    model: str = "sfm"
    agents: int = 30
    domain: float = 6.0
    rc: float = 0.5
    dt: float = 0.002
    output_interval: float = 0.02
    steps: Optional[int] = 10_000
    duration: Optional[float] = None
    seed: int = 1234
    min_radius: float = 0.18
    max_radius: float = 0.21
    desired_speed: float = 1.7
    relaxation: float = 0.5
    sfm_a: float = 2000.0
    sfm_b: float = 0.08
    sfm_k: float = 120000.0
    sfm_kappa: float = 240000.0
    sfm_tau: float = 0.5
    aacpm_kc: float = 2.0
    aacpm_beta: float = 2.0
    aacpm_k: float = 120000.0
    aacpm_kappa: float = 240000.0
    aacpm_kavo: float = 1.5
    aacpm_tau: float = 1.0
    aacpm_delta: float = 0.05
    aacpm_omega: float = 2.0
    aacpm_vdes: float = 1.0
    aacpm_vmax: float = 2.0
    aacpm_alpha: float = 0.5
    output_base: Path = field(default_factory=lambda: DEFAULT_OUTPUT_BASE)
    output_override: Optional[Path] = None

    def with_agents(self, agents: int) -> "SimulationParams":
        return replace(self, agents=agents, output_override=None)

    def with_seed(self, seed: int) -> "SimulationParams":
        return replace(self, seed=seed, output_override=None)

    def with_output(self, output: Path) -> "SimulationParams":
        return replace(self, output_override=output)

    def resolved_output_base(self) -> Path:
        return (self.output_base if self.output_base.is_absolute()
                else (PROJECT_ROOT / self.output_base)).resolve()

    def default_output_dir(self) -> Path:
        tag: str
        if self.steps is not None:
            tag = f"steps{self.steps}"
        elif self.duration is not None:
            tag = f"dur{format_float(self.duration)}"
        else:
            tag = "steps10000"
        dirname = f"{self.model}_n{self.agents}_seed{self.seed}_{tag}"
        return self.resolved_output_base() / dirname

    def output_dir(self) -> Path:
        if self.output_override is not None:
            base = (self.output_override if self.output_override.is_absolute()
                    else (PROJECT_ROOT / self.output_override))
            return base.resolve()
        return self.default_output_dir().resolve()

    def to_make_vars(self, output_dir: Path) -> List[str]:
        vars_ = [
            f"MODEL={self.model}",
            f"N={self.agents}",
            f"L={self.domain}",
            f"RC={self.rc}",
            f"DT={self.dt}",
            f"DT2={self.output_interval}",
            f"SEED={self.seed}",
            f"MIN_RADIUS={self.min_radius}",
            f"MAX_RADIUS={self.max_radius}",
            f"VD={self.desired_speed}",
            f"RELAX={self.relaxation}",
            f"SFM_A={self.sfm_a}",
            f"SFM_B={self.sfm_b}",
            f"SFM_K={self.sfm_k}",
            f"SFM_KAPPA={self.sfm_kappa}",
            f"SFM_TAU={self.sfm_tau}",
            f"AACPM_KC={self.aacpm_kc}",
            f"AACPM_BETA={self.aacpm_beta}",
            f"AACPM_K={self.aacpm_k}",
            f"AACPM_KAPPA={self.aacpm_kappa}",
            f"AACPM_KAVO={self.aacpm_kavo}",
            f"AACPM_TAU={self.aacpm_tau}",
            f"AACPM_DELTA={self.aacpm_delta}",
            f"AACPM_OMEGA={self.aacpm_omega}",
            f"AACPM_VDES={self.aacpm_vdes}",
            f"AACPM_VMAX={self.aacpm_vmax}",
            f"AACPM_ALPHA={self.aacpm_alpha}",
            f"OUTPUT={output_dir}",
            f"OUTPUT_DIR={self.resolved_output_base()}",
        ]
        if self.steps is not None:
            vars_.append(f"STEPS={self.steps}")
        if self.duration is not None:
            vars_.append(f"DURATION={self.duration}")
        return vars_

    def to_java_cli_dict(self) -> dict[str, str]:
        """Return CLI key/value pairs for invoking the Java simulation."""
        args = {
            "model": self.model,
            "agents": str(self.agents),
            "domain": str(self.domain),
            "rc": str(self.rc),
            "dt": str(self.dt),
            "outputInterval": str(self.output_interval),
            "seed": str(self.seed),
            "minRadius": str(self.min_radius),
            "maxRadius": str(self.max_radius),
            "desiredSpeed": str(self.desired_speed),
            "relaxation": str(self.relaxation),
            "sfmA": str(self.sfm_a),
            "sfmB": str(self.sfm_b),
            "sfmK": str(self.sfm_k),
            "sfmKappa": str(self.sfm_kappa),
            "sfmTau": str(self.sfm_tau),
            "aacpmKc": str(self.aacpm_kc),
            "aacpmBeta": str(self.aacpm_beta),
            "aacpmK": str(self.aacpm_k),
            "aacpmKappa": str(self.aacpm_kappa),
            "aacpmKavo": str(self.aacpm_kavo),
            "aacpmTau": str(self.aacpm_tau),
            "aacpmDelta": str(self.aacpm_delta),
            "aacpmOmega": str(self.aacpm_omega),
            "aacpmVdes": str(self.aacpm_vdes),
            "aacpmVmax": str(self.aacpm_vmax),
            "aacpmAlpha": str(self.aacpm_alpha),
        }
        if self.steps is not None:
            args["steps"] = str(self.steps)
        elif self.duration is not None:
            args["duration"] = str(self.duration)
        args["output-base"] = str(self.resolved_output_base())
        if self.output_override is not None:
            args["output"] = str(self.output_dir())
        return args


def add_simulation_arguments(parser) -> None:
    parser.add_argument("--model", default="sfm", help="Dynamics model to use (sfm or aacpm).")
    parser.add_argument("--agents", type=int, default=30, help="Default number of agents.")
    parser.add_argument("--domain", type=float, default=6.0, help="Square domain size L.")
    parser.add_argument("--rc", type=float, default=0.5, help="Interaction radius for CIM.")
    parser.add_argument("--dt", type=float, default=0.002, help="Integration step dt.")
    parser.add_argument("--output-interval", type=float, default=0.02, help="Interval for state logging.")
    parser.add_argument("--steps", type=int, default=10_000, help="Number of integration steps.")
    parser.add_argument("--duration", type=float, help="Alternative to steps: total simulated time.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for initial conditions.")
    parser.add_argument("--min-radius", type=float, default=0.18, help="Minimum agent radius.")
    parser.add_argument("--max-radius", type=float, default=0.21, help="Maximum agent radius.")
    parser.add_argument("--desired-speed", type=float, default=1.7, help="Desired walking speed.")
    parser.add_argument("--relaxation", type=float, default=0.5, help="Relaxation time (SFM).")
    parser.add_argument("--sfm-a", type=float, default=2000.0, help="SFM social force amplitude A.")
    parser.add_argument("--sfm-b", type=float, default=0.08, help="SFM social force range B.")
    parser.add_argument("--sfm-k", type=float, default=120000.0, help="SFM body stiffness k.")
    parser.add_argument("--sfm-kappa", type=float, default=240000.0, help="SFM friction kappa.")
    parser.add_argument("--sfm-tau", type=float, default=0.5, help="SFM relaxation time tau.")
    parser.add_argument("--aacpm-kc", type=float, default=2.0, help="AACPM contractile gain.")
    parser.add_argument("--aacpm-beta", type=float, default=2.0, help="AACPM anisotropy exponent.")
    parser.add_argument("--aacpm-k", type=float, default=120000.0, help="AACPM body stiffness.")
    parser.add_argument("--aacpm-kappa", type=float, default=240000.0, help="AACPM friction coefficient.")
    parser.add_argument("--aacpm-kavo", type=float, default=1.5, help="AACPM avoidance gain.")
    parser.add_argument("--aacpm-tau", type=float, default=1.0, help="AACPM avoidance time threshold.")
    parser.add_argument("--aacpm-delta", type=float, default=0.05, help="AACPM avoidance distance delta.")
    parser.add_argument("--aacpm-omega", type=float, default=2.0, help="AACPM alignment rate.")
    parser.add_argument("--aacpm-vdes", type=float, default=1.0, help="AACPM desired speed scaling.")
    parser.add_argument("--aacpm-vmax", type=float, default=2.0, help="AACPM max speed.")
    parser.add_argument("--aacpm-alpha", type=float, default=0.5, help="AACPM avoidance speed multiplier.")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE, help="Folder where simulations are stored.")
    parser.add_argument("--output", type=Path, help="Override output folder for a single run.")


def params_from_args(args) -> SimulationParams:
    steps = args.steps
    if args.duration is not None:
        steps = None
    output_base = args.output_base if isinstance(args.output_base, Path) else Path(args.output_base)
    output_override = args.output if args.output is None else Path(args.output)
    return SimulationParams(
        model=args.model.lower(),
        agents=args.agents,
        domain=args.domain,
        rc=args.rc,
        dt=args.dt,
        output_interval=args.output_interval,
        steps=steps,
        duration=args.duration,
        seed=args.seed,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        desired_speed=args.desired_speed,
        relaxation=args.relaxation,
        sfm_a=args.sfm_a,
        sfm_b=args.sfm_b,
        sfm_k=args.sfm_k,
        sfm_kappa=args.sfm_kappa,
        sfm_tau=args.sfm_tau,
        aacpm_kc=args.aacpm_kc,
        aacpm_beta=args.aacpm_beta,
        aacpm_k=args.aacpm_k,
        aacpm_kappa=args.aacpm_kappa,
        aacpm_kavo=args.aacpm_kavo,
        aacpm_tau=args.aacpm_tau,
        aacpm_delta=args.aacpm_delta,
        aacpm_omega=args.aacpm_omega,
        aacpm_vdes=args.aacpm_vdes,
        aacpm_vmax=args.aacpm_vmax,
        aacpm_alpha=args.aacpm_alpha,
        output_base=output_base,
        output_override=output_override,
    )


@dataclass(frozen=True)
class StateRecord:
    step: int
    time: float
    agent_id: int
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    radius: float


@dataclass(frozen=True)
class ContactRecord:
    ordinal: int
    time: float
    agent_id: int


def _iter_data_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            yield stripped


def load_states(path: Path) -> List[StateRecord]:
    records: List[StateRecord] = []
    for raw in _iter_data_lines(path):
        parts = raw.split()
        if not parts:
            continue
        if not parts[0].lstrip("-").isdigit():
            continue
        # Accept flexible state row lengths. Expected formats:
        # - Full format (10 columns): step time agent_id x y vx vy ax ay radius
        # - Short format (6 columns): step time agent_id x y radius
        # If a row has between 6 and 10 tokens, pad missing numeric fields with 0.0.
        if len(parts) < 6:
            raise ValueError(f"Unexpected states row format (too few columns): {raw}")
        if len(parts) > 10:
            # If there are extra tokens, keep the first 10 and ignore the rest.
            parts = parts[:10]
        if 6 <= len(parts) < 10:
            # pad missing numeric fields with zeros to reach 10 columns
            parts = parts + ["0.0"] * (10 - len(parts))

        # Now parts has exactly 10 tokens
        records.append(
            StateRecord(
                step=int(parts[0]),
                time=float(parts[1]),
                agent_id=int(parts[2]),
                x=float(parts[3]),
                y=float(parts[4]),
                vx=float(parts[5]),
                vy=float(parts[6]),
                ax=float(parts[7]),
                ay=float(parts[8]),
                radius=float(parts[9]),
            )
        )
    return records


def load_contacts(path: Path) -> List[ContactRecord]:
    records: List[ContactRecord] = []
    for raw in _iter_data_lines(path):
        parts = raw.split()
        if not parts:
            continue
        if not parts[0].lstrip("-").isdigit():
            continue
        if len(parts) != 3:
            raise ValueError(f"Unexpected contacts row format: {raw}")
        ordinal, time, agent = parts
        records.append(
            ContactRecord(
                ordinal=int(ordinal),
                time=float(time),
                agent_id=int(agent),
            )
        )
    return records


def compute_scanning_rate(contacts: Sequence[ContactRecord]) -> float:
    samples = [(c.time, c.ordinal) for c in contacts]
    if len(samples) < 2:
        return float("nan")
    sum_t = sum(t for t, _ in samples)
    sum_q = sum(q for _, q in samples)
    sum_tq = sum(t * q for t, q in samples)
    sum_t2 = sum(t * t for t, _ in samples)
    n = len(samples)
    denom = n * sum_t2 - sum_t * sum_t
    if math.isclose(denom, 0.0):
        return float("nan")
    return (n * sum_tq - sum_t * sum_q) / denom


def inter_contact_times(contacts: Sequence[ContactRecord]) -> List[float]:
    ordered = sorted(contacts, key=lambda c: c.time)
    return [
        ordered[i + 1].time - ordered[i].time
        for i in range(len(ordered) - 1)
    ]


def area_fraction(states: Sequence[StateRecord], domain: float) -> float:
    seen = set()
    total = 0.0
    for rec in states:
        if rec.agent_id in seen:
            continue
        seen.add(rec.agent_id)
        total += math.pi * rec.radius * rec.radius
    return total / (domain * domain)


def ensure_simulation(params: SimulationParams) -> Path:
    output_dir = params.output_dir()
    states_path = output_dir / "states.txt"
    contacts_path = output_dir / "contacts.txt"
    if states_path.exists() and contacts_path.exists():
        return output_dir
    run_simulation(params, output_dir)
    if not (states_path.exists() and contacts_path.exists()):
        raise RuntimeError(f"Simulation failed to produce expected files in {output_dir}")
    return output_dir


def ensure_simulations_parallel(
    base_params: SimulationParams, agent_counts: Sequence[int], seeds: Sequence[int]
) -> None:
    if not agent_counts or not seeds:
        return

    pending: dict[int, List[int]] = {}
    for count in agent_counts:
        missing: List[int] = []
        for seed in seeds:
            params = base_params.with_agents(count).with_seed(seed)
            output_dir = params.output_dir()
            states_path = output_dir / "states.txt"
            contacts_path = output_dir / "contacts.txt"
            if not (states_path.exists() and contacts_path.exists()):
                missing.append(seed)
        if missing:
            pending[count] = missing

    if not pending:
        return

    build_dir = PROJECT_ROOT / "build"
    if not build_dir.exists():
        subprocess.run(["make", "-C", str(PROJECT_ROOT), "build"], check=True)

    for count, seed_list in pending.items():
        params = base_params.with_agents(count)
        cli_args = params.to_java_cli_dict()
        # remove single-run specific keys
        cli_args.pop("seed", None)
        cli_args.pop("agents", None)
        if "output" in cli_args:
            # cannot run batch when explicit output override is requested
            continue
        cli_args["agentsList"] = str(count)
        cli_args["seeds"] = ",".join(str(seed) for seed in seed_list)
        cli_args["threads"] = str(min(5, len(seed_list)))

        command = ["java", "-cp", str(build_dir), "tp5.simulacion.Main"]
        command.extend(f"--{key}={value}" for key, value in cli_args.items())
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def run_simulation(params: SimulationParams, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    args = ["make", "-C", str(PROJECT_ROOT), "simulate"]
    args.extend(params.to_make_vars(output_dir))
    subprocess.run(args, check=True)


def group_states_by_step(states: Sequence[StateRecord]) -> List[Tuple[int, float, List[StateRecord]]]:
    steps: dict[int, Tuple[float, List[StateRecord]]] = {}
    for rec in states:
        if rec.step not in steps:
            steps[rec.step] = (rec.time, [])
        steps[rec.step][1].append(rec)
    ordered = []
    for step in sorted(steps):
        t, agents = steps[step]
        ordered.append((step, t, agents))
    return ordered
