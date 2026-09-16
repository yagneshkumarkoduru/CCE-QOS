# CCE-QOS cloud DAG sweep

Runs the DAG scaling benchmarks (64/96/128 nodes, seeds 42/43/44) on a single
spot EC2 instance and returns the result JSON files plus a console log.

## Why cloud

Local runs cap out: the committed 30-second CP-SAT budget at 64 nodes only
reaches FEAS, and the real-scheduler lookahead benchmark scales to roughly two
minutes per 128-node configuration. A clean multi-seed 64/96/128 sweep with a
300-second CP-SAT budget takes about an hour, which is a better fit for one
dedicated instance than for the laptop.

## Pattern (no IAM instance profile)

The instance has no IAM role. All data movement uses time-limited presigned S3
URLs embedded in the EC2 user data:

- presigned GET for the code bundle,
- presigned PUT for `results.zip` (JSON outputs),
- presigned PUT for the console log.

The instance self-terminates when the driver finishes (`shutdown -h now`, spot
instances terminate on shutdown). A 100-minute watchdog shutdown bounds the
run even if something hangs.

## Usage

```powershell
# from the repo root, with the project venv (needs boto3 + AWS credentials)
.\.venv\Scripts\python.exe cloud\run_sweep.py package
.\.venv\Scripts\python.exe cloud\run_sweep.py launch
.\.venv\Scripts\python.exe cloud\run_sweep.py status
.\.venv\Scripts\python.exe cloud\run_sweep.py collect
.\.venv\Scripts\python.exe cloud\run_sweep.py terminate   # only if needed
```

Collected results land in `results/dag_scaling/cloud_<run_id>/`.

## Fixed parameters

| Item | Value |
| :--- | :--- |
| Region | us-east-1 |
| Instance | c6i.xlarge (4 vCPU, 8 GB), spot, one-time |
| AMI | Ubuntu Server 24.04 LTS (ami-025d99823a4caad37) |
| Subnet | default subnet us-east-1f |
| Bucket | cce-qos-sweeps-969739653654 (created on first launch) |
| Sweep args | `--sizes 64,96,128 --seeds 42,43,44`, CP-SAT timeout 300 s |
| Real scheduler | SA 320 iterations (config.yaml), lookahead depth 2 |

## Cost

Spot for c6i.xlarge in us-east-1 has recently been $0.071-0.082/hour
(`aws ec2 describe-spot-price-history`). A full sweep is about an hour, so the
expected cost is under $0.10, plus pennies for S3 storage and requests. The
100-minute watchdog caps the spend near $0.15 even in the worst case.

## Safety notes

- Presigned URLs expire after 7 days and are scoped to a single run prefix.
- `cloud/.state.json` and `cloud/.bundle/` are gitignored (instance ids and
  temporary archives should not be committed).
- The instance can reach the internet only through the default subnet's
  auto-assigned public IP; no inbound ports are opened.
