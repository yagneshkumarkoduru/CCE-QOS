"""
CCE-QOS cloud DAG sweep orchestrator.

Runs the DAG scaling benchmarks (64/96/128 nodes, multi-seed) on a single spot
EC2 instance with no IAM instance profile. The instance receives time-limited
presigned S3 URLs via user data: one GET for the code bundle and PUTs for the
results archive and console log. The instance self-terminates when done.

Usage:
  python cloud/run_sweep.py package     # build cloud/.bundle/sweep_bundle.tgz
  python cloud/run_sweep.py launch      # upload bundle, presign URLs, launch spot
  python cloud/run_sweep.py status      # instance state + S3 artifacts
  python cloud/run_sweep.py collect     # download + extract results when ready
  python cloud/run_sweep.py terminate   # terminate the instance if still running
"""
import argparse
import base64
import json
import pathlib
import shutil
import sys
import tarfile
import time
import zipfile

import boto3
from botocore.exceptions import ClientError

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLOUD = ROOT / "cloud"
STATE_PATH = CLOUD / ".state.json"
BUNDLE_DIR = CLOUD / ".bundle"
BUNDLE_PATH = BUNDLE_DIR / "sweep_bundle.tgz"

REGION = "us-east-1"
BUCKET = "cce-qos-sweeps-969739653654"
AMI = "ami-025d99823a4caad37"  # Ubuntu Server 24.04 LTS amd64 (us-east-1)
SUBNETS = [
    "subnet-0879a02cc37518afb",  # default subnet, us-east-1a
    "subnet-0de1d5998132ef926",  # default subnet, us-east-1b
    "subnet-0eac15028cb6a014c",  # default subnet, us-east-1d
    "subnet-0b0043a151a024c6b",  # default subnet, us-east-1c
    "subnet-0de53e7b6d7caa40b",  # default subnet, us-east-1f
]
INSTANCE_TYPE = "c6i.xlarge"
PRESIGN_EXPIRY = 604800  # 7 days (max for SigV4)

SWEEP_ARGS = "--sizes 64,96,128 --seeds 42,43,44"

BUNDLE_FILES = [
    "benchmarks/__init__.py",
    "benchmarks/dag_scaling_benchmark.py",
    "benchmarks/dag_scaling_real_scheduler.py",
    "config.yaml",
    "core_types.py",
    "scheduling_engine.py",
    "cost_model.py",
    "memory_hierarchy.py",
    "bandwidth_estimator.py",
    "fusion_logic.py",
]

DRIVER_SH = """#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
echo "=== CCE-QOS cloud DAG sweep: $(date -u +%FT%TZ) ==="

run() {
  echo
  echo "--- $* ---"
  "$@"
  echo "[exit $?] $*"
}

run python3 benchmarks/dag_scaling_benchmark.py __SWEEP_ARGS__ --cpsat-timeout 300 --tag cloud
run python3 benchmarks/dag_scaling_real_scheduler.py __SWEEP_ARGS__ --sa-iterations 320 --lookahead-depth 2 --tag cloud
echo "=== driver finished: $(date -u +%FT%TZ) ==="
"""

USER_DATA_TEMPLATE = """#!/bin/bash
set -uo pipefail
exec > /var/log/sweep.log 2>&1
echo "CCE-QOS sweep __RUN_ID__ starting $(date -u +%FT%TZ)"
shutdown -h +100 "sweep watchdog" || true
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-pip zip
mkdir -p /opt/sweep
cd /opt/sweep
curl -fsSL -o bundle.tgz "__GET_URL__" && tar xzf bundle.tgz || echo "bundle download/extract failed"
pip3 install --break-system-packages --quiet --ignore-installed typing_extensions numpy ortools || echo "pip install failed"
bash run_sweep.sh || echo "sweep driver exited nonzero"
mkdir -p /opt/out
cp -r results /opt/out/ 2>/dev/null || echo "no results dir produced"
cd /opt/out && zip -qr /opt/results.zip results 2>/dev/null || zip -q /opt/results.zip . 2>/dev/null || echo "zip failed"
cp /var/log/sweep.log /opt/sweep_console.log
curl -fsSL -T /opt/results.zip "__PUT_RESULTS__" || echo "results upload failed"
curl -fsSL -T /opt/sweep_console.log "__PUT_LOG__" || echo "log upload failed"
echo "CCE-QOS sweep done $(date -u +%FT%TZ)"
shutdown -h now
"""


def load_state() -> dict:
    if not STATE_PATH.exists():
        raise SystemExit("No cloud/.state.json; run 'launch' first.")
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def s3_client():
    return boto3.client("s3", region_name=REGION)


def ec2_client():
    return boto3.client("ec2", region_name=REGION)


def cmd_package(_args) -> None:
    stage = BUNDLE_DIR / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for rel in BUNDLE_FILES:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"Missing bundle input: {rel}")
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (stage / "run_sweep.sh").write_text(
        DRIVER_SH.replace("__SWEEP_ARGS__", SWEEP_ARGS), encoding="utf-8", newline="\n"
    )
    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    with tarfile.open(BUNDLE_PATH, "w:gz") as tar:
        for item in sorted(stage.rglob("*")):
            tar.add(item, arcname=str(item.relative_to(stage)))
    shutil.rmtree(stage)
    size_kb = BUNDLE_PATH.stat().st_size / 1024
    print(f"Bundle: {BUNDLE_PATH} ({size_kb:.1f} KB)")
    for rel in BUNDLE_FILES + ["run_sweep.sh"]:
        print(f"  {rel}")


def ensure_bucket() -> None:
    s3 = s3_client()
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket", "NotFound"):
            s3.create_bucket(Bucket=BUCKET)
            print(f"Created bucket: {BUCKET}")
        else:
            raise


def cmd_launch(_args) -> None:
    if not BUNDLE_PATH.exists():
        cmd_package(_args)

    ec2 = ec2_client()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    prefix = f"sweeps/{run_id}"
    keys = {
        "bundle_key": f"{prefix}/sweep_bundle.tgz",
        "results_key": f"{prefix}/results.zip",
        "log_key": f"{prefix}/sweep_console.log",
    }

    ensure_bucket()
    s3 = s3_client()
    s3.upload_file(str(BUNDLE_PATH), BUCKET, keys["bundle_key"])
    get_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": keys["bundle_key"]},
        ExpiresIn=PRESIGN_EXPIRY,
    )
    put_results = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": keys["results_key"]},
        ExpiresIn=PRESIGN_EXPIRY,
    )
    put_log = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": keys["log_key"]},
        ExpiresIn=PRESIGN_EXPIRY,
    )

    user_data = (
        USER_DATA_TEMPLATE.replace("__RUN_ID__", run_id)
        .replace("__GET_URL__", get_url)
        .replace("__PUT_RESULTS__", put_results)
        .replace("__PUT_LOG__", put_log)
    )

    resp = None
    chosen_subnet = None
    last_error = None
    for subnet in SUBNETS:
        try:
            resp = ec2.run_instances(
                ImageId=AMI,
                InstanceType=INSTANCE_TYPE,
                MinCount=1,
                MaxCount=1,
                SubnetId=subnet,
                InstanceInitiatedShutdownBehavior="terminate",
                InstanceMarketOptions={
                    "MarketType": "spot",
                    "SpotOptions": {
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                },
                UserData=base64.b64encode(user_data.encode("utf-8")).decode("ascii"),
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": f"cce-qos-dag-sweep-{run_id}"},
                            {"Key": "Project", "Value": "CCE-QOS"},
                            {"Key": "Purpose", "Value": "dag-scaling-sweep"},
                        ],
                    }
                ],
            )
            chosen_subnet = subnet
            break
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "InsufficientInstanceCapacity":
                print(f"No spot capacity in {subnet}, trying next default subnet")
                last_error = exc
                continue
            raise
    if resp is None:
        raise SystemExit(f"No spot capacity in any default subnet: {last_error}")

    instance_id = resp["Instances"][0]["InstanceId"]
    state = {
        "run_id": run_id,
        "instance_id": instance_id,
        "subnet": chosen_subnet,
        "bucket": BUCKET,
        **keys,
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"Launched spot instance: {instance_id}")
    print(f"  type    : {INSTANCE_TYPE} (spot), subnet {chosen_subnet}")
    print(f"  run id  : {run_id}")
    print(f"  bundle  : s3://{BUCKET}/{keys['bundle_key']}")
    print("Estimated cost: ~1h at ~$0.08/h spot = under $0.10 (rates: us-east-1 spot history, i/nl/d ~ $0.071-0.082/h)")
    print("Check progress: python cloud/run_sweep.py status")


def cmd_status(_args) -> None:
    state = load_state()
    ec2 = ec2_client()
    resp = ec2.describe_instances(InstanceIds=[state["instance_id"]])
    inst = resp["Reservations"][0]["Instances"][0]
    print(f"Instance {state['instance_id']}: {inst['State']['Name']}")
    if inst.get("PublicIpAddress"):
        print(f"  public ip: {inst['PublicIpAddress']}")
    print(f"  launched : {inst.get('LaunchTime')}")

    s3 = s3_client()
    resp = s3.list_objects_v2(Bucket=state["bucket"], Prefix=f"sweeps/{state['run_id']}/")
    print(f"Artifacts under sweeps/{state['run_id']}/:")
    for obj in resp.get("Contents", []):
        print(f"  {obj['Key']}  ({obj['Size']} bytes, {obj['LastModified']})")
    if not resp.get("Contents"):
        print("  (none yet)")


def cmd_collect(_args) -> None:
    state = load_state()
    s3 = s3_client()
    try:
        s3.head_object(Bucket=state["bucket"], Key=state["results_key"])
    except ClientError:
        print("Results archive not uploaded yet; try again after the instance finishes.")
        print("Use status to check the instance state.")
        return

    out_dir = ROOT / "results" / "dag_scaling" / f"cloud_{state['run_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = BUNDLE_DIR / f"results_{state['run_id']}.zip"
    s3.download_file(state["bucket"], state["results_key"], str(zip_path))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"Extracted results to: {out_dir}")

    try:
        log_path = out_dir / "sweep_console.log"
        s3.download_file(state["bucket"], state["log_key"], str(log_path))
        print(f"Console log: {log_path}")
    except ClientError:
        print("Console log not uploaded (yet).")

    for json_file in sorted(out_dir.rglob("*.json")):
        print(f"  {json_file.relative_to(out_dir)}")


def cmd_terminate(_args) -> None:
    state = load_state()
    ec2 = ec2_client()
    resp = ec2.describe_instances(InstanceIds=[state["instance_id"]])
    name = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
    if name in ("terminated", "shutting-down"):
        print(f"Instance already {name}.")
        return
    ec2.terminate_instances(InstanceIds=[state["instance_id"]])
    print(f"Termination requested for {state['instance_id']}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CCE-QOS cloud DAG sweep orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("package", help="build the code bundle")
    sub.add_parser("launch", help="upload bundle and launch the spot instance")
    sub.add_parser("status", help="instance state and S3 artifacts")
    sub.add_parser("collect", help="download and extract results")
    sub.add_parser("terminate", help="terminate the instance if still running")
    args = parser.parse_args()

    handlers = {
        "package": cmd_package,
        "launch": cmd_launch,
        "status": cmd_status,
        "collect": cmd_collect,
        "terminate": cmd_terminate,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
