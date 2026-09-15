# Manual Buildkite CI

The pipeline runs Humming's tests on one B200 in vLLM's `b200-k8s` queue.
It uses PyTorch 2.13.0 with CUDA 13.0 and the CUDA development toolkit.

## One-time Buildkite setup

Create a pipeline for `https://github.com/vllm-project/humming.git` in the
vLLM Buildkite organization. Select the cluster containing `b200-k8s` and
`cpu_queue_premerge`. A cluster maintainer may need to grant access.
Disable automatic push and pull-request build triggers.

In the visual step editor, configure a single step:

- **Commands to run:** `buildkite-agent pipeline upload .buildkite/pipeline.yml`
- **Label:** `Upload Humming pipeline`
- **Agent Targeting Rules:** `queue=cpu_queue_premerge`

If using Buildkite's YAML editor instead, enter:

```yaml
steps:
  - label: "Upload Humming pipeline"
    command: buildkite-agent pipeline upload .buildkite/pipeline.yml
    agents:
      queue: cpu_queue_premerge
```

The CPU step checks out the selected Humming commit and uploads this repository's
pipeline. The uploaded GPU step checks out the same commit and runs the tests.
The branch/commit selected for the first build must contain these files.

## Run a build

Select **New Build**, choose the branch/commit, and create the build.
The default tuning source is `heuristic`.

To run the longer sampled configuration tests, add this build environment variable:

```text
HUMMING_TEST_TUNING_SOURCE=sampled
```

`HUMMING_TEST_TUNING_SOURCE=batch_invariant` is also supported.
Only one job in the `humming/b200` concurrency group runs at a time; each has
a 180-minute timeout.

## Optional benchmark capture

To run a dense W4A16 benchmark after successful tests, add:

```text
HUMMING_RUN_BENCHMARK=1
```

Build artifacts include JUnit results, numerical failure details when present,
GPU/toolkit information, installed package versions, and optional benchmark JSON.
The benchmark records timings; it does not enforce a regression threshold.
Compare results using the same GPU model, software versions, and benchmark shapes.

## Configuration references

- [Buildkite pipeline upload](https://buildkite.com/docs/pipelines/configure/defining-steps)
- [vLLM Kubernetes job templates](https://github.com/vllm-project/ci-infra/blob/main/buildkite/pipeline_generator/plugin/k8s_plugin.py)
- [vLLM CPU queues](https://github.com/vllm-project/ci-infra/blob/main/terraform/aws/cloudformation_stack.tf)
