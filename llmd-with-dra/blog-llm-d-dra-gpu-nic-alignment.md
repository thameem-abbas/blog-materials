# Topology-Aware GPU-NIC Allocation for llm-d: A Proof of Concept with Kubernetes DRA

When running large language models with prefill/decode (P/D) disaggregation, the KV cache must be transferred between prefill and decode workers over the network. On multi-GPU nodes with multiple NICs, which GPU talks through which NIC matters enormously. A misaligned GPU-NIC pairing can degrade NIXL KV transfer throughput from **355 Gbps down to 26 Gbps** -- a 14x performance collapse.

We've been exploring how Kubernetes Dynamic Resource Allocation (DRA) can solve this problem at the scheduler level -- making correct GPU-NIC alignment the default rather than an ops burden. This post covers what we found, how we validated it, and what's coming next in [llm-d](https://github.com/llm-d/llm-d).

---

## Why GPU-NIC Alignment Matters for P/D Disaggregation

In P/D disaggregation, the prefill phase (processing the input prompt) and decode phase (generating tokens) run on separate GPU pods. The optimal configuration often uses a subset of a node's GPUs per pod. For example, on an 8-GPU node serving a 32B-parameter model:

- **4 prefill pods** with 1 GPU each (TP=1)
- **1 decode pod** with 4 GPUs (TP=4)

After prefill computes the KV cache, [NIXL](https://github.com/ai-dynamo/nixl) transfers it to a decode worker over RDMA. This transfer uses GPUDirect RDMA (GDR) -- the NIC performs DMA directly to and from GPU memory, bypassing the CPU entirely. But GDR performance depends critically on the PCIe path between the GPU and NIC.

On HGX/DGX-class servers, each GPU is physically paired with a specific NIC through a shared PCIe switch:

```
  Node (8x GPU, 8x NIC)
  =====================

  PCIe Root 0xd2          PCIe Root 0xa0          ...
  ┌─────────────┐         ┌─────────────┐
  │ PCIe Switch │         │ PCIe Switch │
  │  GPU-2      │         │  GPU-7      │
  │  NIC-2      │         │  NIC-7      │
  │  ◄── PIX ──►│         │  ◄── PIX ──►│
  └─────────────┘         └─────────────┘

  PIX = devices share same PCIe switch (fastest path)
```

When GPU and NIC share a PCIe switch (PIX connection), DMA operations traverse a single PCIe bridge. When they don't, the DMA path crosses additional PCIe switches or even CPU sockets, throttling bandwidth:

| Alignment | NIXL KV Transfer | Degradation |
|-----------|-----------------|-------------|
| Same PCIe switch (PIX) | 355 Gbps | baseline |
| Cross-IIO, same NUMA | 26 Gbps | **14x** |
| Cross-socket | 16.5 Gbps | **22x** |

*(Source: [Joshi & Smith, 2025](https://github.com/tlrmchlsmth/j-llm-d/blob/raj/ttft/glm-disagg-pcie/nixl-iio-investigation/nixl-pcie-investigation-blog.md))*

For GDR to work at all, the `nvidia_peermem` kernel module must be loaded and compatible MOFED/NVIDIA driver versions must be installed. But even with GDR enabled, misalignment silently degrades throughput by over an order of magnitude.

---

## Why Traditional Approaches Fall Short

### Device Plugins

The Kubernetes device plugin model treats GPUs and NICs as independent integer counters. `nvidia.com/gpu: 4` gets you four GPUs; `rdma/ib: 4` gets you four NICs. But which four of each? The scheduler doesn't know and doesn't care -- it has no visibility into PCIe topology. The GPU allocator and network operator make independent decisions with no cross-device coordination. The result is correct by count but potentially catastrophic by topology.

### Multus with Manual GPU Selection

Multus CNI can attach multiple NICs to a pod via NetworkAttachmentDefinitions. Combined with careful GPU selection (using `nvidia-smi topo -m` to discover PIX affinity at runtime), this approach **does achieve equivalent bandwidth**. Our testing showed DRA and Multus within ~2% of each other (~377-385 Gb/sec at 4MB message size with GPUDirect RDMA).

The limitation isn't performance -- it's operational correctness. Multus requires manual NAD management, and GPU-NIC alignment must be discovered and enforced at the application level rather than guaranteed by the scheduler. On a cluster with multiple pods sharing a node's GPUs, one misconfigured NAD or incorrect GPU selection silently drops you from 355 Gbps to 26 Gbps with no error or warning.

### SR-IOV VFs

A natural instinct is to slice each physical NIC into SR-IOV Virtual Functions -- one VF per pod -- and allocate them through the SR-IOV device plugin. When VFs are used to share a physical NIC across multiple pods, each VF exposes only a fraction of the NIC's hardware resources (QP-context cache, total Queue Pairs) -- roughly 10% or less of the NIC's capability, leaving the majority of the hardware unutilized. For GDR-intensive KV cache transfers, this is a significant penalty. If the full physical device is passed through exclusively to one pod, this penalty doesn't apply -- but then you lose the ability to share NICs across pods, which is the whole reason for SR-IOV in a multi-pod-per-node P/D setup.

There's also a nesting problem: some cloud environments (like IBM Cloud) already expose SR-IOV VFs as the host NICs. You can't create VFs on top of VFs.

### Shared Device Plugin + Macvlan/IPvlan

An alternative approach gives all pods access to all RDMA devices on the host (via a shared device plugin) and creates per-pod network interfaces using Macvlan or IPvlan. The idea is that networking libraries (NIXL, UCX, NCCL) will discover PCIe topology at runtime and faithfully use only the GPU-aligned NIC.

This works in some environments but has sharp edges. Macvlan creates additional MAC addresses per NIC, which cloud network fabrics often block. IPvlan avoids the MAC problem but requires the fabric to allow multiple IPs on the same MAC (IP spoofing). On RoCEv2, the GID table accumulates entries from all pods on the node, making GID indices dynamic and harder to reason about.

There's also no real isolation. Every pod on the node has access to every RDMA device -- the shared device plugin exposes all `/dev/infiniband/` character devices to all pods. Pods are trusted to only use the NICs aligned to their GPUs, but nothing enforces this. A misbehaving or misconfigured pod can saturate a NIC that another pod is relying on, with no scheduler visibility into the conflict.

### Static Pinning

Hardcoding GPU-to-NIC mappings in pod specs works until it doesn't. It breaks on heterogeneous clusters, doesn't survive node replacements, and doesn't scale to multi-tenant environments where multiple teams share GPU nodes.

### The Common Thread

Every pre-DRA approach shares the same fundamental gap: **no Kubernetes mechanism enforces cross-device topology constraints at allocation time.** GPUs and NICs are separate resource domains with no shared scheduling context. Solutions either sacrifice NIC performance (SR-IOV VFs), rely on application-level topology discovery (shared device plugin), or require manual coordination (static pinning, Multus). DRA closes this gap.

---

## DRA: Cross-Driver Topology Constraints

Dynamic Resource Allocation (DRA) replaces opaque integer counters with a structured device API. Instead of "give me 4 of resource X," workloads can say "give me 4 GPUs and 4 NICs, but only if each pair shares the same PCIe root." DRA [graduated to GA in Kubernetes v1.34](https://kubernetes.io/blog/2025/09/01/kubernetes-v1-34-dra-updates/) with the stable `resource.k8s.io/v1` API. The `pcieRoot` cross-driver constraint specifically requires Kubernetes 1.34.2+ and NVIDIA GPU Operator v25.10+ (with DRA driver v25.12.0+).

### How It Works

Hardware drivers run as DaemonSets on each node. They discover devices and publish **ResourceSlice** objects describing what's available and each device's attributes. Workloads create **ResourceClaim** objects specifying what they need. The scheduler matches claims to devices, enforcing constraints across resource types.

Two DRA drivers are needed for GPU-NIC alignment:

- **NVIDIA GPU DRA driver** (`gpu.nvidia.com`) -- ships with NVIDIA GPU Operator v25.10+. Publishes each GPU with attributes including `resource.kubernetes.io/pcieRoot`.
- **[DRA-Net](https://github.com/kubernetes-sigs/dranet)** (`dra.net`) -- a Kubernetes-SIGs project for network devices. Discovers RDMA-capable NICs and publishes them with `pcieRoot`, `numaNode`, and `rdma` attributes. Uses NRI (Node Resource Interface) for network configuration at pod creation time. Works with both physical functions and SR-IOV VFs.

Both drivers discover `pcieRoot` by traversing the sysfs PCIe hierarchy on the host -- not by parsing bus address prefixes. DRA-Net reads `/sys/bus/pci/devices/*/firmware_node` and walks the PCIe topology tree. This ensures accurate root complex identification regardless of bus numbering. (This matters: a NIC at PCI bus `0000:d5:00.0` has pcieRoot `pci0000:d2`, not `pci0000:d5`.)

### The Key Mechanism

A ResourceClaim can specify a cross-driver constraint:

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: gpu-nic-aligned
spec:
  devices:
    requests:
    - name: gpu
      exactly:
        deviceClassName: gpu.nvidia.com
        count: 1
    - name: nic
      exactly:
        deviceClassName: dranet
        count: 1
        selectors:
        - cel:
            expression: device.attributes["dra.net"].rdma == true
    constraints:
    - requests: ["gpu", "nic"]
      matchAttribute: resource.kubernetes.io/pcieRoot
```

Three lines of YAML -- `constraints`, `requests`, `matchAttribute` -- and the scheduler will only co-allocate devices that share the same PCIe root. No application-level topology discovery, no manual pinning, no silent misalignment. Since both drivers derive this attribute from the same sysfs PCIe hierarchy, the constraint is grounded in physical hardware topology.

### Scaling to Multiple Pairs

The single-pair example above uses one `matchAttribute` constraint across all devices in the claim. This works for one GPU-NIC pair, but requesting `count: 4` for both GPUs and NICs with a single constraint fails -- `matchAttribute` requires *all* devices in the constraint to share the same pcieRoot, and 4 GPUs span 4 different PCIe roots.

The solution is indexed named requests with per-pair constraints:

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: gpu-nic-4pair
spec:
  devices:
    requests:
    - name: gpu0
      exactly:
        deviceClassName: gpu.nvidia.com
        count: 1
    - name: nic0
      exactly:
        deviceClassName: dranet
        count: 1
        selectors:
        - cel:
            expression: device.attributes["dra.net"].rdma == true
    - name: gpu1
      exactly:
        deviceClassName: gpu.nvidia.com
        count: 1
    - name: nic1
      exactly:
        deviceClassName: dranet
        count: 1
        selectors:
        - cel:
            expression: device.attributes["dra.net"].rdma == true
    # ... gpu2/nic2, gpu3/nic3
    constraints:
    - requests: ["gpu0", "nic0"]
      matchAttribute: resource.kubernetes.io/pcieRoot
    - requests: ["gpu1", "nic1"]
      matchAttribute: resource.kubernetes.io/pcieRoot
    # ... per-pair constraints for each indexed pair
```

Each constraint scopes to one GPU-NIC pair, so the scheduler matches each pair independently to its own pcieRoot. We validated this on our cluster -- 4 pairs allocated correctly, each on its own PCIe root, all within a single ResourceClaim.

The DeviceClass definitions are minimal:

```yaml
apiVersion: resource.k8s.io/v1
kind: DeviceClass
metadata:
  name: gpu.nvidia.com
spec:
  selectors:
  - cel:
      expression: device.driver == "nvidia.com/gpu"
---
apiVersion: resource.k8s.io/v1
kind: DeviceClass
metadata:
  name: dranet
spec:
  selectors:
  - cel:
      expression: device.driver == "dra.net" && device.attributes["dra.net"].rdma == true
```

---

## Simplifying DRA for Complex Network Topologies

On a bare metal cluster with a flat RDMA fabric, the indexed ResourceClaim pattern above is all you need. Wrap it in a ResourceClaimTemplate, reference it in the pod spec, and the scheduler handles alignment. No webhook, no custom controllers.

But the verbosity is real. A single GPU-NIC pair is ~15 lines of YAML. Four pairs means 8 named requests, 4 constraints, and 8 `resources.claims` entries in the pod spec -- easily 60+ lines for what is conceptually a single number. You also need to maintain separate templates for each pair count your workloads use (1 pair for TP=1 prefill, 4 pairs for TP=4 decode, 8 pairs for full-node). Compare that to `nvidia.com/gpu: 4`, which is one line. The gap between DRA's power and its usability is real -- the mechanism is correct, but the interface is not yet GPU count-like.

This gap widens further in environments with complex networking. On our IBM Cloud GPU cluster, each NIC sits on a separate rail subnet (10.0.0.0/16 through 10.7.0.0/16) with its own gateway -- a rail-isolated network architecture where each GPU-NIC pair communicates exclusively within its own subnet:

```
  IBM Cloud Rail-Isolated Network Architecture
  =============================================

  Rail 0 (10.0.0.0/16)          Rail 1 (10.1.0.0/16)          ...  Rail 7 (10.7.0.0/16)
  ┌──────────────────┐          ┌──────────────────┐               ┌──────────────────┐
  │   GW 10.0.0.1    │          │   GW 10.1.0.1    │               │   GW 10.7.0.1    │
  └────────┬─────────┘          └────────┬─────────┘               └────────┬─────────┘
           │                             │                                  │
     ┌─────┴─────┐                 ┌─────┴─────┐                     ┌─────┴─────┐
     │  Switch   │                 │  Switch   │                     │  Switch   │
     └─────┬─────┘                 └─────┬─────┘                     └─────┬─────┘
     ┌─────┴─────────────┐         ┌─────┴─────────────┐             ┌─────┴─────────────┐
     │  Node 1: mlx5_7   │         │  Node 1: mlx5_6   │             │  Node 1: mlx5_0   │
     │  Node 2: mlx5_7   │         │  Node 2: mlx5_6   │             │  Node 2: mlx5_0   │
     │  Node 3: mlx5_7   │         │  Node 3: mlx5_6   │             │  Node 3: mlx5_0   │
     │  Node 4: mlx5_7   │         │  Node 4: mlx5_6   │             │  Node 4: mlx5_0   │
     └───────────────────┘         └───────────────────┘             └───────────────────┘

  Each rail: one /16 subnet, one gateway, one NIC per node.
  Cross-rail traffic routes through the supernet (10.0.0.0/13).
```

*[TODO: Replace with proper diagram]*

This rail-isolated architecture made sense for distributed training with NCCL, where collective operations like all-reduce naturally align to rails -- each GPU communicates with its peer GPU on other nodes through the same-rail NIC, and traffic stays within one subnet. Cross-rail communication wasn't a priority because the communication pattern matched the network topology.

P/D disaggregation breaks this assumption. A prefill pod on node A may need to transfer KV cache to a decode pod on node B, and those pods' GPUs may be on different rails. The KV transfer must cross subnets. You can't avoid this by routing within the same rail and then using NVLink to reach the destination GPU -- NVLink is intra-node only and doesn't span across nodes. The traffic must go over the network, through the cross-rail supernet, adding routing complexity that a flat fabric wouldn't have.

The same cross-rail challenge applies to wide endpoint (wideEP) deployments, where every NIC in the cluster needs to reach every other NIC -- an all-to-all communication pattern that the rail-isolated architecture was never designed for.

Each NIC needs per-rail policy routing, MTU configuration, and source-based routing tables. In this environment, each ResourceClaimTemplate must include rail-specific CEL selectors and opaque driver configuration -- a single pod requesting 4 GPU-NIC pairs would need 4 ResourceClaimTemplates with rail-specific network config, 8 claim references, and routing rules for each subnet. That's over 100 lines of YAML that is both error-prone and not discoverable by users who aren't DRA experts.

We built a [mutating admission webhook](https://github.com/openshift-psap/dra-rail-admission-webhook) to collapse all of this into a GPU count-like interface. The idea: a single synthetic resource that the system decomposes into the underlying primitives provided by the DRA drivers -- GPUs from `gpu.nvidia.com`, NICs from `dra.net`, with topology constraints and network configuration injected automatically. Instead of writing per-rail ResourceClaimTemplates, workloads request:

```yaml
resources:
  requests:
    dra.llm-d.io/gpu-nic-pair: "4"
  limits:
    dra.llm-d.io/gpu-nic-pair: "4"
```

The webhook intercepts pod creation and:

1. **Validates** the request against NUMA and node topology limits
2. **Allocates** a node and rail indices, using a priority queue (3-second debounce, larger requests first -- so prefill pods don't starve decode pods of rail capacity)
3. **Generates** N ResourceClaimTemplates, each with GPU+NIC device requests, `pcieRoot` constraints, and rail-specific opaque config (interface naming, MTU 9000, per-rail routing tables)
4. **Mutates** the pod spec: injects resource claims, adds node affinity, removes the synthetic resource
5. A **reconciler** component runs alongside the webhook to detect and clean up orphaned ResourceClaimTemplates -- since the webhook creates templates outside the normal Kubernetes ownership chain, deleted or failed pods can leave behind ResourceClaimTemplates that hold device allocations. The reconciler scans for templates with no referencing pod and reaps them after a configurable grace period (default: 10 minutes)

### Before and After

**What the user writes:**

```yaml
spec:
  containers:
  - name: vllm
    resources:
      requests:
        dra.llm-d.io/gpu-nic-pair: "2"
      limits:
        dra.llm-d.io/gpu-nic-pair: "2"
```

**What the webhook produces** (simplified):

```yaml
spec:
  resourceClaims:
  - name: gpu-nic-pair-0
    resourceClaimTemplateName: gpu-nic-pair-0-rail2-<hash>
  - name: gpu-nic-pair-1
    resourceClaimTemplateName: gpu-nic-pair-1-rail5-<hash>
  containers:
  - name: vllm
    resources:
      claims:
      - name: gpu-nic-pair-0
        request: gpu
      - name: gpu-nic-pair-0
        request: nic
      - name: gpu-nic-pair-1
        request: gpu
      - name: gpu-nic-pair-1
        request: nic
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchFields:
          - key: metadata.name
            operator: In
            values: ["worker-node-1"]
```

Each generated ResourceClaimTemplate includes the `pcieRoot` matching constraint and rail-specific NIC configuration (CEL selector for rail subnet, opaque config with routing rules and MTU).

**Important:** This webhook is specific to rail-aligned network environments. On a bare metal cluster with a flat RDMA fabric, you don't need it -- add ResourceClaimTemplates directly to your deployment values.

---

## What We Changed in the llm-d Deployment

The llm-d P/D disaggregation deployment uses a Helmfile orchestrating three Helm releases: infrastructure (gateway), routing (InferencePool with P/D endpoint picker), and model service (vLLM workers with NIXL). Enabling DRA touched only the model service layer -- two specific changes.

### Resource Suppression in the Helmfile

The llm-d-modelservice Helm chart auto-injects `nvidia.com/gpu` resources based on `parallelism.tensor` and `rdma/ib` from default values. With DRA, GPU and NIC allocation is handled through `dra.llm-d.io/gpu-nic-pair`, so the device plugin resources must be suppressed to prevent double-allocation conflicts:

```yaml
# helmfile.yaml.gotmpl -- null out device plugin resources for DRA environments
{{- if eq .Environment.Name "dra-enabled-env" }}
  - name: 'decode.containers[0].resources.limits.nvidia\.com/gpu'
    value: "null"
  - name: 'decode.containers[0].resources.requests.nvidia\.com/gpu'
    value: "null"
  - name: 'prefill.containers[0].resources.limits.nvidia\.com/gpu'
    value: "null"
  - name: 'prefill.containers[0].resources.requests.nvidia\.com/gpu'
    value: "null"
  - name: 'decode.containers[0].resources.limits.rdma/ib'
    value: "null"
  - name: 'prefill.containers[0].resources.limits.rdma/ib'
    value: "null"
{{- end }}
```

### Model Service Values

The DRA-enabled values file replaces device plugin resources with the synthetic DRA pair and adds operational constraints required by DRA:

```yaml
decode:
  parallelism:
    tensor: 4
  replicas: 4
  strategy:
    type: Recreate    # DRA claims are exclusive -- rolling updates deadlock
  containers:
  - name: vllm
    resources:
      limits:
        dra.llm-d.io/gpu-nic-pair: "4"
      requests:
        dra.llm-d.io/gpu-nic-pair: "4"

prefill:
  parallelism:
    tensor: 1
  replicas: 16
  strategy:
    type: Recreate
  containers:
  - name: vllm
    resources:
      limits:
        dra.llm-d.io/gpu-nic-pair: "1"
      requests:
        dra.llm-d.io/gpu-nic-pair: "1"
```

A `Recreate` strategy is currently required because the webhook's gang scheduling and packing mechanism doesn't support rolling updates -- it allocates node and rail assignments at pod creation time, and a rolling update would conflict with this allocation model. Rolling updates should be feasible once the broader DRA ecosystem matures and allocation can be coordinated without the webhook's upfront scheduling.

---

## Validation

We validated DRA-based GPU-NIC alignment on a 4-node cluster with 8x NVIDIA H100 80GB GPUs and 8x Mellanox ConnectX-7 RDMA NICs per node.

### 100% PCIe Root Alignment

All 8 GPU-NIC pairs per node were correctly matched by PCIe root:

**NUMA 0:**

| GPU | GPU PCIe | NIC | NIC PCIe | PCIe Root | Status |
|-----|----------|-----|----------|-----------|--------|
| gpu-7 | 0000:a4:00.0 | enp163s0 | 0000:a3:00.0 | pci0000:a0 | PIX |
| gpu-6 | 0000:ae:00.0 | enp173s0 | 0000:ad:00.0 | pci0000:aa | PIX |
| gpu-5 | 0000:b8:00.0 | enp183s0 | 0000:b7:00.0 | pci0000:b4 | PIX |
| gpu-4 | 0000:c2:00.0 | enp193s0 | 0000:c1:00.0 | pci0000:be | PIX |

**NUMA 1:**

| GPU | GPU PCIe | NIC | NIC PCIe | PCIe Root | Status |
|-----|----------|-----|----------|-----------|--------|
| gpu-3 | 0000:cc:00.0 | enp203s0 | 0000:cb:00.0 | pci0000:c8 | PIX |
| gpu-2 | 0000:d6:00.0 | enp213s0 | 0000:d5:00.0 | pci0000:d2 | PIX |
| gpu-1 | 0000:e0:00.0 | enp223s0 | 0000:df:00.0 | pci0000:dc | PIX |
| gpu-0 | 0000:ea:00.0 | enp233s0 | 0000:e9:00.0 | pci0000:e6 | PIX |

NUMA alignment comes as a side-effect of PCIe root matching -- every pair on the same PCIe root is naturally on the same NUMA node.

### Topology Verification

`nvidia-smi topo -m` in each pod confirmed PIX connectivity:

```
GPU0  NIC2
GPU0   X   PIX

PIX = Connection traversing at most a single PCIe bridge
```

RDMA devices were accessible in containers (`/dev/infiniband/uverbs*`), and GPUDirect RDMA tests confirmed the aligned path was active.

### Performance Parity with Multus

We ran `ib_write_bw` and `ib_write_lat` benchmarks comparing DRA and Multus allocations with GPUDirect RDMA on the same hardware:

| Method | Same-Rail BW (Gb/sec) | Cross-Rail BW (Gb/sec) |
|--------|-----------------------|------------------------|
| DRA | 377.66 | 377.09 |
| Multus | 385.28 | 380.50 |

Both methods deliver equivalent bandwidth -- within ~2%, well within noise on a shared network. DRA adds no performance overhead. Its advantage is operational: correct-by-construction allocation instead of correct-by-accident.

### Production Deployment

We deployed llm-d with 4 decode pods (TP=4, 4 GPU-NIC pairs each) and 16 prefill pods (TP=1, 1 GPU-NIC pair each) across 4 nodes. Every pod received correctly aligned GPU-NIC pairs:

| Pod | Role | Node | GPU PCIe | NIC | RoCE IP |
|-----|------|------|----------|-----|---------|
| decode-477jb | decode | worker-3-dls46 | A4, AE, B8, C2 | mlx5_7,6,5,4 | 10.0-3.x |
| prefill-2nr9z | prefill | worker-3-dls46 | CC | mlx5_3 | 10.4.0.6 |
| prefill-4nbhq | prefill | worker-3-dls46 | D6 | mlx5_2 | 10.5.0.6 |
| ... | ... | ... | ... | ... | ... |

Each prefill pod's single GPU was paired with the PCIe-adjacent NIC. Each decode pod's 4 GPUs were paired with 4 NICs, all from the same NUMA zone.

---

## Status, Limitations, and What's Next

This work is a proof of concept. We validated that DRA can guarantee topology-correct GPU-NIC allocation for llm-d's P/D disaggregated inference -- and that it does so without sacrificing performance. Here's where things stand.

### What Works Today

- **DRA primitives**: `pcieRoot` cross-driver constraint delivers 100% GPU-NIC alignment on real hardware
- **DRA-Net**: upstream at [kubernetes-sigs/dranet](https://github.com/kubernetes-sigs/dranet), actively maintained, publishes correct PCIe topology attributes
- **Performance**: DRA matches Multus bandwidth (~377-385 Gb/sec with GPUDirect RDMA)
- **NVSHMEM/NIXL**: KV cache transfers work correctly over DRA-allocated NICs

### What's Still Closing

Some gaps are in the platform -- Kubernetes, DRA drivers, and the broader ecosystem -- which are actively being addressed upstream. Others are in llm-d's integration, which we're building toward.

**On the platform side**, DRA provides the right primitives but the usability gap remains. Requesting 4 GPU-NIC pairs requires indexed named requests with per-pair constraints -- correct and validated, but verbose (~40 lines of YAML for what is conceptually a single number). You need separate ResourceClaimTemplates for each pair count your workloads use (1-pair for TP=1 prefill, 4-pair for TP=4 decode). There's no GPU count-like shorthand yet. DRA resource claims are also exclusive, meaning rolling updates deadlock when the new pod waits for claims held by the old pod -- workloads currently require `strategy: Recreate`. The DRA driver ecosystem is maturing: both the NVIDIA GPU DRA driver and DRA-Net are actively developed, but production support matrices are still solidifying, and DRA-Net is not yet productized by any Kubernetes distribution vendor. Consumable Capacity (Beta in Kubernetes 1.36) will improve multi-device scheduling, and major cloud providers are actively adopting DRA -- [AKS](https://blog.aks.azure.com/2026/04/01/dranet-rdma-optimization-for-ai-on-aks) and [OKE](https://substack.com/home/post/p-192535356) have both published DRA-Net guidance.

**On the llm-d side**, the ModelService Helm chart can already handle basic DRA ResourceClaimTemplates -- indexed GPU-NIC pairs with `pcieRoot` constraints and straightforward network config. For clusters with flat RDMA fabrics, a template in the values file works today:

```yaml
# 4-pair DRA ResourceClaimTemplate for decode (TP=4) -- no webhook needed
resourceClaimTemplates:
- metadata:
    name: gpu-nic-4pair
  spec:
    devices:
      requests:
      - name: gpu0
        exactly: { deviceClassName: gpu.nvidia.com, count: 1 }
      - name: nic0
        exactly:
          deviceClassName: dranet
          count: 1
          selectors:
          - cel:
              expression: device.attributes["dra.net"].rdma == true
      - name: gpu1
        exactly: { deviceClassName: gpu.nvidia.com, count: 1 }
      - name: nic1
        exactly:
          deviceClassName: dranet
          count: 1
          selectors:
          - cel:
              expression: device.attributes["dra.net"].rdma == true
      # ... gpu2/nic2, gpu3/nic3 follow the same pattern
      constraints:
      - requests: ["gpu0", "nic0"]
        matchAttribute: resource.kubernetes.io/pcieRoot
      - requests: ["gpu1", "nic1"]
        matchAttribute: resource.kubernetes.io/pcieRoot
      - requests: ["gpu2", "nic2"]
        matchAttribute: resource.kubernetes.io/pcieRoot
      - requests: ["gpu3", "nic3"]
        matchAttribute: resource.kubernetes.io/pcieRoot
```

What the chart can't (and shouldn't have to) handle is per-rail routing configuration, rail-specific CEL selectors, and subnet-aware opaque driver config. That complexity belongs in the webhook for environments that need it. The resource suppression mechanism (nulling out `nvidia.com/gpu` and `rdma/ib` in the helmfile) is a stopgap until the chart natively supports a DRA mode that doesn't auto-inject device plugin resources.

The [admission webhook](https://github.com/openshift-psap/dra-rail-admission-webhook) is not upstream -- it was built for our IBM Cloud rail-aligned networking. Clusters with simpler networking don't need it. The goal is for llm-d to natively generate DRA ResourceClaimTemplates when DRA is enabled, making the webhook unnecessary for most environments and reducing it to a bridge for complex network topologies.

---

## Conclusion

GPU-NIC topology alignment is the difference between 355 Gbps and 26 Gbps for KV cache transfers in P/D disaggregated inference. Today, getting this right requires either manual topology management or accepting the risk of silent misalignment. DRA makes correct allocation the default -- a scheduler-level guarantee, not an ops burden.

For clusters with simple RDMA fabrics, a ResourceClaim with `matchAttribute: resource.kubernetes.io/pcieRoot` is enough. For complex multi-rail environments, a webhook can collapse the configuration to a single GPU count-like resource request.

Full DRA support in llm-d is on the roadmap. What we've shown here is the foundation: the mechanism works, the performance is validated, and the path from proof of concept to production integration is clear.

---

## References

- [llm-d project](https://github.com/llm-d/llm-d)
- [NIXL (NVIDIA Cross-Layer)](https://github.com/ai-dynamo/nixl)
- [DRA-Net (kubernetes-sigs/dranet)](https://github.com/kubernetes-sigs/dranet)
- [DRA Admission Webhook](https://github.com/openshift-psap/dra-rail-admission-webhook)
- [Kubernetes v1.34: DRA Graduates to GA](https://kubernetes.io/blog/2025/09/01/kubernetes-v1-34-dra-updates/)
- [NIXL PCIe Investigation (Joshi & Smith)](https://github.com/tlrmchlsmth/j-llm-d/blob/raj/ttft/glm-disagg-pcie/nixl-iio-investigation/nixl-pcie-investigation-blog.md)
- [AKS: DRA-Net RDMA Optimization](https://blog.aks.azure.com/2026/04/01/dranet-rdma-optimization-for-ai-on-aks)
- [DRA-Net on OKE](https://substack.com/home/post/p-192535356)
- [DRA Consumable Capacity KEP](https://github.com/kubernetes/enhancements/tree/master/keps/sig-scheduling/5075-dra-consumable-capacity)
