// A batch of lives settled on the GPU (WebGPU, WGSL): L brains of one anatomy, each with its own
// synapses, one row each. A sweep is three passes: each neuron's basal and apical input (its
// incoming synapses, the brain's CSR by postsynaptic neuron, read from the activations the sweep
// begins with), each pool's gain (the apical compartment's normalization over the pool's mean
// squares), each neuron's step and activation. The sweeps of a settle are one submission.
// Float32 throughout: WGSL has no float64.

const WG = 64;

const SHADER = /* wgsl */ `
struct Shape { lives: u32, n: u32, edges: u32, pools: u32,
               dt: f32, slope: f32, threshold: f32, rest: f32,
               up: f32, down: f32, fraction: f32, floor2: f32,
               tiny: f32, apical: u32, basal_at: u32, apical_at: u32,
               pool_of_at: u32, pool_start_at: u32, member_at: u32, unused: u32 }
@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read> weight: array<f32>;     // (lives, edges)
@group(0) @binding(2) var<storage, read> index: array<u32>;      // pre | basal | apical | pools
@group(0) @binding(3) var<storage, read> standing: array<f32>;   // (lives, n)
@group(0) @binding(4) var<storage, read_write> v: array<f32>;    // (lives, n)
@group(0) @binding(5) var<storage, read_write> s: array<f32>;    // (lives, n)
@group(0) @binding(6) var<storage, read_write> work: array<f32>; // basal | raw | gain

@compute @workgroup_size(${WG})
fn inputs(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = id.x;
  let cells = shape.lives * shape.n;
  if (k >= cells) { return; }
  let life = k / shape.n;
  let i = k % shape.n;
  let w0 = life * shape.edges;
  let s0 = life * shape.n;
  var b = 0.0;
  for (var e = index[shape.basal_at + i]; e < index[shape.basal_at + i + 1u]; e++) {
    b += weight[w0 + e] * s[s0 + index[e]];
  }
  var r = 0.0;
  for (var e = index[shape.apical_at + i]; e < index[shape.apical_at + i + 1u]; e++) {
    r += weight[w0 + e] * s[s0 + index[e]];
  }
  work[k] = b + standing[k];
  work[cells + k] = r;
}

@compute @workgroup_size(${WG})
fn gains(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = id.x;
  if (k >= shape.lives * shape.pools) { return; }
  let cells = shape.lives * shape.n;
  let life = k / shape.pools;
  let p = k % shape.pools;
  let s0 = life * shape.n;
  var b2 = 0.0;
  var r2 = 0.0;
  let lo = index[shape.pool_start_at + p];
  let hi = index[shape.pool_start_at + p + 1u];
  for (var q = lo; q < hi; q++) {
    let i = index[shape.member_at + q];
    let b = work[s0 + i];
    let r = work[cells + s0 + i];
    b2 += b * b;
    r2 += r * r;
  }
  let m = f32(hi - lo);
  b2 = b2 / m;
  r2 = r2 / m;
  work[2u * cells + k] = shape.fraction * sqrt(b2) / sqrt(r2 + shape.floor2 * b2 + shape.tiny);
}

@compute @workgroup_size(${WG})
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = id.x;
  let cells = shape.lives * shape.n;
  if (k >= cells) { return; }
  let life = k / shape.n;
  let i = k % shape.n;
  var x = work[k];
  let p = index[shape.pool_of_at + i];
  if (shape.apical != 0u && p != 0xffffffffu) {
    x += work[2u * cells + life * shape.pools + p] * work[cells + k];
  }
  let vn = v[k] + (x - v[k]) * shape.dt;
  v[k] = vn;
  let q = 1.0 / (1.0 + exp(-shape.slope * (vn - shape.threshold))) - shape.rest;
  s[k] = select(q * shape.down, q * shape.up, q > 0.0);
}
`;

export class GpuLives {
  static async create() {
    if (!navigator.gpu) throw new Error('no WebGPU here');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no WebGPU adapter');
    const device = await adapter.requestDevice({
      requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                        maxBufferSize: adapter.limits.maxBufferSize } });
    const errors = [];
    device.onuncapturederror = (e) => errors.push(e.error.message);
    const gpu = new GpuLives(device, adapter);
    gpu.errors = errors;
    return gpu;
  }

  constructor(device, adapter) {
    this.device = device;
    this.adapter = adapter;
    const module = device.createShaderModule({ code: SHADER });
    const layout = device.createBindGroupLayout({ entries: [...Array(7).keys()].map((b) => ({
      binding: b, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: b === 0 ? 'uniform' : (b >= 4 ? 'storage' : 'read-only-storage') } })) });
    const pipeline = (entryPoint) => device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint } });
    this.layout = layout;
    this.pipelines = { inputs: pipeline('inputs'), gains: pipeline('gains'), update: pipeline('update') };
  }

  buffer(data, usage) {
    const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
    const b = this.device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
    new data.constructor(b.getMappedRange(0, data.byteLength)).set(data);
    b.unmap();
    return b;
  }

  // anatomy: { n, pre (u32, edges), basalStart, apicalStart (u32, n + 1: a neuron's basal synapses,
  // then its apical ones, in edge order), poolOf (i32, n; -1 none), poolStart (u32, pools + 1),
  // poolMember (u32), rule: { dt, slope, threshold, rest, up, down },
  // apical: { fraction, floor2, tiny } or null }; weights: Float32Array (lives * edges)
  load(anatomy, lives, weights) {
    const S = GPUBufferUsage.STORAGE;
    const n = anatomy.n, edges = anatomy.pre.length;
    const pools = Math.max(1, anatomy.poolStart.length - 1);
    const parts = [anatomy.pre, anatomy.basalStart, anatomy.apicalStart,
                   new Uint32Array(Int32Array.from(anatomy.poolOf).buffer), anatomy.poolStart, anatomy.poolMember];
    const at = [0];
    for (const p of parts) at.push(at[at.length - 1] + p.length);
    const index = new Uint32Array(at[at.length - 1] + 1);
    parts.forEach((p, k) => index.set(p, at[k]));
    const shape = new ArrayBuffer(80);
    const u = new Uint32Array(shape), f = new Float32Array(shape);
    const r = anatomy.rule, a = anatomy.apical || { fraction: 1, floor2: 1, tiny: 0 };
    u[0] = lives; u[1] = n; u[2] = edges; u[3] = anatomy.poolStart.length - 1;
    f[4] = r.dt; f[5] = r.slope; f[6] = r.threshold; f[7] = r.rest;
    f[8] = r.up; f[9] = r.down; f[10] = a.fraction; f[11] = a.floor2;
    f[12] = a.tiny; u[13] = anatomy.apical ? 1 : 0; u[14] = at[1]; u[15] = at[2];
    u[16] = at[3]; u[17] = at[4]; u[18] = at[5];
    this.shape = { lives, n, edges, pools };
    const cells = lives * n;
    const state = (count) => this.device.createBuffer({ size: Math.max(16, count * 4), usage: S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.buffers = {
      shape: this.buffer(new Uint32Array(shape), GPUBufferUsage.UNIFORM),
      weight: this.buffer(weights, S), index: this.buffer(index, S),
      standing: state(cells), v: state(cells), s: state(cells), work: state(2 * cells + lives * pools),
      read: this.device.createBuffer({ size: Math.max(16, cells * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    };
    const b = this.buffers;
    this.group = this.device.createBindGroup({ layout: this.layout, entries: [b.shape, b.weight, b.index,
      b.standing, b.v, b.s, b.work].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.apical = Boolean(anatomy.apical);
  }

  // a buffer's contents as float32, for checks: 'standing', 'v', 's' or 'work'
  async read(name) {
    const src = this.buffers[name];
    const out = this.device.createBuffer({ size: src.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, out, 0, src.size);
    this.device.queue.submit([enc.finish()]);
    await out.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(out.getMappedRange().slice(0));
    out.unmap();
    out.destroy();
    return data;
  }

  // one settle of every life from (v, s) under `standing` (Float32Array, lives * n each) for `sweeps`
  // sweeps; resolves to the activations, Float32Array (lives * n)
  async settle(standing, v, s, sweeps) {
    const q = this.device.queue, b = this.buffers;
    q.writeBuffer(b.standing, 0, standing);
    q.writeBuffer(b.v, 0, v);
    q.writeBuffer(b.s, 0, s);
    const { lives, n, pools } = this.shape;
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.group);
    const cells = Math.ceil((lives * n) / WG), pooled = Math.ceil((lives * pools) / WG);
    for (let t = 0; t < sweeps; t++) {
      pass.setPipeline(this.pipelines.inputs); pass.dispatchWorkgroups(cells);
      if (this.apical) { pass.setPipeline(this.pipelines.gains); pass.dispatchWorkgroups(pooled); }
      pass.setPipeline(this.pipelines.update); pass.dispatchWorkgroups(cells);
    }
    pass.end();
    enc.copyBufferToBuffer(b.s, 0, b.read, 0, lives * n * 4);
    q.submit([enc.finish()]);
    await b.read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(b.read.getMappedRange().slice(0));
    b.read.unmap();
    return out;
  }
}
