// Open /tests/gpu/thinPane.html (?webgl for WebGL 2) on a real GPU.
// A black room with a window, glazed by a closed 24 mm pane (two faces along a
// ray) of smooth thin-walled glass, which MuTracer treats as a thin pane:
// - sun: a floor patch the sun reaches through the window, against the open
//   window times the pane's transmittance at the sun's angle (shadow rays);
// - view: an unlit panel seen through the window, against the open window
//   times the transmittance at normal incidence (camera rays passing);
// - sky: a floor patch lit by an emissive panel through the window, against
//   the same glass at specularRoughness 1e-4, which is no thin pane and takes
//   the full BSDF path (light reached behind the pane, MIS);
// - clear: a grey room lit by a sky through a pane of IOR 1 and no tint, which
//   lets everything through, against the open window: the path's last BSDF
//   ray still passes the pane to the sky its NEE weighed itself against;
// - mis (WebGPU): the same room through the glass, at one light candidate,
//   at four, and at four with the window as an environment portal: the
//   estimates weigh NEE and BSDF light differently and agree.
import {Camera, DirectionalLight, Engine, EnvironmentLight, Mesh, OpenPBRMaterial, Render, Texture, createCubeGeometry} from '../../src/main.js';
import type {Vec3} from '../../src/main.js';

const IOR = 1.52;
const TINT = 0.9;

// Each box its own geometry: instances of one share triangle indices, and a
// ray skips its origin's triangle index in every instance.
function box(min: Vec3, max: Vec3, material: OpenPBRMaterial) {
  const mesh = new Mesh(createCubeGeometry(), material);
  for (let axis = 0; axis < 3; axis++) {
    // The cube spans -0.5..0.5.
    mesh.transform[axis * 5] = max[axis] - min[axis];
    mesh.transform[12 + axis] = (max[axis] + min[axis]) / 2;
  }
  return mesh;
}

/** One face's straight transmission: the thin-wall response at this cosine. */
function faceTransmission(cosine: number) {
  const sine = Math.sqrt(1 - cosine * cosine), cosT = Math.sqrt(1 - (sine / IOR) ** 2);
  const s = ((cosine - IOR * cosT) / (cosine + IOR * cosT)) ** 2;
  const p = ((cosT - IOR * cosine) / (cosT + IOR * cosine)) ** 2;
  const fresnel = (s + p) / 2;
  return TINT ** (1 / cosT) * (1 - 2 * fresnel / (1 + fresnel));
}

function patchMean(data: Float32Array, width: number, height: number, half: number) {
  let sum = 0, count = 0;
  for (let y = height / 2 - half; y < height / 2 + half; y++) {
    for (let x = width / 2 - half; x < width / 2 + half; x++) {
      const i = (y * width + x) * 4;
      sum += 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
      count++;
    }
  }
  return sum / count;
}

function frameMean(data: Float32Array) {
  let sum = 0;
  for (let i = 0; i < data.length; i += 4) sum += 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
  return sum / (data.length / 4);
}

/** A sky: brighter toward the zenith, a bright band low on one side, dark ground below the horizon. */
function skyMap() {
  const width = 128, height = 64;
  const data = new Float32Array(width * height * 4);
  for (let row = 0; row < height; row++) {
    const elevation = Math.PI / 2 - (row + 0.5) / height * Math.PI;
    for (let column = 0; column < width; column++) {
      const azimuth = (column + 0.5) / width * 2 * Math.PI;
      let value = elevation > 0 ? 0.6 + 0.8 * Math.sin(elevation) : 0.04;
      if (elevation > 0 && elevation < 0.4 && Math.cos(azimuth - Math.PI) > 0.7) value *= 3;
      data.set([value * 0.9, value, value * 1.15, 1], (row * width + column) * 4);
    }
  }
  return new Texture({data, width, height});
}

export async function capture() {
  const canvas = document.createElement('canvas');
  document.body.append(canvas);
  const engine = new Engine({canvas, backend: location.search.includes('webgl') ? 'webgl' : 'webgpu'});
  const size = 64;
  const render = new Render(engine, {width: size, height: size, bounces: 4, bevel: false});
  engine.display.setSize(size, size);
  const black = new OpenPBRMaterial({baseColor: [0, 0, 0], specularWeight: 0});
  const floor = new OpenPBRMaterial({baseColor: [0.5, 0.5, 0.5], specularWeight: 0});
  const glass = (roughness: number) => new OpenPBRMaterial({
    thinWalled: true, transmissionWeight: 1, transmissionColor: [TINT, TINT, TINT],
    specularRoughness: roughness, baseMetalness: 0, specularIor: IOR
  });
  // Room x -2..2, y 0..2.5, z -2..2; the window x -0.8..0.8, y 0.6..2 in the wall at z -2.2..-2.
  const room = (floorMaterial: OpenPBRMaterial, walls = black) => [
    box([-2.2, -0.2, -2.2], [2.2, 0, 2.2], floorMaterial),
    box([-2.2, 2.5, -2.2], [2.2, 2.7, 2.2], walls),
    box([-2.2, 0, 2], [2.2, 2.5, 2.2], walls),
    box([-2.2, 0, -2.2], [-2, 2.5, 2.2], walls),
    box([2, 0, -2.2], [2.2, 2.5, 2.2], walls),
    box([-2.2, 0, -2.2], [-0.8, 2.5, -2], walls),
    box([0.8, 0, -2.2], [2.2, 2.5, -2], walls),
    box([-0.8, 0, -2.2], [0.8, 0.6, -2], walls),
    box([-0.8, 2, -2.2], [0.8, 2.5, -2], walls)
  ];
  const pane = (roughness: number) => box([-0.8, 0.6, -2.112], [0.8, 2, -2.088], glass(roughness));
  const down = (target: Vec3) => {
    const camera = new Camera({projection: 'perspective', fov: 20, aspect: 1});
    camera.position = [target[0], 2.4, target[2]]; camera.lookAt(target, [0, 0, 1]);
    return camera;
  };
  const results = [];
  try {
    const info = await engine.init();
    const measure = async (meshes: Mesh[], lights: DirectionalLight[], camera: Camera, samples: number) => {
      engine.scene = {meshes, lights, background: [0, 0, 0, 1]};
      engine.camera = camera;
      engine.update();
      await render.converge({samples, sampleSequenceOffset: 17});
      return patchMean((await render.readPixels('hdr')).data, size, size, 8);
    };

    // The sun from (0, 3, -6) toward (0, 0.5, 0): through the window onto the floor round z 0.5.
    const sun = new DirectionalLight({intensity: 3});
    sun.position = [0, 3, -6]; sun.lookAt([0, 0.5, 0]);
    const sunCosine = 6 / Math.hypot(2.5, 6);
    {
      const open = await measure(room(floor), [sun], down([0, 0, 0.5]), 16);
      const glazed = await measure([...room(floor), pane(0)], [sun], down([0, 0, 0.5]), 16);
      const expected = faceTransmission(sunCosine) ** 2;
      const ratio = glazed / open;
      results.push({name: 'sun', open, glazed, ratio, expected, pass: open > 0 && Math.abs(ratio / expected - 1) < 0.005});
    }

    // An unlit panel 2 m beyond the window, seen straight through it.
    {
      const panel = box([-4, -2, -4.2], [4, 5, -4], new OpenPBRMaterial({
        unlit: true, baseWeight: 0, specularWeight: 0, emissionColor: [0.5, 0.6, 0.7], emissionLuminance: 1
      }));
      const camera = new Camera({projection: 'perspective', fov: 10, aspect: 1});
      camera.position = [0, 1.3, 1]; camera.lookAt([0, 1.3, -4]);
      const open = await measure([...room(black), panel], [], camera, 256);
      const glazed = await measure([...room(black), panel, pane(0)], [], camera, 256);
      const expected = faceTransmission(1) ** 2;
      const ratio = glazed / open;
      results.push({name: 'view', open, glazed, ratio, expected, pass: open > 0 && Math.abs(ratio / expected - 1) < 0.02});
    }

    // An emissive panel 4 m beyond the window lighting the floor inside it.
    {
      const sky = box([-6, -2, -6.2], [6, 8, -6], new OpenPBRMaterial({
        baseColor: [0, 0, 0], specularWeight: 0, emissionColor: [1, 1, 1], emissionLuminance: 2
      }));
      const target: Vec3 = [0, 0, -1.2];
      const open = await measure([...room(floor), sky], [], down(target), 1024);
      const generic = await measure([...room(floor), sky, pane(1e-4)], [], down(target), 1024);
      const glazed = await measure([...room(floor), sky, pane(0)], [], down(target), 1024);
      const ratio = glazed / generic;
      results.push({name: 'sky', open, generic, glazed, ratio, expected: 1, pass: generic > 0 && Math.abs(ratio - 1) < 0.03});
    }
    // A grey room under a sky, its window glazed, seen from the ceiling down onto the floor, ground outside.
    {
      const grey = new OpenPBRMaterial({baseColor: [0.6, 0.6, 0.6], specularWeight: 0});
      const ground = box([-30, -0.4, -40], [30, -0.2, -2.2], new OpenPBRMaterial({baseColor: [0.3, 0.32, 0.25], specularWeight: 0}));
      const clear = box([-0.8, 0.6, -2.112], [0.8, 2, -2.088], new OpenPBRMaterial({
        thinWalled: true, transmissionWeight: 1, transmissionColor: [1, 1, 1], specularRoughness: 0, baseMetalness: 0, specularIor: 1
      }));
      const camera = new Camera({projection: 'perspective', fov: 60, aspect: 1});
      camera.position = [0, 2.4, 0.4]; camera.lookAt([0, 0, 0.4], [0, 0, -1]);
      const map = skyMap();
      const sky = new EnvironmentLight({map});
      const portal = new EnvironmentLight({map, portals: {
        rectangles: [{corner: [-0.8, 0.6, -2], edgeU: [1.6, 0, 0], edgeV: [0, 1.4, 0]}],
        region: {corner: [-2.05, -0.05, -2.05], edges: [[4.1, 0, 0], [0, 2.6, 0], [0, 0, 4.1]]}
      }});
      const lit = async (meshes: Mesh[], light: EnvironmentLight, candidates: number) => {
        engine.scene = {meshes, lights: [light]};
        engine.camera = camera;
        render.directLightCandidates = candidates;
        engine.update();
        await render.converge({samples: 4096, sampleSequenceOffset: 17});
        return frameMean((await render.readPixels('hdr')).data);
      };
      const open = await lit([...room(grey, grey), ground], sky, 1);
      const passed = await lit([...room(grey, grey), ground, clear], sky, 1);
      results.push({name: 'clear', open, glazed: passed, ratio: passed / open, expected: 1, pass: open > 0 && Math.abs(passed / open - 1) < 0.004});
      if (engine.backend === 'webgpu') {
        const glazed = [...room(grey, grey), ground, pane(0)];
        const one = await lit(glazed, sky, 1);
        const four = await lit(glazed, sky, 4);
        const portals = await lit(glazed, portal, 4);
        const ratio = Math.min(one, four) / portals, ratioMax = Math.max(one, four) / portals;
        results.push({name: 'mis', one, four, portals, ratio, ratioMax, expected: 1, pass: portals > 0 && Math.abs(ratio - 1) < 0.004 && Math.abs(ratioMax - 1) < 0.004});
      }
    }
    return {backend: engine.backend, adapter: info.adapterName, results, pass: results.every((result) => result.pass)};
  } finally { render.dispose(); engine.dispose(); }
}

const result = document.querySelector('#result')!;
capture().then((value) => {
  result.textContent = JSON.stringify(value, null, 2);
  document.body.dataset.done = 'true';
}, (error) => {
  result.textContent = `${error}\n${error.stack ?? ''}`;
  document.body.dataset.done = 'true';
});
