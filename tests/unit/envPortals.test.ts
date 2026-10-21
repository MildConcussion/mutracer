import { describe, expect, it, vi } from 'vitest';
import {
  ENV_PORTAL_MISS_SCALE,
  ENV_PORTAL_GUIDE_SHARE,
  ENV_PORTAL_SHARE,
  ENV_PORTAL_SUN_ROULETTE,
  ENV_PORTALS_UNIFORM_BYTES,
  MAX_ENV_PORTALS,
  packEnvPortals
} from '../../src/renderer/envPortals';
import EnvironmentLight, { type EnvironmentPortal } from '../../src/scene/EnvironmentLight';
import Texture from '../../src/scene/Texture';
import sampleLight from '../../src/renderer/wgsl/chunks/sampleLight.wgsl';
import envPortalsWgsl from '../../src/renderer/wgsl/chunks/envPortals.wgsl';

const window: EnvironmentPortal = { corner: [1, 0, 2], edgeU: [2, 0, 0], edgeV: [0, 1.5, 0] };
type Vec3 = [number, number, number];
const region = { corner: [-3, -1, -3] as Vec3, edges: [[6, 0, 0], [0, 4, 0], [0, 0, 6]] as [Vec3, Vec3, Vec3] };
// The same room box turned 35 degrees about the vertical, as a rotated building's room is
const turn = (35 * Math.PI) / 180;
const turned = {
  corner: [1, 0, 2] as Vec3,
  edges: [[4 * Math.cos(turn), 0, -4 * Math.sin(turn)], [0, 3, 0], [2 * Math.sin(turn), 0, 2 * Math.cos(turn)]] as [Vec3, Vec3, Vec3]
};
/** A point's box coordinates from the packed corner and axes, as inEnvPortalRegion computes them */
const local = (data: Float32Array, p: Vec3) =>
  [8, 12, 16].map((at) => [0, 1, 2].reduce((sum, k) => sum + data[at + k] * (p[k] - data[4 + k]), 0));
const inside = (data: Float32Array, p: Vec3) => local(data, p).every((value) => value >= 0 && value <= 1);

describe('environment portals', () => {
  it('packs nothing without portals, so the shader keeps sampling the map alone', () => {
    const data = new Float32Array(ENV_PORTALS_UNIFORM_BYTES / 4).fill(7);
    expect(packEnvPortals(data, undefined)).toBe(0);
    expect(data.every((value) => value === 0)).toBe(true);
  });

  it('packs the header, region and each rectangle with its area', () => {
    const data = new Float32Array(ENV_PORTALS_UNIFORM_BYTES / 4);
    expect(packEnvPortals(data, { rectangles: [window], region }, { tableOffset: 3, choiceCount: 2, pmfScale: 4 })).toBe(1);
    expect([...data.subarray(0, 4)]).toEqual([1, ENV_PORTAL_SHARE, ENV_PORTAL_MISS_SCALE, 1].map(Math.fround));
    expect([...data.subarray(4, 8)]).toEqual([...region.corner, ENV_PORTAL_GUIDE_SHARE]);
    expect([...data.subarray(8, 20)]).toEqual([1 / 6, 0, 0, 0, 0, 1 / 4, 0, 0, 0, 0, 1 / 6, 0].map(Math.fround));
    expect([...data.subarray(20, 24)]).toEqual([3, 2, 4, ENV_PORTAL_SUN_ROULETTE]);
    expect([...data.subarray(24, 36)]).toEqual([1, 0, 2, 3, 2, 0, 0, 0, 0, 1.5, 0, 0]);
    packEnvPortals(data, { rectangles: [window], region });
    expect(data[3]).toBe(0);
    expect([...data.subarray(20, 24)]).toEqual([0, 0, 0, 0]);
  });

  it('keeps a turned room box to itself: its corners in, the world box around it out', () => {
    const data = new Float32Array(ENV_PORTALS_UNIFORM_BYTES / 4);
    expect(packEnvPortals(data, { rectangles: [window], region: turned })).toBe(1);
    const [a, b, c] = turned.edges;
    const at = (s: number, t: number, u: number): Vec3 => [0, 1, 2].map((k) => turned.corner[k] + s * a[k] + t * b[k] + u * c[k]) as Vec3;
    for (const [s, t, u] of [[0.001, 0.001, 0.001], [0.999, 0.999, 0.999], [0.5, 0.5, 0.5], [0.999, 0.001, 0.999]]) {
      expect(inside(data, at(s, t, u))).toBe(true);
    }
    // Inside the world-axis box of the turned box, outside the box itself (a terrace beside the room)
    expect(inside(data, at(1.05, 0.5, 0.5))).toBe(false);
    expect(inside(data, at(-0.05, 0.5, 0.5))).toBe(false);
    expect(inside(data, [turned.corner[0] + 0.1, 1.5, 3.5])).toBe(false);
    expect(local(data, at(0.25, 0.5, 0.75)).map((value) => Number(value.toFixed(5)))).toEqual([0.25, 0.5, 0.75]);
  });

  it('packs nothing for a region that spans no volume', () => {
    const data = new Float32Array(ENV_PORTALS_UNIFORM_BYTES / 4);
    const flat = { corner: [0, 0, 0] as Vec3, edges: [[1, 0, 0], [0, 1, 0], [2, 2, 0]] as [Vec3, Vec3, Vec3] };
    expect(packEnvPortals(data, { rectangles: [window], region: flat })).toBe(0);
    expect(data.every((value) => value === 0)).toBe(true);
  });

  it('skips degenerate rectangles and keeps at most the uniform capacity', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const data = new Float32Array(ENV_PORTALS_UNIFORM_BYTES / 4);
    const flat: EnvironmentPortal = { corner: [0, 0, 0], edgeU: [1, 0, 0], edgeV: [2, 0, 0] };
    expect(packEnvPortals(data, { rectangles: [flat], region })).toBe(0);
    expect(data[0]).toBe(0);
    expect(packEnvPortals(data, { rectangles: Array(MAX_ENV_PORTALS + 3).fill(window), region })).toBe(MAX_ENV_PORTALS);
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });

  it('keeps an environment light without portals as it was', () => {
    const map = new Texture({ data: new Float32Array(4), width: 1, height: 1 });
    expect('portals' in new EnvironmentLight({ map })).toBe(false);
    expect(new EnvironmentLight({ map, portals: { rectangles: [window], region } }).portals?.rectangles).toEqual([window]);
  });

  it('samples the environment through the portals only where the shader has them', () => {
    expect(sampleLight).toMatch(/if \(HAS_ENV_PORTALS\) \{\s*sampleEnvLightFrom\(scatterPos, rnd\.xy, lightSample\);/);
    expect(envPortalsWgsl).toContain('let d = p - envPortals.regionCorner.xyz;');
    expect(envPortalsWgsl).toContain(`array<vec4f, ${3 * MAX_ENV_PORTALS}>`);
  });
});
