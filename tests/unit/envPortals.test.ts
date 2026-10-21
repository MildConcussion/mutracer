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
const region = { min: [-3, -1, -3] as [number, number, number], max: [3, 3, 3] as [number, number, number] };

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
    expect([...data.subarray(4, 8)]).toEqual([...region.min, ENV_PORTAL_GUIDE_SHARE]);
    expect([...data.subarray(8, 11)]).toEqual(region.max);
    expect([...data.subarray(12, 16)]).toEqual([3, 2, 4, ENV_PORTAL_SUN_ROULETTE]);
    expect([...data.subarray(16, 28)]).toEqual([1, 0, 2, 3, 2, 0, 0, 0, 0, 1.5, 0, 0]);
    packEnvPortals(data, { rectangles: [window], region });
    expect(data[3]).toBe(0);
    expect([...data.subarray(12, 16)]).toEqual([0, 0, 0, 0]);
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
    expect(envPortalsWgsl).toContain('return HAS_ENV_PORTALS && all(p >= envPortals.regionMin.xyz)');
    expect(envPortalsWgsl).toContain(`array<vec4f, ${3 * MAX_ENV_PORTALS}>`);
  });
});
